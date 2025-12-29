from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass
import json
import time
from langdetect import detect
import os
from langchain_core.retrievers import BaseRetriever
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------
# ENV & CONFIG
# ---------------------------
load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
deployment_name_fallback = os.getenv("AZURE_OPENAI_DEPLOYMENT_FALLBACK")
api_version_fallback = os.getenv("AZURE_OPENAI_API_VERSION_FALLBACK")

# --------------------------------------------------------------
# 1. ЗАГРУЗКА ГЛАВНОГО ИНДЕКСА
# --------------------------------------------------------------
with open("output1.json", 'r', encoding='utf-8') as f:
    index_data = json.load(f)

# --------------------------------------------------------------
# 2. EMBEDDINGS
# --------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# --------------------------------------------------------------
# 3. СОЗДАЁМ ДОКУМЕНТЫ ИЗ САММАРИ
# --------------------------------------------------------------
summary_documents = []
for i, chunk_info in enumerate(index_data["chunks"]):
    content = f"{chunk_info['summary']}\n" + "\n".join(chunk_info.get('questions', []))
    summary_documents.append(
        Document(
            page_content=content,
            metadata={
                "file": chunk_info["file"],
                "summary": chunk_info["summary"],
                "chunk_id": i + 1
            }
        )
    )

# ---------------------------
# BM25 подготовка
# ---------------------------
documents_texts = [doc.page_content for doc in summary_documents]
vectorizer = TfidfVectorizer()
X_sparse = vectorizer.fit_transform(documents_texts)

# --------------------------------------------------------------
# 4. SETUP QDRANT (основной индекс саммари)
# --------------------------------------------------------------
client = QdrantClient(url='http://localhost:6333', port=6333)
collection_name = "summaries"

summary_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    summary_store.add_documents(summary_documents)

# --------------------------------------------------------------
# SEMANTIC CACHE SETUP
# --------------------------------------------------------------
cache_collection_name = "rag_semantic_cache1"

if not client.collection_exists(cache_collection_name):
    client.create_collection(
        collection_name=cache_collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

cache_vector_store = QdrantVectorStore(
    client=client,
    collection_name=cache_collection_name,
    embedding=embeddings,
)

# --------------------------------------------------------------
# SEMANTIC CACHE CLASS
# --------------------------------------------------------------
class RAGSemanticCache:
    def __init__(self, vector_store, threshold: float = 0.3):
        self.vector_store = vector_store
        self.threshold = threshold

    def retrieve_cached_response(self, query: str):
        """Возвращает Document с кэшированным ответом или None"""
        query_lang = detect(query)
        results = self.vector_store.similarity_search_with_score(
            query,
            k=1,
            score_threshold=self.threshold
        )
        if results:
            best, score = results[0]
            if best.metadata.get("language") == query_lang:
                return best

    def store_response(self, query: str, response: str, tokens: int):
        language = detect(query)
        doc = Document(
            page_content=query,  # эмбеддится именно запрос
            metadata={
                "response": response,
                "tokens": tokens,
                "language": language
            }
        )
        self.vector_store.add_documents([doc])

semantic_cache = RAGSemanticCache(cache_vector_store, threshold=0.3)  # можно менять порог

# --------------------------------------------------------------
# 5. FLASHRANK RERANKER
# --------------------------------------------------------------
FlashrankRerank.model_rebuild()
compressor = FlashrankRerank(
    model="ms-marco-MiniLM-L-12-v2",
    top_n=25
)

# --------------------------------------------------------------
# 6. AZURE OPENAI LLM
# --------------------------------------------------------------
llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=deployment_name,
    api_version=api_version,
    api_key=api_key,
    temperature=1,
    max_retries=3,
    max_tokens=3000,
    timeout=30.0
)

fallback_llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=deployment_name_fallback,
    api_version=api_version_fallback,
    api_key=api_key,
    temperature=1,
    max_retries=3,
    max_tokens=3000,
    timeout=30.0
)

# --------------------------------------------------------------
# 7. COST STATS
# --------------------------------------------------------------
@dataclass
class CostStats:
    llm_calls: int = 0
    cache_hits: int = 0
    spent_tokens: int = 0
    saved_tokens: int = 0

stats = CostStats()

# --------------------------------------------------------------
# 8. Гибридный поиск BM25 + Qdrant
# --------------------------------------------------------------
def hybrid_summary_search(query, top_k=30, category=None):
    # --- BM25 ---
    query_vec = vectorizer.transform([query])
    bm25_scores = (X_sparse @ query_vec.T).toarray().flatten()
    bm25_top_idx = bm25_scores.argsort()[::-1][:top_k]
    bm25_docs = [summary_documents[i] for i in bm25_top_idx]

    # --- Qdrant semantic search ---
    query_embedding = embeddings.embed_query(query)
    filter_condition = None
    if category:
        filter_condition = Filter(
            must=[
                FieldCondition(
                    key="metadata.category",
                    match=MatchValue(value=category)
                )
            ]
        )

    search_result = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
        query_filter=filter_condition,
    )

    qdrant_docs = []
    for hit in search_result.points:
        file_name = hit.payload.get("file")
        for doc in summary_documents:
            if doc.metadata["file"] == file_name:
                qdrant_docs.append(doc)
                break

    # --- Объединяем и уникализируем по файлу ---
    combined_docs = bm25_docs + qdrant_docs
    seen = set()
    final_docs = []
    for doc in combined_docs:
        file_key = doc.metadata["file"]
        if file_key not in seen:
            final_docs.append(doc)
            seen.add(file_key)

    return final_docs[:top_k]

# --------------------------------------------------------------
# 9. RAG ФУНКЦИЯ
# --------------------------------------------------------------
def answer_query(query: str):
    # Этап 1: гибридный retrieval
    summary_docs = hybrid_summary_search(query, top_k=30)
    selected_files = [doc.metadata["file"] for doc in summary_docs]

    # Этап 2: загрузка детальных файлов
    chunks_dir = Path("chunks")
    detailed_docs = []
    for file_name in selected_files:
        file_path = chunks_dir / file_name
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                detailed_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": file_name,
                            "type": "detailed_chunk"
                        }
                    )
                )

    if not detailed_docs:
        return "No relevant information found.", [], selected_files

    # Этап 3: Rerank через FlashRank
    class SimpleRetriever(BaseRetriever):
        docs: list
        def _get_relevant_documents(self, query: str, **kwargs):
            return self.docs

    temp_retriever = SimpleRetriever(docs=detailed_docs)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=temp_retriever
    )
    reranked_docs = compression_retriever.invoke(query)

    # Этап 4: формируем контекст
    context = "\n\n".join(doc.page_content for doc in reranked_docs)

    # --------------------------------------------------
    # Проверка семантического кэша
    # --------------------------------------------------
    cached_doc = semantic_cache.retrieve_cached_response(query)
    if cached_doc:
        stats.cache_hits += 1
        cached_tokens = cached_doc.metadata.get("tokens", 0)
        stats.saved_tokens += cached_tokens
        print("From Semantic Cache")
        return cached_doc.metadata["response"], reranked_docs, selected_files

    # Если нет в кэше — идём в LLM
    print("Request to Azure OpenAI")
    stats.llm_calls += 1

    messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly and professional AI assistant for Birmarket customer support (an online marketplace in Azerbaijan).\n\n"

                "MAIN RULES:\n"
                "1. Answer ONLY based on the provided context if the question is about products, delivery, payment, returns, bonuses, customs, the Birmarket app, website, or related services.\n"
                "2. If the context does not contain the information — honestly say: 'Unfortunately, I don't have exact information on this question. I recommend contacting our support team at ... or via the chat in the app.'\n"
                "3. NEVER invent facts about Birmarket, prices, terms, product availability and etc.\n\n"

                "ALLOWED GENERAL QUESTIONS (answer them naturally and kindly):\n"
                "- Greetings (hi, salam, hello, good day, etc.)\n"
                "- Questions about the language of communication ('Can I ask in Russian?', 'In English?', 'Azərbaycanca?')\n"
                "- Thanks ('thank you', 'təşəkkür')\n"
                "- Questions like 'How are you?', 'What can you do?'\n"
                "- Requests to repeat or clarify\n\n"

                "IMPORTANT:\n"
                "- Determine the user's language from their message and reply in the same language.\n"
                "- The user's question may be in Russian, Azerbaijani or other languages.\n"
                "- You MUST understand the Azerbaijani context and answer in the language of the user's question.\n"
                "- ALWAYS translate relevant facts from the Azerbaijani context into the user's language accurately.\n"
                "- If there is relevant information in the context (even if it's in Azerbaijani), use it and translate the answer.\n"
                "- Be polite, positive, and concise.\n"
                "- You are NOT a universal GPT. If the question clearly goes beyond Birmarket support (politics, weather, programming, personal advice, etc.) — politely redirect: 'Sorry, I specialize only in helping with purchases and Birmarket services. For other questions, I recommend contacting other services.'\n"
                "- Do not discuss your model, training, xAI, Grok, etc."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]

    try:
        response = llm.invoke(messages)
    except Exception:
        response = fallback_llm.invoke(messages)

    # Подсчёт токенов
    usage = response.response_metadata.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens
    if total_tokens == 0:
        total_tokens = int(len((context + query + response.content).split()) * 1.3)

    stats.spent_tokens += total_tokens

    # --------------------------------------------------
    # Сохранение в семантический кэш
    # --------------------------------------------------
    semantic_cache.store_response(query, response.content, total_tokens)

    return response.content, reranked_docs, selected_files

# --------------------------------------------------------------
# 10. RUN QUERY & METRICS
# --------------------------------------------------------------
query = "BirBonus nedir????"

start = time.perf_counter()
response, docs, selected_files = answer_query(query)
elapsed = time.perf_counter() - start

print("\n" + "=" * 50)
print("SELECTED FILES:", docs)
print("=" * 50)
print("\n========== ANSWER ==========")
print(response)
print(f"\nElapsed time: {elapsed:.2f} sec")

# Статистика использования
print("\n--- STATS ---")
print(f"LLM calls: {stats.llm_calls}")
print(f"Cache hits: {stats.cache_hits}")
print(f"Spent tokens: {stats.spent_tokens}")
print(f"Saved tokens: {stats.saved_tokens}")