from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from qdrant_client.models import VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.retrievers import BaseRetriever
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from dataclasses import dataclass
from dotenv import load_dotenv
from diskcache import Cache
from pathlib import Path
from typing import List
import qdrant_client
import hashlib
import json
import time
import math
import os

cache = Cache("./llm_cache")

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
deployment_name_fallback = os.getenv("AZURE_OPENAI_DEPLOYMENT_FALLBACK")
api_version_fallback = os.getenv("AZURE_OPENAI_API_VERSION_FALLBACK")

# --------------------------------------------------------------
# 1. JSON LOADING
# --------------------------------------------------------------
with open("index.json", 'r', encoding='utf-8') as f:
    index_data = json.load(f)

# --------------------------------------------------------------
# 2. EMBEDDINGS
# --------------------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
# --------------------------------------------------------------
# 3. SUMMARY DOCS
# --------------------------------------------------------------
summary_documents = []
for i, chunk_info in enumerate(index_data["chunks"]):
    # Комбинируем саммари и вопросы
    content = f"{chunk_info['summary']}\n" + "\n".join(chunk_info.get('questions', []))

    summary_documents.append(
        Document(
            page_content=content,
            metadata={
                "file": chunk_info["file"],
                "summary": chunk_info["summary"],
                "chunk_id": i
            }
        )
    )
# --------------------------------------------------------------
# 4. SETUP QDRANT
# --------------------------------------------------------------
client = qdrant_client.QdrantClient(
    url="http://localhost:6333",
    api_key=None,
)

collection_name = "summaries"

# Добавляем в векторное хранилище
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

# Retriever для поиска файлов
summary_retriever = summary_store.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 20}  # Берём топ-3 файла
)

# --------------------------------------------------------------
# 5. FLASHRANK RERANKER
# --------------------------------------------------------------
FlashrankRerank.model_rebuild()
compressor = FlashrankRerank(
    model="ms-marco-MiniLM-L-12-v2",
    top_n=7
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

def make_cache_key(query: str, context: str) -> str:
    raw = (query.strip().lower() + context).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

# --------------------------------------------------------------
# 7. RAG FUNCTION
# --------------------------------------------------------------
def answer_query(query: str):
    # Этап 1: retrieve релевантные файлы через саммари
    summary_docs = summary_retriever.invoke(query)
    selected_files = [doc.metadata["file"] for doc in summary_docs]

    # Этап 2: загрузка содержимого выбранных файлов
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

    # Этап 5: кэш
    cache_key = make_cache_key(query, context)
    if cache_key in cache:
        cached = cache[cache_key]
        stats.cache_hits += 1
        stats.saved_tokens += cached["tokens"]
        print("From Cache:")
        return cached["response"], reranked_docs, selected_files

    # Этап 6: LLM вызов
    print("Request to Azure OpenAI")
    stats.llm_calls += 1
    messages = [
        {
            "role": "system",
            "content": (
                "You are the AI assistant of Birmarket support department.\n"
                "Answer ONLY using facts strictly from the provided context.\n"
                "If you not found any information say I don't know anything, call to our support.\n"
                "Do NOT invent information.\n"
                "Determine the user's language from their query.\n"
                "If the context is in another language, translate the answer to the user's language.\n"
                "Do NOT guess or invent information."
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

    usage = response.response_metadata.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens
    if total_tokens == 0:
        total_tokens = int(len((context + query + response.content).split()) * 1.3)

    stats.spent_tokens += total_tokens

    cache.set(
        cache_key,
        {
            "response": response.content,
            "tokens": total_tokens
        },
        expire=60 * 60 * 24
    )

    return response.content, reranked_docs, selected_files


# --------------------------------------------------------------
# 8. METRICS FUNCTIONS
# --------------------------------------------------------------

def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    hit_count = sum(1 for x in retrieved_k if x in relevant)
    return hit_count / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    retrieved_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    hit_count = sum(1 for x in retrieved_k if x in relevant)
    return hit_count / len(relevant) if relevant else 0.0


def mrr(retrieved_ids, relevant_ids):
    relevant = set(relevant_ids)
    for idx, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant:
            return 1 / idx
    return 0.0


def dcg_at_k(retrieved_ids, relevant_ids, k):
    dcg = 0.0
    relevant = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant:
            dcg += 1 / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved_ids, relevant_ids, k):
    ideal_dcg = dcg_at_k(relevant_ids, relevant_ids, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(retrieved_ids, relevant_ids, k) / ideal_dcg


def retrieve_chunk_ids_two_stage(query, summary_ret, comp, k=20):
    """Извлекает chunk_id из двухэтапного поиска"""
    # Этап 1: поиск через саммари
    summary_docs = summary_ret.invoke(query)
    selected_files = [doc.metadata["file"] for doc in summary_docs]

    # Этап 2: загрузка детальных файлов
    chunks_dir = Path("chunks")
    detailed_docs = []

    for file_name in selected_files:
        file_path = chunks_dir / file_name
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Извлекаем chunk_id из имени файла (например, chunk_001.txt -> 1)
                chunk_id = int(file_name.replace('chunk_', '').replace('.txt', ''))
                detailed_docs.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": file_name,
                            "chunk_id": chunk_id,
                            "type": "detailed_chunk"
                        }
                    )
                )

    if not detailed_docs:
        return []

    # Этап 3: Создаём временное векторное хранилище
    temp_collection = f"temp_eval_{hash(query) % 10000}"
    collections = [c.name for c in client.get_collections().collections]
    if temp_collection in collections:
        client.delete_collection(temp_collection)

    client.create_collection(
        collection_name=temp_collection,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )

    temp_store = QdrantVectorStore(
        client=client,
        collection_name=temp_collection,
        embedding=embeddings,
    )
    temp_store.add_documents(detailed_docs)

    temp_retriever = temp_store.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 20}
    )

    # Этап 4: rerank
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=comp,
        base_retriever=temp_retriever
    )

    reranked_docs = compression_retriever.invoke(query)

    # Удаляем временную коллекцию
    client.delete_collection(temp_collection)

    return [doc.metadata["chunk_id"] for doc in reranked_docs[:k]]


def evaluate_retriever(eval_dataset, summary_ret, comp, k_values=[5, 10, 20]):
    results = []
    for item in eval_dataset:
        query = item["query"]
        relevant = item["relevant_chunks"]
        retrieved_ids = retrieve_chunk_ids_two_stage(query, summary_ret, comp, max(k_values))
        metrics = {
            "query": query,
            "MRR": mrr(retrieved_ids, relevant)
        }
        for k in k_values:
            metrics[f"Precision@{k}"] = precision_at_k(retrieved_ids, relevant, k)
            metrics[f"Recall@{k}"] = recall_at_k(retrieved_ids, relevant, k)
            metrics[f"nDCG@{k}"] = ndcg_at_k(retrieved_ids, relevant, k)
        results.append(metrics)
    return results


def print_sources(docs):
    print("\n==== Retrieved & Reranked Chunks ====\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Chunk {i} ---")
        print("Metadata:", doc.metadata)
        print("Text:", doc.page_content[:300], "...\n")
        
# --------------------------------------------------------------
# 9. RUN QUERY & METRICS
# --------------------------------------------------------------
query = "What is Microsoft?"
start = time.perf_counter()

response, docs, selected_files = answer_query(query)

elapsed = time.perf_counter() - start

print("\n" + "=" * 50)
print("SELECTED FILES:", selected_files)
print("=" * 50)
print("\n========== ANSWER ==========")
print(response.content if hasattr(response, 'content') else response)
print(f"\nElapsed time: {elapsed:.2f} sec")

# ------------------ EVALUATION ------------------
eval_dataset = [
    {"query": "YOUR QUERY", "relevant_chunks": [1]},
    {"query": "YOUR QUERY", "relevant_chunks": [5, 6, 7]}
]

'''
print("\n" + "="*60)
print("STARTING EVALUATION OF TWO-STAGE RETRIEVAL")
print("="*60)

eval_results = evaluate_retriever(eval_dataset, summary_retriever, compressor)

for r in eval_results:
    print("\nQuery:", r["query"])
    print("MRR:", round(r["MRR"], 4))
    for k in [5, 10, 20]:
        print(f"Precision@{k}: {r[f'Precision@{k}']:.4f}")
        print(f"Recall@{k}:    {r[f'Recall@{k}']:.4f}")
        print(f"nDCG@{k}:      {r[f'nDCG@{k}']:.4f}")

print("\n========== RERANKED DOCUMENTS (FlashRank) ==========")
print_sources(docs)
'''
