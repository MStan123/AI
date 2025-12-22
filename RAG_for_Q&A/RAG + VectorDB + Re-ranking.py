from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import qdrant_client
import fitz
import time
import os
import re

# --------------------------------------------------------------
# 0. LOAD SECRET DATA
# --------------------------------------------------------------

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
deployment_name_fallback = os.getenv("AZURE_OPENAI_DEPLOYMENT_FALLBACK")
api_version_fallback = os.getenv("AZURE_OPENAI_API_VERSION_FALLBACK")

# --------------------------------------------------------------
# 1. LOAD AND SPLIT PDF
# --------------------------------------------------------------

doc = fitz.open("FAQ_az.pdf")

content = []
for page in doc:
    text = page.get_text()
    content.append(text)
doc.close()
full_text = "\n".join(content)


"""# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20
)

chunked_docs = []
for text in content:
    docs = text_splitter.create_documents([text])
    chunked_docs.extend(docs)"""

def split_faq_by_questions(text):
    """
    Разбивает весь FAQ на логические блоки (Q/A)
    """
    pattern = r'(Q: .+?)(?=Q: |$)'
    blocks = re.findall(pattern, text, flags=re.DOTALL)
    return [b.strip() for b in blocks if b.strip()]

faq_blocks = split_faq_by_questions(full_text)

from langchain_core.documents import Document

documents = []
for i, block in enumerate(faq_blocks):
    documents.append(
        Document(
            page_content=block,
            metadata={
                "source": "FAQ_az.pdf",
                "chunk_id": i,
                "type": "faq"
            }
        )
    )
# --------------------------------------------------------------
# 2. EMBEDDINGS
# --------------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# --------------------------------------------------------------
# 3. SETUP QDRANT
# --------------------------------------------------------------

client = qdrant_client.QdrantClient(
    url="http://localhost:6333",
    api_key=None,
)

collection_name = "vector_db"

# recreate collection safely
collections = [c.name for c in client.get_collections().collections]
if collection_name in collections:
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE
    )
)

# Wrap collection in LangChain store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings,
)

vector_store.add_documents(documents)

# Initial retrieval before reranking
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 20}     # retrieve more before compression
)


# --------------------------------------------------------------
# 4. FLASHRANK RERANKER
# --------------------------------------------------------------
# Required by pydantic for this class
FlashrankRerank.model_rebuild()

# Create compressor
compressor = FlashrankRerank(
    model="ms-marco-MiniLM-L-12-v2",
    top_n = 7,  # print top_n = 7 from k = 20
)
# FlashRank uses cross-encoders optimized for speed and accuracy

# Rerank + filter irrelevant docs automatically
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)


# --------------------------------------------------------------
# 5. AZURE OPENAI LLM
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

#---------------------------------------------------------------
# 6. Print metadata
#---------------------------------------------------------------
def print_sources(docs):
    print("\n==== Retrieved & Reranked Chunks ====\n")
    for i, doc in enumerate(docs, 1):
        print(f"--- Chunk {i} ---")
        print("Metadata:", doc.metadata)
        print("Text:", doc.page_content[:1000], "...\n")

# --------------------------------------------------------------
# 7. RAG PIPELINE
# --------------------------------------------------------------

def answer_query(query: str):
    # Retrieve + rerank + compress (FlashRank CE-reranker)
    compressed_docs = compression_retriever.invoke(query)

    # Build context
    context = "\n\n".join([doc.page_content for doc in compressed_docs])

    messages = [
        {
            "role": "system",
            "content":
                "You are the AI assistant of Microsoft support department.\n"
                "Answer ONLY using facts strictly from the provided context.\n"
                "If you not found any information say I don't know anything, call to our support.\n"
                "Do NOT invent information.\n"
                "Determine the user's language from their query.\n"
                "If the context is in another language, translate the answer to the user's language.\n"
                "Do NOT guess or invent information."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}"
        }
    ]

    # Query LLM
    try:
        response = llm.invoke(messages)
    except Exception:
        response = fallback_llm.invoke(messages)

    return response, compressed_docs


# --------------------------------------------------------------
# 8. RUN QUERY
# --------------------------------------------------------------

query = "What is Microsoft"

start = time.perf_counter()
response, docs = answer_query(query)
elapsed = time.perf_counter() - start

print("\n========== ANSWER ==========")
print(response)

print(f"\nElapsed time: {elapsed:.2f} sec")

print("\n========== RERANKED DOCUMENTS (FlashRank) ==========")
print_sources(docs)
