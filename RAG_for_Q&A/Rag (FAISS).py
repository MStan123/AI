from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_qdrant import QdrantVectorStore
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import qdrant_client
import fitz
import os

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
deployment_name_fallback = os.getenv("AZURE_OPENAI_DEPLOYMENT_FALLBACK")
api_version_fallback = os.getenv("AZURE_OPENAI_API_VERSION_FALLBACK")

# Load environment variables
doc = fitz.open("FAQ.pdf")

content = []
for page in doc:
    text = page.get_text()
    content.append(text)

doc.close()
# Load the document

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
       chunk_size = 500,
       chunk_overlap = 20
)

chunked_docs = [] # this list will store the chunked documents

for text in content:
   docs = text_splitter.create_documents([text])
   chunked_docs.extend(docs)

# 1. Инициализируем модель эмбеддингов
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# FAISS expects document objects and the embedding model
vector_store = FAISS.from_documents(chunked_docs, embeddings)

# Use the vector store's retriever
retriever = vector_store.as_retriever()

# Initialize the LLM (using OpenAI)
llm = AzureChatOpenAI(
    azure_endpoint=endpoint,
    azure_deployment=deployment_name,
    api_version=api_version,
    api_key=api_key
)

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_prompt = SystemMessagePromptTemplate.from_template(
    "You are the AI assistant of Microsoft support department.\n"
    "Answer ONLY using facts strictly from the provided context.\n"
    "If the answer is not in the context, reply exactly: 'I do not know, talk with support team'.\n"
    "Determine the user's language from their query.\n"
    "If the context is in another language, translate the answer to the user's language.\n"
    "Do NOT guess or invent information."
)

human_prompt = HumanMessagePromptTemplate.from_template(
    "Контекст:\n{context}\n\nВопрос: {question}"
)

prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt})

# Example query
query = "What is Microsoft?"
response = qa_chain.invoke({"query": query})

# Print the response
print(response)
