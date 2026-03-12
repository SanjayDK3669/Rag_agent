# rag_agent_app/backend/vectorstore.py

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import API keys from config
from .config import PINECONE_API_KEY, OPENAI_API_KEY

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Pinecone index name
INDEX_NAME = "langgraph-rag-index"

# Embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


def get_pinecone_client():
    """Create Pinecone client (safe for serverless environments)."""
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index_exists(pc: Pinecone):
    """Create Pinecone index if it does not exist."""
    existing_indexes = pc.list_indexes().names()

    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {INDEX_NAME}")

        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print(f"Pinecone index '{INDEX_NAME}' created.")


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------
def get_retriever():
    """Return Pinecone retriever for RAG queries."""

    pc = get_pinecone_client()

    ensure_index_exists(pc)

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    return vectorstore.as_retriever()


# ---------------------------------------------------------
# Add Document
# ---------------------------------------------------------
def add_document_to_vectorstore(text_content: str):
    """
    Adds text content to Pinecone vector store.
    Splits text into chunks and stores embeddings.
    """

    if not text_content.strip():
        raise ValueError("Document content cannot be empty.")

    pc = get_pinecone_client()

    ensure_index_exists(pc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    documents = text_splitter.create_documents([text_content])

    print(f"Splitting document into {len(documents)} chunks for indexing...")

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    vectorstore.add_documents(documents)

    print(f"Successfully added {len(documents)} chunks to Pinecone.")