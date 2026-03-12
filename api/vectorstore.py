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

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# ✅ OpenAI Embedding Model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Pinecone index name
INDEX_NAME = "langgraph-rag-index"


# --- Retriever ---
def get_retriever():
    """Initializes and returns the Pinecone vector store retriever."""

    # Ensure the index exists
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {INDEX_NAME}...")

        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,  # ✅ dimension for text-embedding-3-small
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        print(f"Created new Pinecone index: {INDEX_NAME}")

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    return vectorstore.as_retriever()

def add_document_to_vectorstore(text_content: str):

    if not text_content:
        raise ValueError("Document content cannot be empty.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    documents = text_splitter.create_documents([text_content])

    print(f"Splitting document into {len(documents)} chunks for indexing...")

    # Ensure index exists
    if INDEX_NAME not in pc.list_indexes().names():
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

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    vectorstore.add_documents(documents)

    print(f"Added {len(documents)} chunks to Pinecone.")
    """
    Adds a single text document to the Pinecone vector store.
    Splits the text into chunks before embedding and upserting.
    """

    if not text_content:
        raise ValueError("Document content cannot be empty.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )

    documents = text_splitter.create_documents([text_content])

    print(f"Splitting document into {len(documents)} chunks for indexing...")

    # Ensure index exists before adding docs
    if INDEX_NAME not in pc.list_indexes().names():
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

    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    vectorstore.add_documents(documents)

    print(
        f"Successfully added {len(documents)} chunks to Pinecone index '{INDEX_NAME}'."
    )