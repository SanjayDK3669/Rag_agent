# rag_agent_app/backend/vectorstore.py

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import PINECONE_API_KEY, OPENAI_API_KEY

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

INDEX_NAME = "langgraph-rag-index"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


def get_pinecone_client():
    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index_exists(pc):

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


def get_vectorstore():

    pc = get_pinecone_client()

    ensure_index_exists(pc)

    index = pc.Index(INDEX_NAME)

    return PineconeVectorStore(
        index=index,
        embedding=embeddings
    )


def get_retriever():

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever()


def add_document_to_vectorstore(text_content: str):

    if not text_content.strip():
        raise ValueError("Document content cannot be empty")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    documents = text_splitter.create_documents([text_content])

    print(f"Splitting document into {len(documents)} chunks for indexing...")

    vectorstore = get_vectorstore()

    vectorstore.add_documents(documents)

    print(f"Successfully added {len(documents)} chunks to Pinecone")