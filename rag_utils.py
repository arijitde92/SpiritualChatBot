import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_core.documents import Document
from uuid import uuid4

def load_pdf_documents(pdf_paths: List[str]) -> List[Document]:
    """Load PDF documents and return them as a list of Document objects."""
    documents = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            # Add source information to metadata
            for doc in docs:
                doc.metadata["source"] = os.path.basename(pdf_path)
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {pdf_path}: {e}")
    return documents

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document], embeddings_model: str = "text-embedding-3-small") -> PineconeVectorStore:
    """Create and return a FAISS vector store."""
    # embeddings = OpenAIEmbeddings(model=embeddings_model)
    # vector_store = FAISS.from_documents(documents, embeddings)
    # return vector_store

    """Create and return a Pinecone vector store."""
    try:
        embeddings = OpenAIEmbeddings(model=embeddings_model)
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "spiritual-chatbot-index"
        
        # Check if index exists
        if not pc.has_index(index_name):
            print("Creating new Pinecone index...")
            pc.create_index(
                name=index_name,
                dimension=1536,  # OpenAI embeddings dimension
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        # Get the index
        index = pc.Index(index_name)
        
        # Create vector store
        vector_store = PineconeVectorStore(embedding=embeddings, index=index)
        
        # Add documents in batches to avoid size limits
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            uuids = [str(uuid4()) for _ in range(len(batch))]
            vector_store.add_documents(documents=batch, ids=uuids)
            print(f"Added batch {i//batch_size + 1} of {(len(documents) + batch_size - 1)//batch_size}")
        
        return vector_store
        
    except Exception as e:
        print(f"Error creating Pinecone index: {e}")
        return None

def get_relevant_documents(vector_store: PineconeVectorStore, query: str, k: int = 4) -> List[Document]:
    """Retrieve relevant documents for a given query and print sources to console."""
    try:
        docs = vector_store.similarity_search(query, k=k)
        
        # Print sources to console
        print("\n=== Relevant Sources ===")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            print(f"Source {i}: {source} (Page {page})")
        print("=====================\n")
        
        return docs
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

def initialize_rag():
    """Initialize RAG system with PDF documents."""
    try:
        # Get PDF paths
        pdf_paths = [
            os.path.join("ebook", "Bhagavad-gita.pdf"),
            os.path.join("ebook", "Mahabharata.pdf")
        ]
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "spiritual-chatbot-index"
        
        # Check if index exists and has vectors
        if pc.has_index(index_name):
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            
            # If index has vectors, return existing vector store
            if stats.total_vector_count > 0:
                print("Using existing Pinecone index...")
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                return PineconeVectorStore(embedding=embeddings, index=index)
        
        # If index doesn't exist or is empty, create new vector store
        print("Creating new vector store...")
        documents = load_pdf_documents(pdf_paths)
        split_docs = split_documents(documents)
        vector_store = create_vector_store(split_docs)
        return vector_store
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return None 