from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import PDF_PATH, FAISS_INDEX
import os

def get_vector_db():
    """Get or create FAISS vector database with progress tracking"""
    print("\n=== Setting up vector database ===")
    
    # Initialize embeddings
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(FAISS_INDEX):
        print("Loading existing vector database...")
        return FAISS.load_local(FAISS_INDEX, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Processing PDF: {PDF_PATH}")
        
        # Load document
        print("- Loading PDF pages...")
        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()
        print(f"  Loaded {len(pages)} pages")
        
        # Split into chunks
        print("- Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)
        print(f"  Created {len(chunks)} chunks")
        
        # Create vector store
        print("- Creating embeddings (this may take a while)...")
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        # Save index
        print("- Saving vector database...")
        vector_db.save_local(FAISS_INDEX)
        print("✓ Vector database created successfully")
        return vector_db