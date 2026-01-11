# rag_pipeline.py

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from pypdf import PdfReader
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ChromaDB (local, persistent storage)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"description": "My RAG document collection"}
)

# ========================================
# STEP 1: DOCUMENT INGESTION & CHUNKING
# ========================================

def load_pdf(file_path):
    """Extract text from PDF"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def ingest_documents(documents_folder):
    """Process all PDFs in a folder"""
    documents_path = Path(documents_folder)
    pdf_files = list(documents_path.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files")
    
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    for idx, pdf_file in enumerate(pdf_files):
        print(f"Processing: {pdf_file.name}")
        
        # Extract text
        text = load_pdf(pdf_file)
        
        # Chunk it
        chunks = chunk_text(text)
        
        print(f"  - Created {len(chunks)} chunks")
        
        # Prepare for ChromaDB
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"doc{idx}_chunk{chunk_idx}"
            all_ids.append(chunk_id)
            all_chunks.append(chunk)
            all_metadata.append({
                "source": pdf_file.name,
                "chunk_index": chunk_idx
            })
    
    return all_chunks, all_metadata, all_ids

# ========================================
# STEP 2: EMBEDDING & STORAGE
# ========================================

def get_embedding(text):
    """Get embedding from OpenAI"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def store_in_chromadb(chunks, metadata, ids):
    """Store chunks with embeddings in ChromaDB"""
    print(f"\nEmbedding and storing {len(chunks)} chunks...")
    
    # ChromaDB can handle embeddings automatically if we provide the text
    # But we'll generate them explicitly to understand the process
    embeddings = []
    
    for i, chunk in enumerate(chunks):
        if i % 10 == 0:
            print(f"  Embedding chunk {i}/{len(chunks)}...")
        embedding = get_embedding(chunk)
        embeddings.append(embedding)
    
    # Add to ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
        ids=ids
    )
    
    print(f"✓ Stored {len(chunks)} chunks in ChromaDB")

# ========================================
# STEP 3: QUERY & RETRIEVAL
# ========================================

def search_documents(query, n_results=3):
    """Search for relevant chunks"""
    print(f"\nSearching for: '{query}'")
    
    # Get query embedding
    query_embedding = get_embedding(query)
    
    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    return results

# ========================================
# STEP 4: LLM GENERATION
# ========================================

def generate_answer(query, context_chunks):
    """Generate answer using LLM with retrieved context"""
    
    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[Source: {meta['source']}]\n{doc}" 
        for doc, meta in zip(context_chunks['documents'][0], 
                            context_chunks['metadatas'][0])
    ])
    
    # Create prompt
    prompt = f"""Context from documents:
{context}

Question: {query}

Please answer the question based on the context provided above. If the context doesn't contain enough information to answer the question, say so."""
    
    # Call OpenAI
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# ========================================
# MAIN FUNCTIONS
# ========================================

def setup_rag(documents_folder):
    """Complete setup: ingest, embed, store"""
    chunks, metadata, ids = ingest_documents(documents_folder)
    store_in_chromadb(chunks, metadata, ids)
    print("\n✓ RAG pipeline setup complete!")

def ask_question(query):
    """Complete RAG query flow"""
    # Retrieve relevant chunks
    results = search_documents(query, n_results=3)
    
    print("\nRetrieved chunks:")
    for i, (doc, meta) in enumerate(zip(results['documents'][0], 
                                        results['metadatas'][0])):
        print(f"\n--- Chunk {i+1} (from {meta['source']}) ---")
        print(doc[:200] + "..." if len(doc) > 200 else doc)
    
    # Generate answer
    print("\n" + "="*50)
    print("ANSWER:")
    print("="*50)
    answer = generate_answer(query, results)
    print(answer)
    
    return answer

# ========================================
# USAGE
# ========================================

if __name__ == "__main__":
    # Example usage:
    
    # First time setup (do this once):
    # setup_rag("./documents")
    
    # Then ask questions:
    # ask_question("What is the main topic of these documents?")
    
    print("RAG Pipeline Ready!")
    print("\nTo use:")
    print("1. First run: setup_rag('./documents')")
    print("2. Then ask: ask_question('your question here')")
