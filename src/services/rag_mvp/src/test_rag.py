# test_rag.py

from rag_pipeline import setup_rag, ask_question

# Step 1: Ingest documents (run once)
print("Setting up RAG pipeline...")
setup_rag("./documents")

# Step 2: Ask questions
print("\n" + "="*60)
print("Testing queries...")
print("="*60)

# Example questions (modify based on your documents)
questions = [
    "What are the main topics covered in these documents?",
    "Can you summarize the key points?",
    # Add questions relevant to YOUR documents
]

for question in questions:
    print("\n" + "="*60)
    ask_question(question)
    print("\n")
