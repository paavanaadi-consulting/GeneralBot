"""Test script for API endpoints."""
import requests
import json
from pathlib import Path


BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n" + "="*60)
    print("Testing /health endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_chat():
    """Test chat endpoint."""
    print("\n" + "="*60)
    print("Testing /chat endpoint")
    print("="*60)
    
    payload = {
        "query": "What are your business hours?",
        "use_rag": True
    }
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_ingest():
    """Test ingest endpoint."""
    print("\n" + "="*60)
    print("Testing /ingest endpoint")
    print("="*60)
    
    payload = {
        "source_path": "./data/documents"
    }
    
    response = requests.post(
        f"{BASE_URL}/ingest",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_upload():
    """Test upload endpoint."""
    print("\n" + "="*60)
    print("Testing /upload endpoint")
    print("="*60)
    
    # Check if a test document exists
    test_doc = Path("./data/documents/company_knowledge.md")
    
    if not test_doc.exists():
        print(f"Test document not found: {test_doc}")
        return False
    
    with open(test_doc, 'rb') as f:
        files = {'file': (test_doc.name, f, 'text/markdown')}
        response = requests.post(f"{BASE_URL}/upload", files=files)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_session():
    """Test session management endpoints."""
    print("\n" + "="*60)
    print("Testing session endpoints")
    print("="*60)
    
    # First create a chat to establish a session
    payload = {
        "query": "Hello, this is a test",
        "session_id": "test_session_123",
        "use_rag": False
    }
    
    response = requests.post(
        f"{BASE_URL}/chat",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Chat Status: {response.status_code}")
    
    # Get session history
    session_id = "test_session_123"
    response = requests.get(f"{BASE_URL}/session/{session_id}")
    print(f"\nGet History Status: {response.status_code}")
    print(f"History: {json.dumps(response.json(), indent=2)}")
    
    # Clear session
    response = requests.delete(f"{BASE_URL}/session/{session_id}")
    print(f"\nClear Session Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def main():
    """Run all API tests."""
    print("\n" + "="*60)
    print("GeneralBot API Tests")
    print("="*60)
    print("\nNote: Make sure the API server is running on http://localhost:8000")
    print("      Start it with: python main.py")
    
    try:
        # Test health endpoint
        test_health()
        
        # Test ingest endpoint
        test_ingest()
        
        # Test chat endpoint
        test_chat()
        
        # Test upload endpoint (if document exists)
        test_upload()
        
        # Test session endpoints
        test_session()
        
        print("\n" + "="*60)
        print("API Tests Completed!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the API server")
        print("   Make sure the server is running: python main.py")
    except Exception as e:
        print(f"\n❌ Error running tests: {str(e)}")


if __name__ == "__main__":
    main()
