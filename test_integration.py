"""
Integration test script to verify core functionality.

This script tests the main components without requiring API keys.
"""
import sys
import os
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        import config
        print("  ✓ config module")
    except Exception as e:
        print(f"  ✗ config module: {e}")
        return False
    
    try:
        import logger
        print("  ✓ logger module")
    except Exception as e:
        print(f"  ✗ logger module: {e}")
        return False
    
    try:
        import document_processor
        print("  ✓ document_processor module")
    except Exception as e:
        print(f"  ✗ document_processor module: {e}")
        return False
    
    try:
        import vector_store
        print("  ✓ vector_store module")
    except Exception as e:
        print(f"  ✗ vector_store module: {e}")
        return False
    
    try:
        import llm_manager
        print("  ✓ llm_manager module")
    except Exception as e:
        print(f"  ✗ llm_manager module: {e}")
        return False
    
    try:
        import assistant
        print("  ✓ assistant module")
    except Exception as e:
        print(f"  ✗ assistant module: {e}")
        return False
    
    try:
        import main
        print("  ✓ main module")
    except Exception as e:
        print(f"  ✗ main module: {e}")
        return False
    
    try:
        import cli
        print("  ✓ cli module")
    except Exception as e:
        print(f"  ✗ cli module: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from config import settings
        print(f"  ✓ Configuration loaded")
        print(f"    - Model: {settings.openai_model}")
        print(f"    - Chunk size: {settings.chunk_size}")
        print(f"    - Top K results: {settings.top_k_results}")
        return True
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False


def test_document_structure():
    """Test that example documents exist."""
    print("\nTesting document structure...")
    
    docs_dir = Path("./data/documents")
    
    if not docs_dir.exists():
        print(f"  ✗ Documents directory not found: {docs_dir}")
        return False
    
    print(f"  ✓ Documents directory exists")
    
    md_files = list(docs_dir.glob("*.md"))
    if md_files:
        print(f"  ✓ Found {len(md_files)} markdown files")
        for f in md_files:
            print(f"    - {f.name}")
        return True
    else:
        print("  ✗ No markdown files found")
        return False


def test_docker_files():
    """Test that Docker files exist."""
    print("\nTesting Docker configuration...")
    
    dockerfile = Path("./Dockerfile")
    compose_file = Path("./docker-compose.yml")
    
    if not dockerfile.exists():
        print("  ✗ Dockerfile not found")
        return False
    print("  ✓ Dockerfile exists")
    
    if not compose_file.exists():
        print("  ✗ docker-compose.yml not found")
        return False
    print("  ✓ docker-compose.yml exists")
    
    return True


def test_setup_files():
    """Test that setup files exist."""
    print("\nTesting setup files...")
    
    setup_py = Path("./setup.py")
    setup_sh = Path("./setup.sh")
    requirements = Path("./requirements.txt")
    env_example = Path("./.env.example")
    
    if not setup_py.exists():
        print("  ✗ setup.py not found")
        return False
    print("  ✓ setup.py exists")
    
    if not setup_sh.exists():
        print("  ✗ setup.sh not found")
        return False
    print("  ✓ setup.sh exists")
    
    if not requirements.exists():
        print("  ✗ requirements.txt not found")
        return False
    print("  ✓ requirements.txt exists")
    
    if not env_example.exists():
        print("  ✗ .env.example not found")
        return False
    print("  ✓ .env.example exists")
    
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("GeneralBot Integration Tests")
    print("="*60)
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Configuration", test_configuration()))
    results.append(("Document Structure", test_document_structure()))
    results.append(("Docker Files", test_docker_files()))
    results.append(("Setup Files", test_setup_files()))
    
    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n✓ All tests passed!")
        print("\nNext steps:")
        print("1. Create virtual environment: python -m venv venv")
        print("2. Activate virtual environment: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Set OPENAI_API_KEY in .env file")
        print("5. Ingest documents: python cli.py ingest ./data/documents")
        print("6. Start the server: python main.py")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
