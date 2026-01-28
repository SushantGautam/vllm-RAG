"""
Test script for the RAG FastAPI server.
This script tests the server endpoints without requiring Milvus or OpenAI API.
"""

import json
import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import fastapi
        import uvicorn
        import langchain
        import pydantic
        print("✓ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_server_structure():
    """Test that the server file has the correct structure."""
    print("\nTesting server structure...")
    try:
        # Import without running main
        import rag_server
        
        # Check for required components
        assert hasattr(rag_server, 'app'), "Missing FastAPI app"
        assert hasattr(rag_server, 'QuestionRequest'), "Missing QuestionRequest model"
        assert hasattr(rag_server, 'AnswerResponse'), "Missing AnswerResponse model"
        assert hasattr(rag_server, 'parse_args'), "Missing parse_args function"
        assert hasattr(rag_server, 'initialize_rag_system'), "Missing initialize_rag_system function"
        
        # Check endpoints
        routes = [route.path for route in rag_server.app.routes]
        assert '/' in routes, "Missing root endpoint"
        assert '/health' in routes, "Missing health endpoint"
        assert '/ask' in routes, "Missing ask endpoint"
        
        print("✓ Server structure is correct")
        print(f"  Found endpoints: {', '.join(routes)}")
        return True
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        return False


def test_request_models():
    """Test that request/response models work correctly."""
    print("\nTesting Pydantic models...")
    try:
        from rag_server import QuestionRequest, AnswerResponse
        
        # Test QuestionRequest
        req = QuestionRequest(question="What is FastAPI?")
        assert req.question == "What is FastAPI?"
        
        # Test AnswerResponse
        resp = AnswerResponse(answer="FastAPI is a web framework")
        assert resp.answer == "FastAPI is a web framework"
        
        print("✓ Pydantic models work correctly")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False


def test_documents_exist():
    """Test that sample documents exist."""
    print("\nTesting sample documents...")
    import os
    
    docs_path = "./documents"
    expected_files = ["fastapi.txt", "langchain.txt", "milvus.txt"]
    
    if not os.path.exists(docs_path):
        print(f"✗ Documents directory not found: {docs_path}")
        return False
    
    found_files = []
    for file in expected_files:
        file_path = os.path.join(docs_path, file)
        if os.path.exists(file_path):
            found_files.append(file)
        else:
            print(f"  Warning: {file} not found")
    
    if found_files:
        print(f"✓ Found {len(found_files)}/{len(expected_files)} sample documents")
        for file in found_files:
            print(f"  - {file}")
        return True
    else:
        print("✗ No sample documents found")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("RAG FastAPI Server Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_server_structure,
        test_request_models,
        test_documents_exist,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
