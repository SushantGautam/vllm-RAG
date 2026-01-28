"""
Test script for OpenAI-compatible configuration.
This script tests the new CLI arguments for custom base URLs and API keys.
"""

import re
import sys


def test_rag_server_args():
    """Test that rag_server.py has OpenAI-compatible arguments defined."""
    print("Testing rag_server.py CLI arguments...")
    
    # Read the file and check for argument definitions
    with open('rag_server.py', 'r') as f:
        content = f.read()
    
    # Check for required arguments
    required_args = [
        '--openai-base-url',
        '--embedding-base-url',
        '--embedding-model-name',
        '--embedding-api-key',
    ]
    
    for arg in required_args:
        pattern = f'"{arg}"'
        if pattern not in content:
            raise AssertionError(f"Argument {arg} not found in rag_server.py")
    
    print("✓ rag_server.py has all required CLI arguments")
    print("  - --openai-base-url")
    print("  - --embedding-base-url")
    print("  - --embedding-model-name")
    print("  - --embedding-api-key")
    return True


def test_ingest_documents_args():
    """Test that ingest_documents.py has OpenAI-compatible arguments defined."""
    print("\nTesting ingest_documents.py CLI arguments...")
    
    # Read the file and check for argument definitions
    with open('ingest_documents.py', 'r') as f:
        content = f.read()
    
    # Check for required arguments
    required_args = [
        '--embedding-base-url',
        '--embedding-model-name',
    ]
    
    for arg in required_args:
        pattern = f'"{arg}"'
        if pattern not in content:
            raise AssertionError(f"Argument {arg} not found in ingest_documents.py")
    
    print("✓ ingest_documents.py has all required CLI arguments")
    print("  - --embedding-base-url")
    print("  - --embedding-model-name")
    return True


def test_embedding_initialization():
    """Test that embedding initialization uses custom parameters."""
    print("\nTesting embedding initialization in rag_server.py...")
    
    with open('rag_server.py', 'r') as f:
        content = f.read()
    
    # Check that embeddings are initialized with kwargs
    checks = [
        ('embedding_kwargs', 'embedding_kwargs dictionary is created'),
        ('args.embedding_model_name', 'embedding model name is used'),
        ('args.embedding_base_url', 'embedding base URL is checked'),
        ('OpenAIEmbeddings\\(\\*\\*embedding_kwargs\\)', 'embeddings initialized with kwargs'),
    ]
    
    for pattern, description in checks:
        if not re.search(pattern, content):
            raise AssertionError(f"Missing: {description}")
    
    print("✓ Embedding initialization uses custom parameters correctly")
    return True


def test_llm_initialization():
    """Test that LLM initialization uses custom base URL."""
    print("\nTesting LLM initialization in rag_server.py...")
    
    with open('rag_server.py', 'r') as f:
        content = f.read()
    
    # Check that LLM is initialized with kwargs
    checks = [
        ('llm_kwargs', 'llm_kwargs dictionary is created'),
        ('args.openai_base_url', 'LLM base URL is checked'),
        ('ChatOpenAI\\(\\*\\*llm_kwargs\\)', 'LLM initialized with kwargs'),
    ]
    
    for pattern, description in checks:
        if not re.search(pattern, content):
            raise AssertionError(f"Missing: {description}")
    
    print("✓ LLM initialization uses custom base URL correctly")
    return True


def test_readme_documentation():
    """Test that README.md has been updated with new documentation."""
    print("\nTesting README.md documentation...")
    
    with open('README.md', 'r') as f:
        content = f.read()
    
    # Check for documentation of new features
    checks = [
        ('OpenAI-Compatible Endpoints', 'Section on OpenAI-compatible endpoints'),
        ('--embedding-base-url', 'Documentation for embedding-base-url'),
        ('--embedding-model-name', 'Documentation for embedding-model-name'),
        ('--openai-base-url', 'Documentation for openai-base-url'),
        ('vLLM', 'Example with vLLM'),
    ]
    
    for pattern, description in checks:
        if pattern not in content:
            raise AssertionError(f"Missing in README: {description}")
    
    print("✓ README.md has been updated with comprehensive documentation")
    return True


def test_env_example():
    """Test that .env.example has been updated."""
    print("\nTesting .env.example configuration...")
    
    with open('.env.example', 'r') as f:
        content = f.read()
    
    # Check for new configuration options
    checks = [
        'EMBEDDING_MODEL_NAME',
        'OPENAI_BASE_URL',
        'EMBEDDING_BASE_URL',
    ]
    
    for check in checks:
        if check not in content:
            raise AssertionError(f"Missing in .env.example: {check}")
    
    print("✓ .env.example has been updated with new configuration options")
    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("OpenAI-Compatible Configuration Test Suite")
    print("=" * 70)
    print()
    
    tests = [
        test_rag_server_args,
        test_ingest_documents_args,
        test_embedding_initialization,
        test_llm_initialization,
        test_readme_documentation,
        test_env_example,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print("\n" + "=" * 70)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 70)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
