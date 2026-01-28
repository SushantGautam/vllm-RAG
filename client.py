"""
Example client for the RAG FastAPI server.
Demonstrates how to interact with the server using Python requests.
"""

import argparse
import requests
import json


def ask_question(url: str, question: str):
    """Ask a question to the RAG server."""
    endpoint = f"{url}/ask"
    
    try:
        response = requests.post(
            endpoint,
            json={"question": question},
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result["answer"]
    
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


def check_health(url: str):
    """Check the health of the server."""
    endpoint = f"{url}/health"
    
    try:
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"


def main():
    parser = argparse.ArgumentParser(description="Example client for RAG server")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Question to ask"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check server health"
    )
    
    args = parser.parse_args()
    
    if args.health:
        print("Checking server health...")
        health = check_health(args.url)
        print(json.dumps(health, indent=2))
    
    elif args.question:
        print(f"Question: {args.question}")
        print("-" * 60)
        answer = ask_question(args.url, args.question)
        print(f"Answer: {answer}")
    
    else:
        # Interactive mode
        print("RAG Server Client (Interactive Mode)")
        print("=" * 60)
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60)
        
        # Check health first
        health = check_health(args.url)
        if isinstance(health, dict) and health.get("status") == "healthy":
            print("✓ Server is healthy and ready")
        else:
            print(f"⚠ Server health check failed: {health}")
            return
        
        print()
        
        while True:
            try:
                question = input("\nYour question: ").strip()
                
                if question.lower() in ["quit", "exit"]:
                    print("Goodbye!")
                    break
                
                if not question:
                    continue
                
                print("\nThinking...")
                answer = ask_question(args.url, question)
                print(f"\nAnswer: {answer}")
                print("-" * 60)
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()
