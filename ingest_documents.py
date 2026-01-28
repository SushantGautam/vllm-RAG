"""
CLI tool for ingesting documents into Milvus vector database.

This script loads documents, splits them into chunks, and stores them in a local
Milvus database. Run this before starting the server to index your documents.
"""

import argparse
import os
from dotenv import load_dotenv, dotenv_values, find_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai import OpenAIEmbeddings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into Milvus vector database"
    )
    parser.add_argument(
        "--documents-path",
        type=str,
        default=os.getenv("DOCUMENTS_PATH", "./documents"),
        help="Path to directory containing documents (default: ./documents or DOCUMENTS_PATH env var)",
    )
    parser.add_argument(
        "--milvus-db",
        type=str,
        default=os.getenv("MILVUS_DB", "./milvus_demo.db"),
        help="Path to local Milvus database file (default: ./milvus_demo.db or MILVUS_DB env var)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default=os.getenv("COLLECTION_NAME", "rag_collection"),
        help="Milvus collection name (default: rag_collection or COLLECTION_NAME env var)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key for embeddings (default: reads from OPENAI_API_KEY env var or .env)",
    )
    parser.add_argument(
        "--embedding-base-url",
        type=str,
        default=os.getenv("EMBEDDING_BASE_URL"),
        help="OpenAI-compatible base URL for embedding model (default: uses OpenAI's default or EMBEDDING_BASE_URL env var)",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-ada-002"),
        help="Embedding model name (default: text-embedding-ada-002 or EMBEDDING_MODEL_NAME env var)",
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default=os.getenv("EMBEDDING_API_KEY"),
        help="API key for embedding model if different from main OpenAI API key (default: reads from EMBEDDING_API_KEY env var or .env)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.getenv("CHUNK_SIZE", "1000")),
        help="Chunk size for text splitting (default: 1000 or CHUNK_SIZE env var)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=int(os.getenv("CHUNK_OVERLAP", "200")),
        help="Chunk overlap for text splitting (default: 200 or CHUNK_OVERLAP env var)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection (delete existing data)",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default=os.getenv("GLOB_PATTERN", "**/*.txt"),
        help="Glob pattern for files to load (default: **/*.txt or GLOB_PATTERN env var)",
    )
    return parser.parse_args()


def validate_env_vars(required_vars):
    """
    Validate that required environment variables are declared. If a .env file
    exists, ensure those variables are present in the file. Otherwise, ensure
    they are set in the environment.
    """
    env_path = ".env"
    if os.path.exists(env_path):
        env_vals = dotenv_values(env_path)
        missing = [v for v in required_vars if not env_vals.get(v)]
        if missing:
            raise ValueError(f".env file is missing required variables: {', '.join(missing)}")
    else:
        missing = [v for v in required_vars if v not in os.environ]
        if missing:
            raise ValueError(f"Required environment variables not set: {', '.join(missing)}. Create a .env file or set them in the environment.")


async def ingest_documents(args):
    """
    Load documents, split into chunks, and store in Milvus vector database.
    """
    print("=" * 60)
    print("Document Ingestion Tool")
    print("=" * 60)
    print()
    
    # Validate OpenAI API key
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OpenAI API key must be provided via --openai-api-key or OPENAI_API_KEY env var"
        )
    
    # Load documents
    print(f"Loading documents from {args.documents_path}...")
    if not os.path.exists(args.documents_path):
        raise ValueError(f"Documents path '{args.documents_path}' does not exist")
    
    loader = DirectoryLoader(
        args.documents_path,
        glob=args.glob_pattern,
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents = loader.load()
    
    if not documents:
        print(f"⚠ No documents found in {args.documents_path} matching {args.glob_pattern}")
        return
    
    print(f"✓ Loaded {len(documents)} document(s)")
    print()
    
    # Split documents
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    splits = text_splitter.split_documents(documents)
    print(f"✓ Created {len(splits)} chunk(s)")
    print()
    
    # Initialize embeddings
    print(f"Initializing embeddings (model: {args.embedding_model_name})...")
    embedding_kwargs = {"model": args.embedding_model_name}
    
    # Set embedding base URL if provided
    if args.embedding_base_url:
        embedding_kwargs["base_url"] = args.embedding_base_url
    if args.embedding_api_key:
        embedding_kwargs["openai_api_key"] = args.embedding_api_key or os.environ.get("OPENAI_API_KEY")
    
    embeddings = OpenAIEmbeddings(**embedding_kwargs)
    print("✓ Embeddings initialized")
    print()
    
    # Initialize Milvus vector store
    print(f"Connecting to Milvus database at {args.milvus_db}...")
    
    if args.recreate and os.path.exists(args.milvus_db):
        print(f"⚠ Recreating collection (deleting existing database)")
        # Delete the database file to recreate
        os.remove(args.milvus_db)

    # Run inside an asyncio event loop so AsyncMilvusClient can be initialized
    # without producing a warning. Create the Milvus instance directly in this
    # (main) thread while the event loop is active.
    import asyncio
    from langchain_milvus import Milvus

    try:
        vectorstore = Milvus.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name=args.collection_name,
            connection_args={
                "uri": args.milvus_db,
            },
            index_params={"index_type": "FLAT", "metric_type": "L2"},
        )
    except Exception as e:
        import traceback
        print("\n✗ Error while creating/connecting to Milvus vector store:")
        traceback.print_exc()
        raise

    print("✓ Vector store created and populated")
    print()
    
    # Print summary
    print("=" * 60)
    print("Ingestion Complete!")
    print("=" * 60)
    print(f"Documents processed: {len(documents)}")
    print(f"Chunks created: {len(splits)}")
    print(f"Database file: {args.milvus_db}")
    print(f"Collection name: {args.collection_name}")
    print()
    print("You can now start the server with:")
    print(f"  python rag_server.py --milvus-db {args.milvus_db}")
    print()


def main():
    """Main entry point."""
    # Prefer the project's `.env` in the current working directory (usecwd=True)
    # so running via `uv run --with ...` or from a different install location
    # still picks up project-level env vars.
    _env_path = find_dotenv(usecwd=True)
    if _env_path:
        print(f"LOADING .env from project path: {_env_path}")
        load_dotenv(_env_path, override=True)
    else:
        load_dotenv()

    # Validate required env vars are declared in .env or the environment
    try:
        validate_env_vars(["OPENAI_API_KEY"])
    except Exception as e:
        print(f"\n✗ Environment validation failed: {e}")
        return 1

    args = parse_args()
    
    try:
        import asyncio

        asyncio.run(ingest_documents(args))
    except Exception as e:
        import traceback
        print()
        print("✗ Unhandled error during ingestion:")
        traceback.print_exc()
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
