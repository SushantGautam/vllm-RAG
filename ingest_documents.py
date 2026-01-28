"""
CLI tool for ingesting documents into Milvus vector database.

This script loads documents, splits them into chunks, and stores them in a local
Milvus database. Run this before starting the server to index your documents.
"""

import argparse
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into Milvus vector database"
    )
    parser.add_argument(
        "--documents-path",
        type=str,
        default="./documents",
        help="Path to directory containing documents (default: ./documents)",
    )
    parser.add_argument(
        "--milvus-db",
        type=str,
        default="./milvus_demo.db",
        help="Path to local Milvus database file (default: ./milvus_demo.db)",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="rag_collection",
        help="Milvus collection name (default: rag_collection)",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key for embeddings (default: reads from OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--embedding-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for embedding model (default: uses OpenAI's default)",
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="text-embedding-ada-002",
        help="Embedding model name (default: text-embedding-ada-002)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size for text splitting (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Chunk overlap for text splitting (default: 200)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate collection (delete existing data)",
    )
    parser.add_argument(
        "--glob-pattern",
        type=str,
        default="**/*.txt",
        help="Glob pattern for files to load (default: **/*.txt)",
    )
    return parser.parse_args()


def ingest_documents(args):
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
    
    embeddings = OpenAIEmbeddings(**embedding_kwargs)
    print("✓ Embeddings initialized")
    print()
    
    # Initialize Milvus vector store
    print(f"Connecting to Milvus database at {args.milvus_db}...")
    
    if args.recreate and os.path.exists(args.milvus_db):
        print(f"⚠ Recreating collection (deleting existing database)")
        # Delete the database file to recreate
        os.remove(args.milvus_db)
    
    vectorstore = Milvus.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=args.collection_name,
        connection_args={"uri": args.milvus_db},
    )
    
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
    args = parse_args()
    
    try:
        ingest_documents(args)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
