# vllm-RAG

FastAPI server for LangChain RAG (Retrieval-Augmented Generation) with local Milvus vector database.

## Features

- FastAPI server with automatic OpenAPI documentation
- LangChain RAG system with:
  - Document loading and text splitting
  - **Local file-based Milvus vector database (no Docker needed!)**
  - OpenAI embeddings and ChatGPT
  - Retrieval-based question answering
- Thread-safe parallel request handling
- CLI configuration via argparse
- Health check endpoint

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

That's it! No Docker or external services needed.

## Usage

### Step 1: Index Your Documents

Before starting the server, you need to index your documents:

```bash
python ingest_documents.py
```

This will:
- Load documents from the `./documents` directory
- Split them into chunks
- Create embeddings and store them in `milvus_demo.db`

With custom options:
```bash
python ingest_documents.py \
  --documents-path ./documents \
  --milvus-db ./milvus_demo.db \
  --collection-name rag_collection \
  --chunk-size 1000 \
  --chunk-overlap 200 \
  --recreate  # Optional: recreate the collection from scratch
```

### Step 2: Start the Server

Basic usage:
```bash
python rag_server.py
```

With custom options:
```bash
python rag_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --milvus-db ./milvus_demo.db \
  --collection-name rag_collection \
  --model-name gpt-3.5-turbo
```

### Step 3: Ask Questions

```bash
uvicorn rag_server:app --host 0.0.0.0 --port 8000
```

Note: When using uvicorn directly, you need to set CLI args via environment or modify the code.

### Using Uvicorn directly

```bash
uvicorn rag_server:app --host 0.0.0.0 --port 8000
```

Note: When using uvicorn directly, you need to set CLI args via environment or modify the code.

## API Endpoints

### POST /ask

Ask a question to the RAG system.

**Request:**
```json
{
  "question": "What is FastAPI?"
}
```

**Response:**
```json
{
  "answer": "FastAPI is a modern, fast web framework for building APIs with Python..."
}
```

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?"}'
```

**Example with Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"question": "What is FastAPI?"}
)
print(response.json())
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "rag_initialized": true
}
```

### GET /

Get API information.

## Interactive API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## CLI Options

## CLI Options

### ingest_documents.py

- `--documents-path`: Path to directory containing documents (default: ./documents)
- `--milvus-db`: Path to local Milvus database file (default: ./milvus_demo.db)
- `--collection-name`: Milvus collection name (default: rag_collection)
- `--openai-api-key`: OpenAI API key (default: reads from OPENAI_API_KEY env var)
- `--chunk-size`: Chunk size for text splitting (default: 1000)
- `--chunk-overlap`: Chunk overlap for text splitting (default: 200)
- `--recreate`: Recreate collection (delete existing data)
- `--glob-pattern`: Glob pattern for files to load (default: **/*.txt)

### rag_server.py

- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--milvus-db`: Path to local Milvus database file (default: ./milvus_demo.db)
- `--collection-name`: Milvus collection name (default: rag_collection)
- `--openai-api-key`: OpenAI API key (default: reads from OPENAI_API_KEY env var)
- `--model-name`: OpenAI model name (default: gpt-3.5-turbo)

## Adding Documents

**Two-step process:**

1. **Index documents** using `ingest_documents.py`:
   ```bash
   python ingest_documents.py --documents-path ./documents
   ```

2. **Start the server** to serve the indexed documents:
   ```bash
   python rag_server.py
   ```

To add more documents to an existing index:
```bash
python ingest_documents.py --documents-path ./new_documents
```

To rebuild the entire index from scratch:
```bash
python ingest_documents.py --recreate
```

Supported formats:
- `.txt` files by default
- Can be extended to support other formats (PDF, Word, etc.) by modifying `ingest_documents.py`

## Architecture

**Document Ingestion (ingest_documents.py):**
1. **Document Loading**: Loads documents from the specified directory
2. **Text Splitting**: Splits documents into chunks for better retrieval
3. **Vector Store Creation**: Creates/updates local file-based Milvus database with OpenAI embeddings

**Server Runtime (rag_server.py):**
1. **Connect to Vector Store**: Connects to pre-existing Milvus database
2. **Initialize Retriever**: Configures retrieval strategy (top-k search)
3. **Setup Prompt Template**: Defines the RAG prompt format
4. **Initialize ChatOpenAI**: Sets up the language model
5. **Build QA Chain**: Combines all components into a question-answering chain

Parallel requests are handled safely by running the synchronous `qa_chain.invoke()` in a thread pool using `asyncio.run_in_executor()`.

The local Milvus database (`milvus_demo.db`) persists between server restarts, allowing fast startup times as documents don't need to be reindexed.

## License

MIT