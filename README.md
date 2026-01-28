# vllm-RAG

FastAPI server for LangChain RAG (Retrieval-Augmented Generation) with local Milvus vector database. **OpenAI-compatible API** for seamless integration.

## Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI's chat completion endpoint
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

**Using OpenAI-compatible embedding endpoints:**
```bash
python ingest_documents.py \
  --documents-path ./documents \
  --embedding-base-url http://localhost:8002/v1 \
  --embedding-model-name your-embedding-model \
  --openai-api-key your-api-key
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

**Using OpenAI-compatible endpoints (e.g., vLLM, LocalAI, LM Studio):**
```bash
python rag_server.py \
  --openai-base-url http://localhost:8001/v1 \
  --openai-api-key your-api-key \
  --model-name your-model-name \
  --embedding-base-url http://localhost:8002/v1 \
  --embedding-api-key your-embedding-api-key \
  --embedding-model-name your-embedding-model
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

---

### Run via `uv` (Astral) 🔧

You can run the tools directly from a remote Git repository with Astral `uv`. The repository exposes two entry points (console scripts): `rag-server` and `ingest-document`. Example:

```bash
uv run git+https://github.com/<owner>/<repo>.git#rag-server
uv run git+https://github.com/<owner>/<repo>.git#ingest-document
```

You can also use `--with` to install the package first and then run its console script. When using `--with`, add `--` before the script name to separate `uv`'s options from the command. When running with `--with` or using the local path, `uv` will run commands from your current working directory and the code will prefer the `.env` file in that directory — so project-specific environment values (like `OPENAI_API_KEY` or `MILVUS_DB`) will be picked up automatically.

Example:

```bash
uv run --with git+https://github.com/<owner>/<repo>.git -- rag-server --openai-api-key "$OPENAI_API_KEY" --port 8000
```

Replace `<owner>/<repo>` with the real Git URL (or a `git+ssh://` URL). You can also pin a branch, tag, or commit, for example:

```bash
uv run git+https://github.com/<owner>/<repo>.git@main#rag-server
```

These entry points map to the package's console scripts defined in `pyproject.toml`:

- `rag-server` → `rag_server:main` (starts the FastAPI server)
- `ingest-document` → `ingest_documents:main` (runs document ingestion)

---


## API Endpoints

### POST /v1/chat/completions

OpenAI-compatible chat completion endpoint for asking questions to the RAG system.

**Request:**
```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "What is FastAPI?"}
  ]
}
```

**Response:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "FastAPI is a modern, fast web framework for building APIs with Python..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

**Example with curl:**
```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "What is FastAPI?"}]
  }'
```

**Example with Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "What is FastAPI?"}]
    }
)
print(response.json()["choices"][0]["message"]["content"])
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
- `--openai-api-key`: OpenAI API key for embeddings (default: reads from OPENAI_API_KEY env var)
- `--embedding-base-url`: OpenAI-compatible base URL for embedding model (default: uses OpenAI's default)
- `--embedding-model-name`: Embedding model name (default: text-embedding-ada-002)
- `--chunk-size`: Chunk size for text splitting (default: 1000)
- `--chunk-overlap`: Chunk overlap for text splitting (default: 200)
- `--recreate`: Recreate collection (delete existing data)
- `--glob-pattern`: Glob pattern for files to load (default: **/*.txt)

### rag_server.py

- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--milvus-db`: Path to local Milvus database file (default: ./milvus_demo.db)
- `--collection-name`: Milvus collection name (default: rag_collection)
- `--openai-api-key`: OpenAI API key for LLM (default: reads from OPENAI_API_KEY env var)
- `--openai-base-url`: OpenAI-compatible base URL for LLM (default: uses OpenAI's default)
- `--model-name`: OpenAI model name (default: gpt-3.5-turbo)
- `--embedding-api-key`: API key for embedding model (default: uses same as --openai-api-key)
- `--embedding-base-url`: OpenAI-compatible base URL for embedding model (default: uses OpenAI's default)
- `--embedding-model-name`: Embedding model name (default: text-embedding-ada-002)

## Using OpenAI-Compatible Endpoints

This RAG system supports any OpenAI-compatible API endpoint, including:
- **vLLM**: High-performance LLM inference server
- **LocalAI**: Local, privacy-focused AI
- **LM Studio**: Easy-to-use local LLM server
- **Ollama**: Run large language models locally
- **Text Generation WebUI**: Gradio web UI for LLMs

### Configuration

Both the **LLM** and **embedding model** can be configured independently with:
- Custom base URLs
- Custom model names  
- Separate API keys

### Example: Using vLLM for both LLM and embeddings

1. **Start vLLM servers:**
   ```bash
   # Start LLM server
   vllm serve your-llm-model --port 8001
   
   # Start embedding server
   vllm serve your-embedding-model --port 8002
   ```

2. **Ingest documents with custom embedding endpoint:**
   ```bash
   python ingest_documents.py \
     --documents-path ./documents \
     --embedding-base-url http://localhost:8002/v1 \
     --embedding-model-name your-embedding-model \
     --openai-api-key dummy-key
   ```

3. **Start RAG server with custom endpoints:**
   ```bash
   python rag_server.py \
     --openai-base-url http://localhost:8001/v1 \
     --openai-api-key dummy-key \
     --model-name your-llm-model \
     --embedding-base-url http://localhost:8002/v1 \
     --embedding-api-key dummy-key \
     --embedding-model-name your-embedding-model
   ```

### Example: Using different providers for LLM and embeddings

You can mix and match providers. For example, use OpenAI for embeddings but a local vLLM server for the LLM:

```bash
python rag_server.py \
  --openai-base-url http://localhost:8001/v1 \
  --openai-api-key dummy-key \
  --model-name your-local-model \
  --embedding-model-name text-embedding-ada-002 \
  --embedding-api-key sk-your-openai-key
```

**Note:** Make sure your embedding model dimensions match between document ingestion and server runtime. Using different embedding models will result in poor retrieval quality.

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