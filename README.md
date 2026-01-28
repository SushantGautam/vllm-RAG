# vllm-RAG

FastAPI RAG (Retrieval-Augmented Generation) server with a local file-based Milvus vector store. Works with OpenAI or any OpenAI-compatible LLM/embedding endpoints.

## Prerequisites

- Python 3.8+ (project tested on modern Python versions)
- `pip` to install dependencies
- An OpenAI API key or an OpenAI-compatible LLM/embedding endpoint (optional)
- Local Milvus via `milvus-lite` is supported (no separate Milvus service or Docker required)

---

## Quick Start ✅

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file (strongly recommended — use the interactive `setup`):

- Interactive (strongly recommended):
  ```bash
  python setup.py
  # The script prompts for HOST, PORT, MILVUS_DB, COLLECTION_NAME, DOCUMENTS_PATH,
  # MODEL_NAME, EMBEDDING_MODEL_NAME, CHUNK_SIZE and CHUNK_OVERLAP.
  # It writes to ./.env by default and backs up any existing .env to ./.env.bak.
  # After running, edit ./.env and add sensitive keys such as OPENAI_API_KEY, EMBEDDING_API_KEY, and API_SECRET.
  # Re-run `python setup.py` anytime to override values; the previous .env will be backed up to .env.bak.
  # Note: When running `setup` via remote install (for example with `uv run --with ... -- setup`), the script will use the packaged `.env.example` by default so you don't need to pass `--example`.
  ```

- Quick (copy example):
  ```bash
  cp .env.example .env
  # Edit .env and add your OPENAI_API_KEY and any secrets
  ```

3. Index your documents (default path: `./documents`):

```bash
python ingest_documents.py
```

Optional flags:
```bash
python ingest_documents.py --documents-path ./documents --milvus-db ./milvus_demo.db --collection-name rag_collection --chunk-size 1000 --chunk-overlap 200
```

4. Start the server:

```bash
python rag_server.py
# or run uvicorn directly
uvicorn rag_server:app --host 0.0.0.0 --port 8000
```

5. Test the server:

- Swagger UI: http://localhost:8000/docs
- Health check: `curl http://localhost:8000/health`
- Chat completions POST: `http://localhost:8000/v1/chat/completions`

---

## Run quickly with Astral `uv` ⚡

An easy one-liner to spin up the server from a remote repo (uses defaults from `.env` if present):

```bash
uv run --refresh --with git+https://github.com/SushantGautam/vllm-RAG.git rag-server
```

Recommended flow (strongly encourage using `setup` to create `./.env` and then add API keys there):

1. Create `./.env` interactively (writes `./.env` and backs up existing `.env` to `.env.bak`):

```bash
uv run --refresh --with git+https://github.com/SushantGautam/vllm-RAG.git -- setup
```

2. Edit `./.env` and add sensitive keys: `OPENAI_API_KEY`, `EMBEDDING_API_KEY`, `API_SECRET`.

3. Ingest your documents (indexes `./documents` into `./milvus_demo.db`):

```bash
uv run --refresh --with git+https://github.com/SushantGautam/vllm-RAG.git -- ingest-document
# Or pass a key directly (not recommended for long-term use):
uv run --refresh --with git+https://github.com/SushantGautam/vllm-RAG.git -- ingest-document --openai-api-key "$OPENAI_API_KEY"
```

4. Start the server:

```bash
uv run --refresh --with git+https://github.com/SushantGautam/vllm-RAG.git rag-server
```

Note: When using `--with` or running the console scripts locally, `uv` runs commands in your current working directory and will pick up the `./.env` file you created, so the `setup`-generated `.env` will be used automatically.

---

## Configuration & Notes 🔧

- Default host/port: `0.0.0.0:8000`.
- Default Milvus DB file: `./milvus_demo.db`.
- Add `OPENAI_API_KEY` and any secrets to your `.env` (do not commit secrets to source control).
- If you set `API_SECRET`, include it in requests using `Authorization: Bearer $API_SECRET`.
- Console scripts (defined in `pyproject.toml`):
  - `rag-server` → `rag_server:main`
  - `ingest-document` → `ingest_documents:main`
  - `setup` → `setup:main`

---

## API Overview

- POST `/v1/chat/completions` — OpenAI-compatible chat completion endpoint (see Swagger UI for full schema).
- GET `/health` — Health check.
- Interactive docs: `/docs` (Swagger), `/redoc` (ReDoc).

---

## Adding Documents

- Add `.txt` files to `./documents` (or point ingestion to another folder with `--documents-path`).
- Re-run ingestion to update the DB (`--recreate` to rebuild from scratch) and restart the server.

---

## Architecture (short)

- `ingest_documents.py` — loads documents, splits them, builds embeddings and stores vectors in a local Milvus file.
- `rag_server.py` — loads Milvus DB, initializes retriever and LLM, serves OpenAI-compatible API.

---

## License

MIT


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