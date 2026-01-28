# Quick Start Guide

## Prerequisites

- Python 3.7+
- Docker (for Milvus)
- OpenAI API Key

## Setup Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Milvus

Using Docker Compose:
```bash
docker-compose up -d
```

Or using a simple Docker command:
```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest
```

### 3. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your API key
```

### 4. Start the Server

```bash
python rag_server.py
```

The server will start on http://localhost:8000

### 5. Test the Server

#### Using the web browser:
Visit http://localhost:8000/docs for interactive API documentation

#### Using curl:
```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is FastAPI?"}'
```

#### Using the provided client:
```bash
# Interactive mode
python client.py

# Single question
python client.py --question "What is FastAPI?"

# Health check
python client.py --health
```

## Configuration

All configuration options can be set via command line arguments:

```bash
python rag_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --documents-path ./documents \
  --milvus-host localhost \
  --milvus-port 19530 \
  --model-name gpt-3.5-turbo
```

See `python rag_server.py --help` for all options.

## Adding Your Own Documents

1. Place your `.txt` files in the `documents` directory
2. Restart the server
3. The documents will be automatically loaded and indexed

## Troubleshooting

### Milvus Connection Error
- Make sure Milvus is running: `docker ps | grep milvus`
- Check Milvus logs: `docker logs milvus-standalone`

### OpenAI API Error
- Verify your API key is set correctly
- Check your OpenAI account has credits

### Server Won't Start
- Check if port 8000 is already in use
- Try a different port: `python rag_server.py --port 8001`

## Architecture Overview

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ POST /ask
       ▼
┌─────────────────────────────────────────┐
│         FastAPI Server                  │
│  ┌──────────────────────────────────┐  │
│  │   Lifespan (Startup)             │  │
│  │   - Load documents               │  │
│  │   - Split into chunks            │  │
│  │   - Create Milvus vector store   │  │
│  │   - Initialize retriever         │  │
│  │   - Create prompt template       │  │
│  │   - Initialize ChatOpenAI        │  │
│  │   - Build QA chain               │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   POST /ask Endpoint             │  │
│  │   - Receive question             │  │
│  │   - Run in thread pool           │  │
│  │   - Return answer                │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
         │                    │
         ▼                    ▼
   ┌─────────┐          ┌──────────┐
   │  Milvus │          │  OpenAI  │
   │ (Vector │          │   API    │
   │  Store) │          └──────────┘
   └─────────┘
```

## Next Steps

- Add more document formats (PDF, Word, etc.)
- Implement authentication
- Add rate limiting
- Set up monitoring and logging
- Deploy to production
