# Quick Start Guide

## Prerequisites

- Python 3.7+
- OpenAI API Key

**No Docker needed!** The application uses a local file-based Milvus database.

## Setup Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env and add your API key
```

### 3. Start the Server

```bash
python rag_server.py
```

The server will:
- Load documents from the `./documents` directory
- Create a local Milvus database file (`milvus_demo.db`)
- Start on http://localhost:8000

That's it! No external services or Docker required.

### 4. Test the Server

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
  --milvus-db ./milvus_demo.db \
  --model-name gpt-3.5-turbo
```

See `python rag_server.py --help` for all options.

## Adding Your Own Documents

1. Place your `.txt` files in the `documents` directory
2. Restart the server
3. The documents will be automatically loaded and indexed

## Troubleshooting

### OpenAI API Error
- Verify your API key is set correctly
- Check your OpenAI account has credits

### Server Won't Start
- Check if port 8000 is already in use
- Try a different port: `python rag_server.py --port 8001`

### Database Issues
- Delete `milvus_demo.db` to start fresh
- Check file permissions in the directory

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
│  │   - Create local Milvus DB       │  │
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
   ┌─────────────┐      ┌──────────┐
   │ milvus_demo │      │  OpenAI  │
   │    .db      │      │   API    │
   │ (local file)│      └──────────┘
   └─────────────┘
```

## Next Steps

- Add more document formats (PDF, Word, etc.)
- Implement authentication
- Add rate limiting
- Set up monitoring and logging
- Deploy to production
