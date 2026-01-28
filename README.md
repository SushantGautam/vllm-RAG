# RAG-infer ✅

Quick steps to set up and run the RAG demo.

## Steps to run

1) Run setup:

```bash
uv run --refresh --with git+https://github.com/SushantGautam/RAG-infer.git -- setup
```

- This will create the `documents/` directory. Add the files you want RAG to access into that folder.

2) Ingest documents:

```bash
uv run --refresh --with git+https://github.com/SushantGautam/RAG-infer.git -- ingest_documents
```

3) Start the server:

```bash
uv run --refresh --with git+https://github.com/SushantGautam/RAG-infer.git -- start-server
```

## Test a query

If the server starts, test with:

```bash
curl -s -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer no_need" \
  -d '{"model": "gpt-3.5-turbo",
    "messages": [{"role":"user","content":"Give a one-sentence summary of FastAPI RAG."}]}'
```

---

**Notes:** Add documents to `documents/` before ingesting. Keep it simple and enjoy!