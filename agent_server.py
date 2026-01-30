from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from agent_test import clean_page_text
import os
import asyncio
from multiprocessing import Process, SimpleQueue
from typing import Optional

app = FastAPI(title="Agent Minimal Server")

from dotenv import load_dotenv
load_dotenv()

# Allow local browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (JS/CSS/preview + documents)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR, exist_ok=True)

# --- Security: require a password / API key to access the server ---
# Configure via environment variable:
# - AGENT_SERVER_PASSWORD: the secret API key/password required (default provided)
import secrets as _secrets
import base64 as _base64
from starlette.middleware.base import BaseHTTPMiddleware

AGENT_SERVER_PASSWORD = os.environ.get("AGENT_SERVER_PASSWORD")

if not AGENT_SERVER_PASSWORD:
    import warnings
    warnings.warn("AGENT_SERVER_PASSWORD is not set. Set AGENT_SERVER_PASSWORD to enable authentication.")


def _extract_key(auth_header_or_key):
    """Extract API key from several supported formats."""
    if not auth_header_or_key:
        return None
    v = auth_header_or_key.strip()
    # Bearer: "Bearer <key>"
    if v.lower().startswith("bearer "):
        return v[7:].strip()
    # Basic: base64("user:password") -> return password part
    if v.lower().startswith("basic "):
        try:
            b64 = v.split(" ", 1)[1]
            decoded = _base64.b64decode(b64).decode("utf-8")
            if ":" in decoded:
                return decoded.split(":", 1)[1]
        except Exception:
            return None
    # If the header was already the key (e.g., X-API-Key: <key>), return as-is
    return v


def _is_valid_key(key):
    if not key or not AGENT_SERVER_PASSWORD:
        return False
    return _secrets.compare_digest(key, AGENT_SERVER_PASSWORD)


class _AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Allow certain public paths to be served without auth so the preview UI and assets can load
        # This lets the UI present a password prompt and then send the key on subsequent requests.
        if request.url.path.startswith("/static") or request.url.path in ("/openapi.json", "/docs", "/redoc", "/preview"):
            return await call_next(request)

        # Allow explicitly-decorated public endpoints (use @public_endpoint)
        endpoint = request.scope.get("endpoint")
        # If routing hasn't set the endpoint yet, defer authentication to route-level dependencies
        if endpoint is None:
            return await call_next(request)
        if getattr(endpoint, "_public", False):
            return await call_next(request)

        # Look for key in X-API-Key or Authorization header
        key = request.headers.get("x-api-key") or request.headers.get("authorization")
        key = _extract_key(key)
        if not _is_valid_key(key):
            return JSONResponse(status_code=401, content={"detail": "Unauthorized: valid API key required"})
        return await call_next(request)


app.add_middleware(_AuthMiddleware)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class ScenarioRequest(BaseModel):
    scenario: str


def _worker_invoke(state0, out_q):
    """Worker invoked in a separate process to run the agent workflow and put a serialized result on the queue.

    We import `agent_test` locally here to avoid performing heavy/oracle imports during module import time
    (which can fail when Python's multiprocessing spawn mechanism re-imports the module).
    Any exception (including import/connection failures) is caught and serialized back via the queue.
    """
    try:
        # local import to avoid import-time side effects during spawn
        import agent_test
        agent_app = getattr(agent_test, 'app', None)
        assemble_fn = getattr(agent_test, 'assemble_fn', None)
        if agent_app is None or assemble_fn is None:
            out_q.put({"_error": "agent_test missing required attributes (app/assemble_fn)"})
            return

        final_state = agent_app.invoke(state0)
        output = assemble_fn(final_state)
        out_q.put(output.model_dump())
    except Exception as e:
        out_q.put({"_error": repr(e)})


@app.post("/query")
async def run_scenario(req: ScenarioRequest, request: Request):
    scenario = req.scenario.strip()
    if not scenario:
        raise HTTPException(status_code=400, detail="scenario cannot be empty")
    state0 = {
        "scenario": scenario,
        "search_plan": None,
        "retrieved_chunks": [],
        "candidate_chunks": [],
        "relevant_chunks": [],
        "considerations": [],
        "qa_pairs": [],
    }

    # start worker process to run the potentially long workflow
    q = SimpleQueue()
    p = Process(target=_worker_invoke, args=(state0, q))
    p.start()
    try:
        while p.is_alive():
            # if the client disconnected, terminate the worker process
            if await request.is_disconnected():
                p.terminate()
                p.join(timeout=1)
                raise HTTPException(status_code=499, detail="Client disconnected")
            await asyncio.sleep(0.1)

        # worker finished; gather result
        if not q.empty():
            res = q.get()
            if isinstance(res, dict) and res.get("_error"):
                raise HTTPException(status_code=500, detail=f"Worker error: {res.get('_error')}")
            return JSONResponse(content=res)
        else:
            raise HTTPException(status_code=500, detail="Worker finished but produced no result")
    finally:
        if p.is_alive():
            p.terminate()
            p.join(timeout=1)


@app.get("/preview", response_class=HTMLResponse)
def preview_ui():
    """Serve the interactive preview UI."""
    preview_path = os.path.join(STATIC_DIR, "preview.html")
    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="preview UI not found")
    with open(preview_path, "r", encoding="utf-8") as fh:
        return HTMLResponse(fh.read())


# @app.get("/doc_list")
# async def doc_list():
#     """Return a list of markdown files available in /static/documents"""
#     docs_dir = os.path.join(STATIC_DIR, "documents")
#     if not os.path.exists(docs_dir):
#         return JSONResponse(content={"files": []})
#     files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".md")]
#     return JSONResponse(content={"files": sorted(files)})

import os, unicodedata
def norm_path(p):
    for f in ("NFC", "NFD"):
        n = unicodedata.normalize(f, p)
        if os.path.exists(n):
            return n
    return None


def public_endpoint(func):
    """Decorator to mark a route as public (no API key required). Can be applied before or after route decorators."""
    setattr(func, "_public", True)
    return func

@app.get("/doc_content")
async def doc_content(file: str = Query(..., description="Filename (with or without .md)"), start: Optional[int] = None, end: Optional[int] = None):
    """Return file content as numbered lines and optionally a slice (1-indexed inclusive)."""
    # Sanitize file name and enforce .md
    base = os.path.basename(file)
    if not base.lower().endswith(".md"):
        base = base + ".md"

    docs_dir = os.path.join(STATIC_DIR, "documents")
    fpath = os.path.join(docs_dir, base)
    real = norm_path(fpath)
    if not real:
        raise HTTPException(status_code=404, detail=f"Document {base} not found")

    with open(real, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    # Use the same line-count logic as ingest_documents.py (newline positions -> line numbers)
    from bisect import bisect_right
    nl = [i for i, c in enumerate(raw) if c == "\n"]
    ln = lambda p: bisect_right(nl, p) + 1

    # Clean the text for presentation, but split on "\n" so trailing empty lines are preserved
    t = clean_page_text(raw)
    lines = t.split("\n")

    total = len(lines)
    # normalize start/end
    s = 1 if start is None else max(1, int(start))
    e = total if end is None else min(total, int(end))
    s = min(s, total)
    e = max(s, e)

    # prepare indexed lines
    indexed = [{"line_no": idx + 1, "text": lines[idx]} for idx in range(s - 1, e)]

    return JSONResponse(content={"file": base, "total_lines": total, "start": s, "end": e, "lines": indexed})


@public_endpoint
@app.get("/")
def root():
    return {"status": "ok", "message": "Server is up. Contact sushant@simula.no for instructions."}

# Dependency-based auth check used for routes (executes after routing, so endpoint is available)
from fastapi import Depends
async def require_api_key(request: Request):
    # Allow public static paths
    if request.url.path.startswith("/static") or request.url.path in ("/openapi.json", "/docs", "/redoc", "/preview"):
        return
    endpoint = request.scope.get("endpoint")
    if endpoint and getattr(endpoint, "_public", False):
        return
    key = request.headers.get("x-api-key") or request.headers.get("authorization")
    key = _extract_key(key)
    if not _is_valid_key(key):
        raise HTTPException(status_code=401, detail="Unauthorized: valid API key required")

# Attach `require_api_key` as a dependency to all API routes except those marked public or special paths.
def _attach_auth_dependency():
    from fastapi.routing import APIRoute
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if route.path.startswith("/static") or route.path in ("/openapi.json", "/docs", "/redoc", "/preview"):
            continue
        if getattr(route.endpoint, "_public", False):
            continue
        # avoid adding duplicate
        if any(getattr(d, 'dependency', None) == require_api_key for d in route.dependencies):
            continue
        route.dependencies.append(Depends(require_api_key))

_attach_auth_dependency()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent_server:app", host="0.0.0.0", port=8001, reload=True)
