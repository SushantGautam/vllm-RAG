"""
FastAPI server for LangChain RAG with Milvus vector store.

This server initializes document loading, splitting, Milvus vector store,
retriever, prompt, and ChatOpenAI once at startup. It exposes a POST /ask
endpoint for question answering and supports parallel requests safely.
"""

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_community.vectorstores import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Global variables to store initialized components
qa_chain = None
vectorstore = None


class QuestionRequest(BaseModel):
    """Request model for the /ask endpoint."""
    question: str = Field(..., description="The question to ask the RAG system")


class AnswerResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str = Field(..., description="The answer from the RAG system")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FastAPI server for LangChain RAG")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)",
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
        help="OpenAI API key (default: reads from OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model name (default: gpt-3.5-turbo)",
    )
    return parser.parse_args()


def initialize_rag_system(args):
    """
    Initialize the RAG system by connecting to existing Milvus vector store,
    setting up retriever, prompt, and ChatOpenAI.
    
    Note: Documents should be pre-loaded using ingest_documents.py
    """
    global qa_chain, vectorstore
    
    print("Initializing RAG system...")
    
    # Validate OpenAI API key
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    
    if "OPENAI_API_KEY" not in os.environ:
        raise ValueError(
            "OpenAI API key must be provided via --openai-api-key or OPENAI_API_KEY env var"
        )
    
    # Initialize embeddings
    print("Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings()
    
    # Connect to existing Milvus vector store
    print(f"Connecting to Milvus database at {args.milvus_db}...")
    
    if not os.path.exists(args.milvus_db):
        raise ValueError(
            f"Milvus database not found at {args.milvus_db}. "
            f"Please run 'python ingest_documents.py' first to create and populate the database."
        )
    
    vectorstore = Milvus(
        embedding_function=embeddings,
        collection_name=args.collection_name,
        connection_args={"uri": args.milvus_db},
    )
    print("✓ Connected to vector store")
    
    # Initialize retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}
Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )
    
    # Initialize ChatOpenAI
    print(f"Initializing ChatOpenAI with model {args.model_name}...")
    llm = ChatOpenAI(model_name=args.model_name, temperature=0)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=False,
    )
    
    print("RAG system initialized successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup: Initialize RAG system
    initialize_rag_system(app.state.args)
    yield
    # Shutdown: Clean up resources
    global vectorstore
    if vectorstore:
        print("Cleaning up vector store...")
        # Milvus connections are cleaned up automatically when the process exits


# Create FastAPI app with lifespan
app = FastAPI(
    title="LangChain RAG API",
    description="FastAPI server for question answering using LangChain RAG with Milvus",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "LangChain RAG API",
        "version": "1.0.0",
        "endpoints": {
            "/ask": "POST - Ask a question to the RAG system",
            "/health": "GET - Health check endpoint",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "rag_initialized": qa_chain is not None}


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the RAG system.
    
    Supports parallel requests by running qa_chain.invoke() in a thread pool
    to avoid blocking the async event loop.
    """
    if qa_chain is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Run the synchronous qa_chain.invoke() in a thread pool
        # to support parallel requests without blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # Use default executor (ThreadPoolExecutor)
            qa_chain.invoke,
            {"query": request.question}
        )
        
        answer = result.get("result", "No answer generated")
        return AnswerResponse(answer=answer)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


def main():
    """Main entry point for the application."""
    import uvicorn
    
    # Parse command line arguments
    args = parse_args()
    
    # Store args in app state so they're accessible in lifespan
    app.state.args = args
    
    # Run the server with uvicorn
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
