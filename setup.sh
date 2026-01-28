#!/bin/bash
# Setup script for RAG FastAPI Server

set -e

echo "========================================"
echo "RAG FastAPI Server Setup"
echo "========================================"
echo

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Error: Python 3 not found. Please install Python 3.7+"; exit 1; }
echo "✓ Python 3 found"
echo

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo

# Check for .env file
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "⚠ Please edit .env and add your OPENAI_API_KEY"
else
    echo "✓ .env file exists"
fi
echo

# Check for OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ] && ! grep -q "OPENAI_API_KEY=your-openai-api-key-here" .env 2>/dev/null; then
    echo "⚠ Warning: OPENAI_API_KEY not set"
    echo "  Set it with: export OPENAI_API_KEY='your-key'"
    echo "  Or add it to the .env file"
else
    echo "✓ OpenAI API key configured"
fi
echo

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "To start the server, run:"
echo "  python rag_server.py"
echo
echo "Or with custom options:"
echo "  python rag_server.py --host 0.0.0.0 --port 8000"
echo
echo "Interactive API docs will be available at:"
echo "  http://localhost:8000/docs"
echo
echo "Note: The server will create a local Milvus database"
echo "      file (milvus_demo.db) automatically."
echo
