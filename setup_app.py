#!/usr/bin/env python3
"""Interactive setup script to create/overwrite a .env from .env.example.

Usage: python setup_app.py
This script will copy values from .env.example and prompt you to accept or override
selected configuration values. It preserves comments and ordering from the example.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from typing import Dict, List, Tuple

KEY_RE = re.compile(r"^([A-Z0-9_]+)=(.*)$")

# Keys we will explicitly prompt the user for
PROMPT_KEYS = [
    "HOST",
    "PORT",
    "MILVUS_DB",
    "COLLECTION_NAME",
    "DOCUMENTS_PATH",
    "MODEL_NAME",
    "EMBEDDING_MODEL_NAME",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
]

# Fallback default example content (used when the packaged or local .env.example is not available)
DEFAULT_EXAMPLE = """
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# OpenAI-Compatible API Configuration (optional)
# Uncomment and set these if using a custom OpenAI-compatible endpoint
# OPENAI_BASE_URL=https://api.openai.com/v1
# EMBEDDING_BASE_URL=https://api.openai.com/v1
# EMBEDDING_API_KEY=your-embedding-api-key-here

# API_SECRET=your-api-secret-here  

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Milvus Configuration (local file-based)
MILVUS_DB=./milvus_demo.db
COLLECTION_NAME=rag_collection

# Document Configuration
DOCUMENTS_PATH=./documents

# Model Configuration
MODEL_NAME=gpt-3.5-turbo
EMBEDDING_MODEL_NAME=text-embedding-ada-002
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
"""


def read_env_file(path: str) -> Tuple[List[str], Dict[str, str]]:
    """Return file lines and a mapping of key->value for assignments found."""
    lines: List[str] = []
    mapping: Dict[str, str] = {}
    if not os.path.exists(path):
        return lines, mapping
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            # keep original lines for reconstructing with minimal changes
            lines.append(line.rstrip("\n"))
            m = KEY_RE.match(line.strip())
            if m:
                k, v = m.group(1), m.group(2)
                mapping[k] = v
    return lines, mapping


def prompt(value_name: str, default: str, display_default: str | None = None) -> str:
    """Prompt the user, showing `display_default` if provided but returning `default` when accepted."""
    shown = display_default if display_default is not None else default
    prompt_text = f"{value_name} [{shown}]: "
    try:
        resp = input(prompt_text).strip()
    except EOFError:
        # Non-interactive environment: return default
        return default
    return resp if resp != "" else default


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive .env setup using embedded defaults")
    parser.add_argument(
        "--dest",
        default=".env",
        help="Destination .env path (default: ./.env)",
    )
    args = parser.parse_args()

    dest_path = args.dest

    # Use embedded DEFAULT_EXAMPLE as the source for prompts
    ex_lines = [line.rstrip("\n") for line in DEFAULT_EXAMPLE.strip().splitlines()]
    ex_map = {}
    for line in ex_lines:
        m = KEY_RE.match(line.strip())
        if m:
            ex_map[m.group(1)] = m.group(2)
    _, existing_map = read_env_file(dest_path) if os.path.exists(dest_path) else ([], {})

    print("\nInteractive setup — enter values or press Enter to accept defaults.")
    print("(Values in [] are the current defaults or existing values.)\n")

    # Determine new values for prompt keys
    new_values: Dict[str, str] = {}
    for key in PROMPT_KEYS:
        default = existing_map.get(key, ex_map.get(key, ""))
        # strip surrounding quotes if present
        if default.startswith('"') and default.endswith('"'):
            default = default[1:-1]

        # For path-like defaults (containing '/' or starting with '.' or '/'), display absolute path only
        display_default = None
        if default:
            if "/" in default or default.startswith(".") or default.startswith("/"):
                display_default = os.path.abspath(default)
            else:
                display_default = default

        new_val = prompt(key, default, display_default)
        new_values[key] = new_val

    # Ask user where to write the file (confirm or override dest)
    full_dest = os.path.abspath(dest_path)
    try:
        dest_input = input(f"Write configuration to file [default: {full_dest}]: ").strip()
    except EOFError:
        # Non-interactive environment: accept the default
        dest_input = ""
    if dest_input:
        dest_path = dest_input

    # If dest exists and has other variables, keep them unless overwritten by new_values
    dest_exists = os.path.exists(dest_path)
    if dest_exists:
        print(f"\nNote: {os.path.abspath(dest_path)} already exists. Values you choose below will override existing keys listed in the prompts.")

    # Reconstruct output lines by iterating over example lines and replacing prompted keys
    out_lines: List[str] = []
    for line in ex_lines:
        m = KEY_RE.match(line.strip())
        if m:
            k = m.group(1)
            if k in new_values:
                out_lines.append(f"{k}={new_values[k]}")
            else:
                # If dest exists and has a value for this key, prefer dest's value; otherwise use example's
                if dest_exists and k in existing_map:
                    out_lines.append(f"{k}={existing_map[k]}")
                else:
                    out_lines.append(line)
        else:
            out_lines.append(line)

    # If example did not have some of the PROMPT_KEYS, append them
    for k in PROMPT_KEYS:
        keys_in_example = any(KEY_RE.match(l.strip()) and KEY_RE.match(l.strip()).group(1) == k for l in ex_lines)
        if not keys_in_example:
            out_lines.append(f"{k}={new_values[k]}")

    # Write the destination file (create parent dir if necessary)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)

    # Backup existing file if present
    if dest_exists:
        backup_path = dest_path + ".bak"
        shutil.copy2(dest_path, backup_path)
        print(f"Existing file backed up to {backup_path}")

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
        f.write("\n")

    print("\nConfiguration written to:", os.path.abspath(dest_path))
    print("\nPlease edit the file to add sensitive values such as OPENAI_API_KEY, EMBEDDING_API_KEY, and any API secrets (they are intentionally not prompted here).")
    print("You can re-run this setup script to override values at any time: \n\n    python setup.py\n")

    # Ensure DOCUMENTS_PATH exists and encourage users to add documents there before ingest
    doc_path = new_values.get("DOCUMENTS_PATH") or existing_map.get("DOCUMENTS_PATH") or ex_map.get("DOCUMENTS_PATH")
    if doc_path:
        abs_doc_path = os.path.abspath(doc_path)
        try:
            os.makedirs(abs_doc_path, exist_ok=True)
            readme_path = os.path.join(abs_doc_path, "README.md")
            if not os.path.exists(readme_path):
                with open(readme_path, "w", encoding="utf-8") as rf:
                    rf.write("# Documents directory\n\nAdd your text documents (e.g., .txt files) here before running `ingest_documents`.\n")
            print(f"\nDocuments directory ensured at: {abs_doc_path}")
            print("→ Add your documents (e.g., .txt files) into that directory before running `ingest_documents` to ingest them.")
        except Exception as e:
            print(f"\n⚠ Could not create documents directory {abs_doc_path}: {e}")


if __name__ == "__main__":
    main()
