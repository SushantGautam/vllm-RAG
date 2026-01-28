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


def prompt(value_name: str, default: str) -> str:
    prompt_text = f"{value_name} [{default}]: "
    try:
        resp = input(prompt_text).strip()
    except EOFError:
        # Non-interactive environment: return default
        return default
    return resp if resp != "" else default


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive .env setup from .env.example")
    parser.add_argument(
        "--example",
        default=".",
        help="Path to the example env file. Use '.' (default) to copy the bundled example packaged with the library",
    )

    parser.add_argument(
        "--dest",
        default=".env",
        help="Destination .env path (default: ./.env)",
    )
    args = parser.parse_args()

    example_path = args.example
    dest_path = args.dest

    # Resolve example path: allow default '.' to mean the packaged .env.example next to this module
    packaged_example = os.path.join(os.path.dirname(__file__), ".env.example")
    if example_path in (".", "./"):
        if os.path.exists(packaged_example):
            example_path = packaged_example
        else:
            # try local .env.example in current working dir
            local_example = ".env.example"
            if os.path.exists(local_example):
                example_path = local_example
            else:
                print(f"Error: example file not found at {packaged_example} or {local_example}")
                return
    elif not os.path.exists(example_path):
        # If provided a path that doesn't exist, try the packaged example as a fallback
        if os.path.exists(packaged_example):
            example_path = packaged_example
        else:
            print(f"Error: example file not found at {example_path}")
            return


    ex_lines, ex_map = read_env_file(example_path)
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
        new_val = prompt(key, default)
        new_values[key] = new_val

    # Ask user where to write the file (confirm or override dest)
    dest_input = input(f"Write configuration to file [default: {dest_path}]: ").strip()
    if dest_input:
        dest_path = dest_input

    # If dest exists and has other variables, keep them unless overwritten by new_values
    dest_exists = os.path.exists(dest_path)
    if dest_exists:
        print(f"\nNote: {dest_path} already exists. Values you choose below will override existing keys listed in the prompts.")

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

    print("\nConfiguration written to:", dest_path)
    print("\nPlease edit the file to add sensitive values such as OPENAI_API_KEY, EMBEDDING_API_KEY, and any API secrets (they are intentionally not prompted here).")
    print("You can re-run this setup script to override values at any time: \n\n    python setup.py\n")


if __name__ == "__main__":
    main()
