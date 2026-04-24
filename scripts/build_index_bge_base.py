"""
build_index_bge_base.py
~~~~~~~~~~~~~~~~~~~~~~~
Build vector index with bge-base-zh-v1.5 (768-dim) instead of bge-small-zh-v1.5 (384-dim).
Reuses the same chunking logic from build_index_bge.py, only changes embedding model and output dir.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import build_index_bge as base

# Override constants
base.CHROMA_DIR = "./chroma_db_bge_base"
base.EMBEDDING_MODEL = "BAAI/bge-base-zh-v1.5"

if __name__ == "__main__":
    base.main()
