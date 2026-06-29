# AGENTS.md

## Project Overview
MiniSearch is a lightweight full-text search engine library and CLI tool written in Python. It provides BM25-ranked search over text documents with boolean queries, phrase matching, and fuzzy search.

## Architecture
- **tokenizer.py** — Porter stemmer, stop words, camelCase splitting, position tracking
- **index.py** — Inverted index with document/term/posting data structures, JSON serialization
- **scoring.py** — BM25 scoring formula implementation
- **query.py** — Recursive descent parser for boolean/phrase/fuzzy/prefix queries
- **fuzzy.py** — Levenshtein and Damerau-Levenshtein distance (dynamic programming)
- **storage.py** — SQLite persistence with WAL mode for concurrent reads
- **search.py** — Orchestrates tokenization → indexing → querying → scoring
- **cli.py** — argparse-based CLI with index/search/info/clear/export commands

## Key Algorithms
1. **Porter Stemmer** — Rule-based suffix stripping (5 steps)
2. **BM25 IDF** — log((N - df + 0.5) / (df + 0.5) + 1)
3. **BM25 TF normalization** — tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl/avgdl))
4. **Levenshtein DP** — O(n*m) time, O(min(n,m)) space
5. **Query Parser** — Recursive descent with operator precedence: NOT > AND > OR

## Running Tests
```bash
cd /root/workspace/minisearch
pip install -e ".[dev]"
pytest -v
```

## Code Conventions
- Type hints on all public methods
- Docstrings on all public classes/functions
- Use `from __future__ import annotations` for modern type syntax
- Tests use pytest, organized by module
- No external dependencies (pure Python, only stdlib + sqlite3)
