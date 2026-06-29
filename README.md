# MiniSearch

A lightweight full-text search engine for Python with BM25 ranking, boolean queries, and fuzzy matching.

## Features

- **BM25 Ranking** — Industry-standard relevance scoring (used by Elasticsearch, Lucene, etc.)
- **Boolean Queries** — AND, OR, NOT operators with parentheses for complex queries
- **Phrase Queries** — Exact phrase matching with positional tracking
- **Prefix Queries** — Wildcard matching with `*` (e.g., `hel*`)
- **Fuzzy Queries** — Approximate matching with Levenshtein distance (e.g., `hello~2`)
- **Persistent Index** — SQLite-backed storage for fast repeated searches
- **Multi-format Support** — Index text, code, markdown, and config files
- **Porter Stemming** — Automatic word normalization for better recall
- **Stop Word Removal** — Filters common English words
- **Code-aware Tokenization** — Splits camelCase, snake_case, and kebab-case

## Quick Start

### Install

```bash
pip install -e .
```

### Index Files

```bash
# Index a directory
minisearch index ./docs

# Index specific files
minisearch index readme.md notes.txt

# Index with custom extensions
minisearch index ./src --extensions .py .js .ts
```

### Search

```bash
# Simple search
minisearch search "hello world"

# Boolean query
minisearch search "python AND (async OR await)"

# Phrase query
minisearch search '"exact phrase match"'

# Fuzzy query (max 2 edits)
minisearch search "pytho~2"

# Prefix query
minisearch search "hel*"

# JSON output
minisearch search "hello" --json

# Limit results
minisearch search "hello" -n 5
```

### Manage Index

```bash
# Show index statistics
minisearch info

# Export index as JSON
minisearch export -o index.json

# Clear index
minisearch clear -y
```

## Python API

```python
from minisearch import SearchEngine

# Create engine with persistent storage
engine = SearchEngine(index_path="my_index.db")

# Index text
engine.index_text("Hello world", "greeting.txt")
engine.index_text("Goodbye world", "farewell.txt")

# Index files
engine.index_file("path/to/document.txt")
engine.index_directory("path/to/docs/")

# Search
results = engine.search("hello")
for result in results:
    print(f"{result.path}: {result.score:.3f}")

# Boolean queries
results = engine.search("hello OR goodbye")

# Phrase queries
results = engine.search('"hello world"')

# Save index
engine.save()
```

## Query Syntax

| Syntax | Description | Example |
|--------|-------------|---------|
| `term` | Simple term | `hello` |
| `term1 term2` | Implicit AND | `hello world` |
| `term1 AND term2` | Explicit AND | `hello AND world` |
| `term1 OR term2` | OR | `hello OR world` |
| `NOT term` | Negation | `NOT goodbye` |
| `"phrase"` | Exact phrase | `"hello world"` |
| `term~N` | Fuzzy (max N edits) | `hello~2` |
| `prefix*` | Prefix wildcard | `hel*` |
| `(expr)` | Grouping | `(a OR b) AND c` |

## Architecture

```
minisearch/
├── src/minisearch/
│   ├── __init__.py        # Package init
│   ├── tokenizer.py       # Porter stemmer, stop words, tokenization
│   ├── index.py           # Inverted index data structure
│   ├── scoring.py         # BM25 relevance scoring
│   ├── query.py           # Recursive descent query parser
│   ├── fuzzy.py           # Levenshtein distance algorithms
│   ├── storage.py         # SQLite persistence
│   ├── search.py          # Search engine orchestrator
│   └── cli.py             # Command-line interface
└── tests/                 # Test suite
```

### Core Algorithms

1. **Inverted Index** — Maps terms to document IDs with positional information and term frequencies
2. **BM25 Scoring** — Probabilistic ranking based on term frequency and inverse document frequency
3. **Recursive Descent Parser** — Parses boolean query expressions into an AST
4. **Levenshtein Distance** — Dynamic programming algorithm for fuzzy string matching
5. **Porter Stemmer** — Rule-based suffix stripping for word normalization

## Performance

MiniSearch is designed for small-to-medium document collections (up to ~100K documents). For larger collections, consider Elasticsearch or Meilisearch.

Benchmark (1000 documents, ~500K tokens):
- Index time: ~2 seconds
- Search time: ~5ms per query
- Index size: ~10MB on disk

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=minisearch

# Lint
ruff check src/ tests/
```

## License

MIT
