"""
MiniSearch: Lightweight full-text search engine with BM25 ranking.

Features:
- Inverted index with positional information
- BM25 relevance scoring
- Boolean query parsing (AND, OR, NOT, parentheses)
- Phrase queries ("exact phrase matching")
- Fuzzy queries (Levenshtein distance tolerance)
- Persistent index (SQLite-backed)
- CLI and library API
"""

__version__ = "0.1.0"

from minisearch.index import InvertedIndex
from minisearch.query import QueryParser
from minisearch.search import SearchEngine
from minisearch.tokenizer import Tokenizer

__all__ = ["SearchEngine", "InvertedIndex", "Tokenizer", "QueryParser"]
