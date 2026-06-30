"""
Search engine that ties together tokenization, indexing, querying, and scoring.

The SearchEngine is the main entry point for using MiniSearch. It orchestrates:
1. Tokenizing documents
2. Building/maintaining the inverted index
3. Parsing and executing queries
4. Scoring and ranking results with BM25
5. Persistent storage via SQLite
"""

from __future__ import annotations

import os
from pathlib import Path

from minisearch.fuzzy import find_closest_matches
from minisearch.index import InvertedIndex
from minisearch.query import (
    QueryNode,
    QueryNodeType,
    QueryParser,
    normalize_query_term,
)
from minisearch.scoring import BM25Scorer, ScoredDocument
from minisearch.storage import SearchStorage
from minisearch.tokenizer import Tokenizer, TokenizeResult

# Default file extensions to index
DEFAULT_INDEX_EXTENSIONS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".scala",
    ".html",
    ".css",
    ".scss",
    ".less",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".csv",
    ".tsv",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".r",
    ".lua",
    ".perl",
    ".pl",
}

# Directories to skip during indexing
SKIP_DIRECTORIES = {
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    "venv",
    ".venv",
    "env",
    ".env",
    "dist",
    "build",
    ".eggs",
    ".idea",
    ".vscode",
    "target",  # Rust
    "vendor",  # Go
}


class SearchResult:
    """A single search result with context."""

    def __init__(
        self,
        path: str,
        score: float,
        title: str | None = None,
        snippet: str | None = None,
        line_number: int | None = None,
        metadata: dict | None = None,
    ):
        self.path = path
        self.score = score
        self.title = title
        self.snippet = snippet
        self.line_number = line_number
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"SearchResult(path={self.path!r}, score={self.score}, line={self.line_number})"


class SearchEngine:
    """
    Full-text search engine with BM25 ranking.

    Main interface for indexing and searching documents.

    Usage:
        engine = SearchEngine(index_path="my_index.db")

        # Index documents
        engine.index_file("path/to/document.txt")
        engine.index_directory("path/to/docs/")

        # Search
        results = engine.search("hello world")
        for result in results:
            print(f"{result.path}:{result.line_number} (score={result.score})")
            print(f"  {result.snippet}")
    """

    def __init__(
        self,
        index_path: str | Path | None = None,
        tokenizer: Tokenizer | None = None,
        scorer: BM25Scorer | None = None,
        extensions: set[str] | None = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB default
    ):
        """
        Initialize the search engine.

        Args:
            index_path: Path to the SQLite index file. If None, uses in-memory only.
            tokenizer: Custom tokenizer. If None, uses default with stemming.
            scorer: Custom BM25 scorer. If None, uses default parameters.
            extensions: Set of file extensions to index. If None, uses defaults.
            max_file_size: Maximum file size to index (in bytes).
        """
        self.index = InvertedIndex()
        self.tokenizer = tokenizer or Tokenizer()
        self.scorer = scorer or BM25Scorer()
        self.extensions = extensions or DEFAULT_INDEX_EXTENSIONS
        self.max_file_size = max_file_size
        self._storage: SearchStorage | None = None

        if index_path is not None:
            self._storage = SearchStorage(index_path)
            if self._storage.exists():
                self.index = self._storage.load()

    def save(self) -> None:
        """Save the index to persistent storage."""
        if self._storage is not None:
            self._storage.save(self.index)

    def index_text(
        self,
        text: str,
        path: str,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """
        Index a text string.

        Args:
            text: The text content to index.
            path: Unique identifier for this document.
            title: Optional display title.
            metadata: Optional metadata dict.

        Returns:
            Document ID.
        """
        result: TokenizeResult = self.tokenizer.tokenize(text)
        tokens = [t.key for t in result.tokens]

        doc_id = self.index.add_document(
            path=path,
            tokens=tokens,
            doc_length=result.doc_length,
            title=title or path,
            metadata=metadata,
        )

        return doc_id

    def index_file(
        self,
        file_path: str | Path,
        relative_to: str | Path | None = None,
        metadata: dict | None = None,
    ) -> int | None:
        """
        Index a single file.

        Args:
            file_path: Path to the file to index.
            relative_to: Base path for display purposes.
            metadata: Optional metadata dict.

        Returns:
            Document ID, or None if file was skipped.
        """
        file_path = Path(file_path)

        if not file_path.exists() or not file_path.is_file():
            return None

        # Check extension
        if file_path.suffix.lower() not in self.extensions:
            return None

        # Check file size
        try:
            file_size = file_path.stat().st_size
        except OSError:
            return None

        if file_size > self.max_file_size:
            return None

        if file_size == 0:
            return None

        # Read file content
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            return None

        # Determine display path
        if relative_to:
            display_path = str(Path(file_path).relative_to(Path(relative_to)))
        else:
            display_path = str(file_path)

        return self.index_text(
            text=content,
            path=display_path,
            title=file_path.name,
            metadata=metadata,
        )

    def index_directory(
        self,
        dir_path: str | Path,
        recursive: bool = True,
        extensions: set[str] | None = None,
    ) -> int:
        """
        Index all matching files in a directory.

        Args:
            dir_path: Path to the directory.
            recursive: Whether to recurse into subdirectories.
            extensions: Override file extensions to index.

        Returns:
            Number of files indexed.
        """
        dir_path = Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            return 0

        exts = extensions or self.extensions
        count = 0

        for root, dirs, files in os.walk(dir_path):
            # Skip hidden and known non-essential directories
            dirs[:] = [d for d in dirs if d not in SKIP_DIRECTORIES and not d.startswith(".")]

            for filename in sorted(files):
                file_path = Path(root) / filename

                if file_path.suffix.lower() not in exts:
                    continue

                try:
                    file_size = file_path.stat().st_size
                except OSError:
                    continue

                if file_size > self.max_file_size or file_size == 0:
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except (OSError, UnicodeDecodeError):
                    continue

                display_path = str(file_path.relative_to(dir_path))

                self.index_text(
                    text=content,
                    path=display_path,
                    title=file_path.name,
                )
                count += 1

            if not recursive:
                break

        return count

    def search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """
        Search the index with a query string.

        Supports boolean queries, phrase queries, and fuzzy matching.

        Args:
            query: Search query string.
            max_results: Maximum number of results.
            min_score: Minimum score threshold.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        if not query.strip():
            return []

        parser = QueryParser(query)
        try:
            query_ast = parser.parse()
        except Exception:
            # Fallback: treat entire query as a simple term
            terms = [normalize_query_term(t, self.tokenizer._stemmer) for t in query.split()]
            return self._search_simple(terms, max_results, min_score)

        return self._execute_query(query_ast, max_results, min_score)

    def _execute_query(
        self,
        node: QueryNode,
        max_results: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Execute a query AST node and return results."""
        if node.type == QueryNodeType.TERM:
            term = normalize_query_term(node.term, self.tokenizer._stemmer)
            return self._search_simple([term], max_results, min_score)

        elif node.type == QueryNodeType.PHRASE:
            terms = [normalize_query_term(t, self.tokenizer._stemmer) for t in node.phrase]
            return self._search_phrase(terms, max_results, min_score)

        elif node.type == QueryNodeType.PREFIX:
            prefix = normalize_query_term(node.prefix, self.tokenizer._stemmer)
            return self._search_prefix(prefix, max_results, min_score)

        elif node.type == QueryNodeType.FUZZY:
            term = normalize_query_term(node.fuzzy_term, self.tokenizer._stemmer)
            return self._search_fuzzy(term, node.max_distance, max_results, min_score)

        elif node.type == QueryNodeType.AND:
            if len(node.children) < 2:
                if node.children:
                    return self._execute_query(node.children[0], max_results, min_score)
                return []
            left = self._execute_query(node.children[0], max_results * 2, 0)
            right = self._execute_query(node.children[1], max_results * 2, 0)
            return self._intersect_results(left, right, max_results)

        elif node.type == QueryNodeType.OR:
            if len(node.children) < 2:
                if node.children:
                    return self._execute_query(node.children[0], max_results, min_score)
                return []
            left = self._execute_query(node.children[0], max_results, min_score)
            right = self._execute_query(node.children[1], max_results, min_score)
            return self._union_results(left, right, max_results, min_score)

        elif node.type == QueryNodeType.NOT:
            if not node.children:
                return []
            # NOT is tricky - we return all docs NOT matching the query
            # For simplicity, we just return empty (NOT is mainly for future use)
            return []

        return []

    def _scored_to_result(self, scored: ScoredDocument) -> SearchResult:
        """Convert a ScoredDocument to a SearchResult."""
        doc_info = self.index.get_document(scored.doc_id)
        snippet = None
        if doc_info:
            # Try to get a snippet from the document
            try:
                path = Path(doc_info.path)
                if path.exists():
                    content = path.read_text(encoding="utf-8", errors="replace")
                    lines = content.splitlines()
                    if lines:
                        snippet = lines[0][:200]  # First line as snippet
            except (OSError, UnicodeDecodeError):
                pass
        return SearchResult(
            path=scored.path,
            score=scored.score,
            title=scored.title,
            snippet=snippet,
            metadata=scored.metadata or {},
        )

    def _search_simple(
        self,
        terms: list[str],
        max_results: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Search for documents matching simple terms."""
        scored = self.scorer.score_documents(
            self.index, terms, max_results=max_results, min_score=min_score
        )
        return [self._scored_to_result(s) for s in scored]

    def _search_phrase(
        self,
        terms: list[str],
        max_results: int,
        min_score: float,
    ) -> list[SearchResult]:
        """
        Search for documents containing an exact phrase.

        First finds documents containing all terms, then verifies
        that the terms appear consecutively in order.
        """
        if not terms:
            return []

        # Find documents containing all terms
        term_sets: list[set[int]] = []
        for term in terms:
            term_info = self.index.get_term_info(term)
            if term_info is None:
                return []  # Term not found, no results
            doc_ids = {p.doc_id for p in term_info.postings}
            term_sets.append(doc_ids)

        # Intersect to find docs with all terms
        candidate_ids = term_sets[0]
        for ts in term_sets[1:]:
            candidate_ids = candidate_ids & ts

        if not candidate_ids:
            return []

        # Verify phrase order using positions
        matching_docs: list[tuple[int, float]] = []

        for doc_id in candidate_ids:
            # Get positions for each term in this document
            all_positions: list[list[int]] = []
            for term in terms:
                term_info = self.index.get_term_info(term)
                if term_info is None:
                    break
                for posting in term_info.postings:
                    if posting.doc_id == doc_id:
                        all_positions.append(sorted(posting.positions))
                        break

            if len(all_positions) != len(terms):
                continue

            # Check if positions form a consecutive sequence
            if self._is_consecutive_phrase(all_positions):
                # Score based on BM25 for the first term
                score = self.scorer.score_documents(self.index, [terms[0]], max_results=1)
                for s in score:
                    if s.doc_id == doc_id:
                        matching_docs.append((doc_id, s.score))
                        break
                else:
                    matching_docs.append((doc_id, 0.0))

        # Sort by score
        matching_docs.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for doc_id, score in matching_docs[:max_results]:
            if score < min_score:
                continue
            doc_info = self.index.get_document(doc_id)
            if doc_info:
                snippet = self._get_phrase_snippet(doc_info.path, terms)
                results.append(
                    SearchResult(
                        path=doc_info.path,
                        score=round(score, 4),
                        title=doc_info.title,
                        snippet=snippet,
                    )
                )

        return results

    def _is_consecutive_phrase(self, positions: list[list[int]]) -> bool:
        """Check if positions form a consecutive sequence."""
        if not positions or not positions[0]:
            return False

        # Sort positions within each term
        sorted_pos = [sorted(p) for p in positions]

        # Check if any combination of positions forms consecutive sequence
        def check_combinations(idx: int, prev_pos: int) -> bool:
            if idx == len(sorted_pos):
                return True
            for pos in sorted_pos[idx]:
                if pos == prev_pos + 1 and check_combinations(idx + 1, pos):
                    return True
            return False

        return any(check_combinations(1, start_pos) for start_pos in sorted_pos[0])

    def _get_phrase_snippet(self, path: str, terms: list[str]) -> str | None:
        """Get a snippet showing the phrase match."""
        # For now, return the terms as the snippet
        return " ".join(terms)

    def _search_prefix(
        self,
        prefix: str,
        max_results: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Search for documents with terms matching a prefix."""
        all_terms = self.index.get_all_terms()
        matching_terms = [t for t in all_terms if t.startswith(prefix)]

        if not matching_terms:
            return []

        return self._search_simple(matching_terms, max_results, min_score)

    def _search_fuzzy(
        self,
        term: str,
        max_distance: int,
        max_results: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Search with fuzzy/approximate matching."""
        all_terms = self.index.get_all_terms()
        fuzzy_matches = find_closest_matches(term, all_terms, max_distance, max_results=20)

        if not fuzzy_matches:
            return []

        matching_terms = [match[0] for match in fuzzy_matches]
        return self._search_simple(matching_terms, max_results, min_score)

    def _intersect_results(
        self,
        left: list[SearchResult],
        right: list[SearchResult],
        max_results: int,
    ) -> list[SearchResult]:
        """Intersect two result sets (AND operation)."""
        right_paths = {r.path: r for r in right}
        result: list[SearchResult] = []

        for lr in left:
            if lr.path in right_paths:
                rr = right_paths[lr.path]
                # Combine scores (sum for AND)
                combined = SearchResult(
                    path=lr.path,
                    score=lr.score + rr.score,
                    title=lr.title or rr.title,
                    snippet=lr.snippet or rr.snippet,
                    line_number=lr.line_number or rr.line_number,
                )
                result.append(combined)

        result.sort(key=lambda x: x.score, reverse=True)
        return result[:max_results]

    def _union_results(
        self,
        left: list[SearchResult],
        right: list[SearchResult],
        max_results: int,
        min_score: float,
    ) -> list[SearchResult]:
        """Union two result sets (OR operation)."""
        seen: dict[str, SearchResult] = {}

        for r in left + right:
            if r.path in seen:
                # Combine scores (max for OR)
                existing = seen[r.path]
                if r.score > existing.score:
                    seen[r.path] = r
            else:
                seen[r.path] = r

        result = list(seen.values())
        result.sort(key=lambda x: x.score, reverse=True)
        return [r for r in result if r.score >= min_score][:max_results]

    def index_info(self) -> dict:
        """Get information about the current index."""
        return self.index.stats()

    def clear(self) -> None:
        """Clear the index."""
        self.index.clear()

    def __enter__(self) -> SearchEngine:
        return self

    def __exit__(self, *args) -> None:
        self.save()
