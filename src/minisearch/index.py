"""
Inverted index data structure for efficient text search.

The inverted index maps terms to document IDs with positional information,
term frequencies, and document-level statistics. Supports incremental updates
and serialization to/from SQLite.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class PostingEntry:
    """A single posting in the inverted index (one per document per term)."""

    doc_id: int
    term_freq: int  # number of times term appears in this document
    positions: list[int] = field(default_factory=list)  # character positions in document

    def __post_init__(self) -> None:
        if self.positions is None:
            self.positions = []


@dataclass
class TermInfo:
    """Information about a term across all documents."""

    doc_freq: int = 0  # number of documents containing this term
    postings: list[PostingEntry] = field(default_factory=list)

    @property
    def total_freq(self) -> int:
        """Total occurrences across all documents."""
        return sum(p.term_freq for p in self.postings)


@dataclass
class DocumentInfo:
    """Metadata about an indexed document."""

    doc_id: int
    path: str  # file path or document identifier
    length: int  # number of tokens in the document
    title: str | None = None  # optional display title
    metadata: dict = field(default_factory=dict)  # arbitrary metadata


class InvertedIndex:
    """
    In-memory inverted index with positional information.

    Maps terms to posting lists containing document IDs, term frequencies,
    and token positions. Supports incremental document addition and
    serialization to/from SQLite for persistence.

    Usage:
        index = InvertedIndex()
        index.add_document("doc1.txt", tokens, doc_length=42)
        results = index.search("hello world")
    """

    def __init__(self) -> None:
        # term -> TermInfo
        self._terms: dict[str, TermInfo] = {}
        # doc_id -> DocumentInfo
        self._documents: dict[int, DocumentInfo] = {}
        # path -> doc_id for deduplication
        self._path_to_id: dict[str, int] = {}
        self._next_doc_id: int = 0
        self._total_docs: int = 0
        self._total_tokens: int = 0

    @property
    def num_documents(self) -> int:
        """Number of indexed documents."""
        return self._total_docs

    @property
    def num_terms(self) -> int:
        """Number of unique terms in the index."""
        return len(self._terms)

    @property
    def total_tokens(self) -> int:
        """Total number of tokens across all documents."""
        return self._total_tokens

    @property
    def avg_doc_length(self) -> float:
        """Average document length in tokens."""
        if self._total_docs == 0:
            return 0.0
        return self._total_tokens / self._total_docs

    def get_document(self, doc_id: int) -> DocumentInfo | None:
        """Get document info by ID."""
        return self._documents.get(doc_id)

    def get_term_info(self, term: str) -> TermInfo | None:
        """Get term information."""
        return self._terms.get(term)

    def get_doc_freq(self, term: str) -> int:
        """Get document frequency for a term."""
        info = self._terms.get(term)
        return info.doc_freq if info else 0

    def get_postings(self, term: str) -> list[PostingEntry]:
        """Get posting list for a term."""
        info = self._terms.get(term)
        return info.postings if info else []

    def get_all_terms(self) -> list[str]:
        """Get all terms in the index, sorted."""
        return sorted(self._terms.keys())

    def add_document(
        self,
        path: str,
        tokens: list[str],
        doc_length: int,
        positions: list[list[int]] | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """
        Add a document to the index.

        Args:
            path: Unique document path/identifier.
            tokens: List of normalized token strings (after stemming).
            doc_length: Number of tokens in the document.
            positions: Optional list of positions for each token.
            title: Optional display title.
            metadata: Optional metadata dict.

        Returns:
            Document ID assigned to this document.
        """
        # Check if document already exists (update)
        if path in self._path_to_id:
            doc_id = self._path_to_id[path]
            self._remove_document(doc_id)
        else:
            doc_id = self._next_doc_id
            self._next_doc_id += 1

        # Store document info
        self._documents[doc_id] = DocumentInfo(
            doc_id=doc_id,
            path=path,
            length=doc_length,
            title=title,
            metadata=metadata or {},
        )
        self._path_to_id[path] = doc_id
        self._total_docs += 1
        self._total_tokens += doc_length

        # Build term frequencies and add to index
        term_positions: dict[str, list[int]] = defaultdict(list)
        term_freqs: dict[str, int] = defaultdict(int)

        for i, token in enumerate(tokens):
            term_freqs[token] += 1
            if positions and i < len(positions):
                term_positions[token].extend(positions[i])
            else:
                term_positions[token].append(i)

        for term, freq in term_freqs.items():
            if term not in self._terms:
                self._terms[term] = TermInfo()

            posting = PostingEntry(
                doc_id=doc_id,
                term_freq=freq,
                positions=term_positions[term],
            )
            self._terms[term].postings.append(posting)
            self._terms[term].doc_freq += 1

        return doc_id

    def _remove_document(self, doc_id: int) -> None:
        """Remove a document from the index."""
        if doc_id not in self._documents:
            return

        doc_info = self._documents[doc_id]
        self._total_docs -= 1
        self._total_tokens -= doc_info.length

        # Remove from path mapping
        self._path_to_id.pop(doc_info.path, None)

        # Remove postings
        terms_to_remove = []
        for term, term_info in self._terms.items():
            term_info.postings = [p for p in term_info.postings if p.doc_id != doc_id]
            term_info.doc_freq = len(term_info.postings)
            if term_info.doc_freq == 0:
                terms_to_remove.append(term)

        for term in terms_to_remove:
            del self._terms[term]

        del self._documents[doc_id]

    def remove_document_by_path(self, path: str) -> bool:
        """Remove a document by its path. Returns True if found and removed."""
        doc_id = self._path_to_id.get(path)
        if doc_id is not None:
            self._remove_document(doc_id)
            return True
        return False

    def clear(self) -> None:
        """Clear all data from the index."""
        self._terms.clear()
        self._documents.clear()
        self._path_to_id.clear()
        self._next_doc_id = 0
        self._total_docs = 0
        self._total_tokens = 0

    def to_dict(self) -> dict:
        """Serialize the index to a dictionary."""
        return {
            "documents": {
                str(did): {
                    "path": info.path,
                    "length": info.length,
                    "title": info.title,
                    "metadata": info.metadata,
                }
                for did, info in self._documents.items()
            },
            "terms": {
                term: {
                    "doc_freq": ti.doc_freq,
                    "postings": [
                        {
                            "doc_id": p.doc_id,
                            "term_freq": p.term_freq,
                            "positions": p.positions,
                        }
                        for p in ti.postings
                    ],
                }
                for term, ti in self._terms.items()
            },
            "next_doc_id": self._next_doc_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> InvertedIndex:
        """Deserialize an index from a dictionary."""
        index = cls()
        index._next_doc_id = data.get("next_doc_id", 0)

        # Restore documents
        for did_str, doc_data in data.get("documents", {}).items():
            did = int(did_str)
            info = DocumentInfo(
                doc_id=did,
                path=doc_data["path"],
                length=doc_data["length"],
                title=doc_data.get("title"),
                metadata=doc_data.get("metadata", {}),
            )
            index._documents[did] = info
            index._path_to_id[info.path] = did
            index._total_docs += 1
            index._total_tokens += info.length

        # Restore terms
        for term, term_data in data.get("terms", {}).items():
            ti = TermInfo(doc_freq=term_data["doc_freq"])
            for p_data in term_data.get("postings", []):
                ti.postings.append(PostingEntry(
                    doc_id=p_data["doc_id"],
                    term_freq=p_data["term_freq"],
                    positions=p_data.get("positions", []),
                ))
            index._terms[term] = ti

        return index

    def export_json(self) -> str:
        """Export index as JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def import_json(cls, json_str: str) -> InvertedIndex:
        """Import index from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def stats(self) -> dict:
        """Get index statistics."""
        return {
            "num_documents": self.num_documents,
            "num_terms": self.num_terms,
            "total_tokens": self.total_tokens,
            "avg_doc_length": round(self.avg_doc_length, 2),
        }
