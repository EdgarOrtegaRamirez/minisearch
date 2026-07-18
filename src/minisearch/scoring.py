"""
BM25 scoring algorithm for relevance ranking.

BM25 (Best Matching 25) is a probabilistic information retrieval function
that ranks documents based on term frequency and inverse document frequency.
It's the standard ranking function used in modern search engines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from minisearch.index import InvertedIndex


@dataclass
class ScoredDocument:
    """A document with its relevance score."""

    doc_id: int
    score: float
    path: str
    title: str | None = None
    metadata: dict | None = None


class BM25Scorer:
    """
    BM25 scoring implementation.

    BM25 formula for a single term:
        score(D, Q) = IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl))

    Where:
        - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
        - f(qi, D) = term frequency of qi in document D
        - |D| = document length in tokens
        - avgdl = average document length
        - N = total number of documents
        - n(qi) = number of documents containing term qi
        - k1 = term frequency saturation parameter (default: 1.5)
        - b = document length normalization parameter (default: 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer.

        Args:
            k1: Term frequency saturation. Higher values make term frequency
                matter more. Typical range: 1.2-2.0.
            b: Document length normalization. 0 = no normalization,
                1 = full normalization. Typical value: 0.75.
        """
        self.k1 = k1
        self.b = b

    def compute_idf(self, doc_freq: int, total_docs: int) -> float:
        """
        Compute Inverse Document Frequency for a term.

        IDF measures how rare/common a term is across the corpus.
        Rare terms get higher IDF scores.

        Args:
            doc_freq: Number of documents containing the term.
            total_docs: Total number of documents in the index.

        Returns:
            IDF score.
        """
        if doc_freq == 0:
            return 0.0
        # BM25 IDF formula with +1 floor to avoid negative values
        return math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def score_term(
        self, term_freq: int, doc_freq: int, doc_length: int, avg_doc_length: float, total_docs: int
    ) -> float:
        """
        Compute BM25 score for a single term in a document.

        Args:
            term_freq: Number of times the term appears in the document.
            doc_freq: Number of documents containing the term.
            doc_length: Length of the document in tokens.
            avg_doc_length: Average document length across all documents.
            total_docs: Total number of documents.

        Returns:
            BM25 score for this term.
        """
        idf = self.compute_idf(doc_freq, total_docs)
        if idf <= 0:
            return 0.0

        # BM25 term frequency normalization
        tf_norm = (term_freq * (self.k1 + 1)) / (
            term_freq + self.k1 * (1 - self.b + self.b * doc_length / max(avg_doc_length, 1.0))
        )

        return idf * tf_norm

    def score_documents(
        self,
        index: InvertedIndex,
        query_terms: list[str],
        max_results: int = 10,
        min_score: float = 0.0,
    ) -> list[ScoredDocument]:
        """
        Score documents for a set of query terms using BM25.

        For multi-term queries, scores are summed across all query terms.
        Documents that match more terms get higher scores.

        Args:
            index: The inverted index to search.
            query_terms: List of query terms (after stemming/normalization).
            max_results: Maximum number of results to return.
            min_score: Minimum score threshold.

        Returns:
            List of ScoredDocument objects, sorted by score descending.
        """
        total_docs = index.num_documents
        if total_docs == 0:
            return []

        avg_doc_length = index.avg_doc_length

        # Accumulate scores per document
        doc_scores: dict[int, float] = {}

        for term in query_terms:
            term_info = index.get_term_info(term)
            if term_info is None:
                continue

            doc_freq = term_info.doc_freq
            for posting in term_info.postings:
                doc_info = index.get_document(posting.doc_id)
                doc_length = doc_info.length if doc_info else 0
                score_val = self.score_term(
                    term_freq=posting.term_freq,
                    doc_freq=doc_freq,
                    doc_length=doc_length,
                    avg_doc_length=avg_doc_length,
                    total_docs=total_docs,
                )
                doc_scores[posting.doc_id] = doc_scores.get(posting.doc_id, 0.0) + score_val

        # Sort by score and return top results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results: list[ScoredDocument] = []
        for doc_id, score in sorted_docs[:max_results]:
            if score < min_score:
                continue
            doc_info = index.get_document(doc_id)
            if doc_info is None:
                continue
            results.append(
                ScoredDocument(
                    doc_id=doc_id,
                    score=round(score, 4),
                    path=doc_info.path,
                    title=doc_info.title,
                    metadata=doc_info.metadata,
                )
            )

        return results

    def explain(self, term: str, doc_id: int, index: InvertedIndex) -> dict:
        """
        Explain the BM25 score for a specific term in a specific document.

        Useful for debugging and understanding why documents are ranked
        the way they are.

        Args:
            term: The query term.
            doc_id: The document ID.
            index: The inverted index.

        Returns:
            Dictionary with score breakdown.
        """
        term_info = index.get_term_info(term)
        doc_info = index.get_document(doc_id)

        if not term_info or not doc_info:
            return {"error": "Term or document not found"}

        posting = None
        for p in term_info.postings:
            if p.doc_id == doc_id:
                posting = p
                break

        if posting is None:
            return {"error": "Term not found in document"}

        idf = self.compute_idf(term_info.doc_freq, index.num_documents)
        tf_norm = (posting.term_freq * (self.k1 + 1)) / (
            posting.term_freq
            + self.k1 * (1 - self.b + self.b * doc_info.length / max(index.avg_doc_length, 1.0))
        )
        score = idf * tf_norm

        return {
            "term": term,
            "doc_id": doc_id,
            "doc_path": doc_info.path,
            "term_freq": posting.term_freq,
            "doc_freq": term_info.doc_freq,
            "total_docs": index.num_documents,
            "doc_length": doc_info.length,
            "avg_doc_length": round(index.avg_doc_length, 2),
            "idf": round(idf, 4),
            "tf_normalized": round(tf_norm, 4),
            "score": round(score, 4),
            "k1": self.k1,
            "b": self.b,
        }
