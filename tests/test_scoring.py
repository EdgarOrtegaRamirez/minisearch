"""Tests for the BM25 scoring module."""

import math

from minisearch.index import InvertedIndex
from minisearch.scoring import BM25Scorer, ScoredDocument


class TestBM25Scorer:
    """Tests for BM25 scoring."""

    def setup_method(self):
        self.scorer = BM25Scorer()
        self.index = InvertedIndex()

    def test_idf_basic(self):
        # IDF should be higher for rare terms
        idf_1doc = self.scorer.compute_idf(doc_freq=1, total_docs=100)
        idf_50doc = self.scorer.compute_idf(doc_freq=50, total_docs=100)

        assert idf_1doc > idf_50doc

    def test_idf_zero_doc_freq(self):
        idf = self.scorer.compute_idf(doc_freq=0, total_docs=100)
        assert idf == 0.0

    def test_idf_formula(self):
        # Verify the IDF formula
        doc_freq = 10
        total_docs = 100
        expected = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        actual = self.scorer.compute_idf(doc_freq, total_docs)
        assert abs(actual - expected) < 1e-10

    def test_score_term_basic(self):
        score = self.scorer.score_term(
            term_freq=1,
            doc_freq=10,
            doc_length=100,
            avg_doc_length=100,
            total_docs=1000,
        )
        assert score > 0

    def test_score_term_higher_tf(self):
        score_low = self.scorer.score_term(
            term_freq=1, doc_freq=10, doc_length=100,
            avg_doc_length=100, total_docs=1000,
        )
        score_high = self.scorer.score_term(
            term_freq=5, doc_freq=10, doc_length=100,
            avg_doc_length=100, total_docs=1000,
        )
        assert score_high > score_low

    def test_score_term_tf_saturation(self):
        # BM25 has diminishing returns for term frequency
        score_1 = self.scorer.score_term(
            term_freq=1, doc_freq=10, doc_length=100,
            avg_doc_length=100, total_docs=1000,
        )
        score_10 = self.scorer.score_term(
            term_freq=10, doc_freq=10, doc_length=100,
            avg_doc_length=100, total_docs=1000,
        )
        score_100 = self.scorer.score_term(
            term_freq=100, doc_freq=10, doc_length=100,
            avg_doc_length=100, total_docs=1000,
        )
        # Higher TF should still give higher score, but with diminishing returns
        assert score_10 > score_1
        assert score_100 > score_10

    def test_score_documents(self):
        # Add documents with different content
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)
        self.index.add_document("doc2.txt", ["hello", "foo"], doc_length=2)
        self.index.add_document("doc3.txt", ["bar", "baz"], doc_length=2)

        results = self.scorer.score_documents(
            self.index, ["hello"], max_results=10
        )

        assert len(results) == 2  # Only doc1 and doc2 contain "hello"
        # Both should have positive scores
        for r in results:
            assert r.score > 0

    def test_score_documents_ranking(self):
        # "hello" appears more in doc1
        self.index.add_document("doc1.txt", ["hello", "hello", "hello"], doc_length=3)
        self.index.add_document("doc2.txt", ["hello", "foo"], doc_length=2)

        results = self.scorer.score_documents(
            self.index, ["hello"], max_results=10
        )

        assert len(results) == 2
        # doc1 should rank higher (more occurrences of "hello")
        assert results[0].doc_id == 0  # doc1

    def test_score_documents_max_results(self):
        for i in range(20):
            self.index.add_document(
                f"doc{i}.txt", ["hello", f"word{i}"], doc_length=2
            )

        results = self.scorer.score_documents(
            self.index, ["hello"], max_results=5
        )

        assert len(results) == 5

    def test_score_documents_empty_index(self):
        results = self.scorer.score_documents(
            self.index, ["hello"], max_results=10
        )
        assert len(results) == 0

    def test_score_documents_no_matching_terms(self):
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)
        results = self.scorer.score_documents(
            self.index, ["nonexistent"], max_results=10
        )
        assert len(results) == 0

    def test_score_documents_multi_term_query(self):
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)
        self.index.add_document("doc2.txt", ["hello", "foo"], doc_length=2)
        self.index.add_document("doc3.txt", ["world", "foo"], doc_length=2)

        results = self.scorer.score_documents(
            self.index, ["hello", "world"], max_results=10
        )

        # doc1 should rank highest (contains both terms)
        assert len(results) >= 1
        assert results[0].doc_id == 0  # doc1

    def test_explain(self):
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)

        explanation = self.scorer.explain("hello", 0, self.index)

        assert "term" in explanation
        assert explanation["term"] == "hello"
        assert "idf" in explanation
        assert "tf_normalized" in explanation
        assert "score" in explanation

    def test_explain_missing_term(self):
        self.index.add_document("doc1.txt", ["hello"], doc_length=1)
        explanation = self.scorer.explain("nonexistent", 0, self.index)
        assert "error" in explanation

    def test_explain_missing_doc(self):
        self.index.add_document("doc1.txt", ["hello"], doc_length=1)
        explanation = self.scorer.explain("hello", 999, self.index)
        assert "error" in explanation

    def test_custom_k1_and_b(self):
        # Different k1 and b values should produce different scores
        scorer1 = BM25Scorer(k1=1.0, b=0.5)
        scorer2 = BM25Scorer(k1=2.0, b=0.9)

        score1 = scorer1.score_term(3, 10, 100, 100, 1000)
        score2 = scorer2.score_term(3, 10, 100, 100, 1000)

        # Scores should be different
        assert score1 != score2


class TestScoredDocument:
    """Tests for ScoredDocument."""

    def test_creation(self):
        doc = ScoredDocument(
            doc_id=1,
            score=0.5,
            path="test.txt",
            title="Test",
        )
        assert doc.doc_id == 1
        assert doc.score == 0.5
        assert doc.path == "test.txt"
