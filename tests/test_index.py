"""Tests for the inverted index module."""

from minisearch.index import (
    InvertedIndex,
)


class TestInvertedIndex:
    """Tests for the InvertedIndex class."""

    def setup_method(self):
        self.index = InvertedIndex()

    def test_empty_index(self):
        assert self.index.num_documents == 0
        assert self.index.num_terms == 0
        assert self.index.total_tokens == 0
        assert self.index.avg_doc_length == 0.0

    def test_add_single_document(self):
        doc_id = self.index.add_document(
            path="test.txt",
            tokens=["hello", "world"],
            doc_length=2,
        )
        assert doc_id == 0
        assert self.index.num_documents == 1
        assert self.index.total_tokens == 2
        assert self.index.avg_doc_length == 2.0

    def test_add_multiple_documents(self):
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)
        self.index.add_document("doc2.txt", ["hello", "foo"], doc_length=2)
        self.index.add_document("doc3.txt", ["bar", "baz"], doc_length=2)

        assert self.index.num_documents == 3
        assert self.index.total_tokens == 6
        assert self.index.avg_doc_length == 2.0

    def test_term_frequency(self):
        self.index.add_document("doc1.txt", ["hello", "hello", "world"], doc_length=3)

        hello_info = self.index.get_term_info("hello")
        assert hello_info is not None
        assert hello_info.doc_freq == 1
        assert hello_info.total_freq == 2

        world_info = self.index.get_term_info("world")
        assert world_info is not None
        assert world_info.doc_freq == 1
        assert world_info.total_freq == 1

    def test_document_frequency_across_docs(self):
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)
        self.index.add_document("doc2.txt", ["hello", "foo"], doc_length=2)

        hello_info = self.index.get_term_info("hello")
        assert hello_info is not None
        assert hello_info.doc_freq == 2
        assert hello_info.total_freq == 2

    def test_get_document(self):
        self.index.add_document("test.txt", ["hello"], doc_length=1, title="Test")

        doc = self.index.get_document(0)
        assert doc is not None
        assert doc.path == "test.txt"
        assert doc.title == "Test"

    def test_get_nonexistent_document(self):
        assert self.index.get_document(999) is None

    def test_get_postings(self):
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)
        self.index.add_document("doc2.txt", ["hello", "foo"], doc_length=2)

        postings = self.index.get_postings("hello")
        assert len(postings) == 2
        doc_ids = {p.doc_id for p in postings}
        assert 0 in doc_ids
        assert 1 in doc_ids

    def test_get_all_terms(self):
        self.index.add_document("doc1.txt", ["alpha", "beta", "gamma"], doc_length=3)
        self.index.add_document("doc2.txt", ["beta", "delta"], doc_length=2)

        terms = self.index.get_all_terms()
        assert terms == ["alpha", "beta", "delta", "gamma"]

    def test_update_existing_document(self):
        self.index.add_document("test.txt", ["hello", "world"], doc_length=2)
        assert self.index.num_documents == 1

        # Update the same document
        self.index.add_document("test.txt", ["hello", "new", "world"], doc_length=3)
        assert self.index.num_documents == 1
        assert self.index.total_tokens == 3

    def test_remove_document(self):
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)
        self.index.add_document("doc2.txt", ["hello", "foo"], doc_length=2)

        removed = self.index.remove_document_by_path("doc1.txt")
        assert removed is True
        assert self.index.num_documents == 1

        # "hello" should still exist (in doc2)
        hello_info = self.index.get_term_info("hello")
        assert hello_info is not None
        assert hello_info.doc_freq == 1

    def test_remove_nonexistent_document(self):
        removed = self.index.remove_document_by_path("nonexistent.txt")
        assert removed is False

    def test_clear(self):
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)
        self.index.add_document("doc2.txt", ["foo", "bar"], doc_length=2)

        self.index.clear()
        assert self.index.num_documents == 0
        assert self.index.num_terms == 0
        assert self.index.total_tokens == 0

    def test_serialization_roundtrip(self):
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)
        self.index.add_document("doc2.txt", ["hello", "foo"], doc_length=2)

        # Export to dict
        data = self.index.to_dict()

        # Import from dict
        restored = InvertedIndex.from_dict(data)

        assert restored.num_documents == 2
        assert restored.num_terms == 3  # hello, world, foo
        assert restored.total_tokens == 4

        # Check that searches still work
        postings = restored.get_postings("hello")
        assert len(postings) == 2

    def test_json_roundtrip(self):
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)

        json_str = self.index.export_json()
        restored = InvertedIndex.import_json(json_str)

        assert restored.num_documents == 1
        assert restored.num_terms == 2

    def test_stats(self):
        self.index.add_document("doc1.txt", ["hello", "world"], doc_length=2)
        stats = self.index.stats()
        assert stats["num_documents"] == 1
        assert stats["num_terms"] == 2
        assert stats["total_tokens"] == 2
        assert stats["avg_doc_length"] == 2.0

    def test_positions_tracking(self):
        self.index.add_document(
            "doc1.txt",
            ["hello", "world", "hello"],
            doc_length=3,
            positions=[[0], [1], [2]],
        )

        hello_info = self.index.get_term_info("hello")
        assert hello_info is not None
        assert hello_info.postings[0].positions == [0, 2]

    def test_metadata(self):
        self.index.add_document(
            "doc1.txt",
            ["hello"],
            doc_length=1,
            metadata={"author": "test", "version": 1},
        )

        doc = self.index.get_document(0)
        assert doc is not None
        assert doc.metadata["author"] == "test"
        assert doc.metadata["version"] == 1
