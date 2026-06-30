"""Tests for the SQLite storage module."""

import tempfile
from pathlib import Path

import pytest

from minisearch.index import InvertedIndex
from minisearch.storage import SearchStorage


class TestSearchStorage:
    """Tests for SQLite-backed storage."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test.db"

    def teardown_method(self):
        import shutil

        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        # Create an index with some data
        index = InvertedIndex()
        index.add_document("doc1.txt", ["hello", "world"], doc_length=2)
        index.add_document("doc2.txt", ["hello", "foo"], doc_length=2)

        # Save
        storage = SearchStorage(self.db_path)
        storage.save(index)
        storage.close()

        # Load
        storage2 = SearchStorage(self.db_path)
        loaded = storage2.load()
        storage2.close()

        assert loaded.num_documents == 2
        assert loaded.num_terms == 3  # hello, world, foo
        assert loaded.total_tokens == 4

    def test_exists(self):
        storage = SearchStorage(self.db_path)
        assert storage.exists() is False

        # Create the database
        index = InvertedIndex()
        storage.save(index)
        storage.close()

        assert storage.exists() is True

    def test_load_nonexistent(self):
        storage = SearchStorage(self.db_path)
        with pytest.raises(FileNotFoundError):
            storage.load()

    def test_overwrite_index(self):
        # Save initial index
        index1 = InvertedIndex()
        index1.add_document("doc1.txt", ["hello"], doc_length=1)

        storage = SearchStorage(self.db_path)
        storage.save(index1)
        storage.close()

        # Save new index (should overwrite)
        index2 = InvertedIndex()
        index2.add_document("doc2.txt", ["world"], doc_length=1)

        storage2 = SearchStorage(self.db_path)
        storage2.save(index2)
        loaded = storage2.load()
        storage2.close()

        assert loaded.num_documents == 1
        postings = loaded.get_postings("world")
        assert len(postings) == 1

    def test_save_empty_index(self):
        index = InvertedIndex()

        storage = SearchStorage(self.db_path)
        storage.save(index)
        storage.close()

        storage2 = SearchStorage(self.db_path)
        loaded = storage2.load()
        storage2.close()

        assert loaded.num_documents == 0

    def test_context_manager(self):
        index = InvertedIndex()
        index.add_document("doc1.txt", ["hello"], doc_length=1)

        with SearchStorage(self.db_path) as storage:
            storage.save(index)

        with SearchStorage(self.db_path) as storage:
            loaded = storage.load()
            assert loaded.num_documents == 1

    def test_large_index(self):
        index = InvertedIndex()
        for i in range(100):
            # Use unique terms per document to ensure all are stored
            terms = [f"term{i}_{j}" for j in range(10)]
            index.add_document(
                f"doc{i}.txt",
                terms,
                doc_length=10,
            )

        storage = SearchStorage(self.db_path)
        storage.save(index)
        loaded = storage.load()
        storage.close()

        assert loaded.num_documents == 100
        assert loaded.num_terms == 1000  # 100 docs * 10 unique terms each

    def test_metadata_preserved(self):
        index = InvertedIndex()
        index.add_document(
            "doc1.txt",
            ["hello"],
            doc_length=1,
            title="Hello Doc",
            metadata={"author": "test", "version": 42},
        )

        storage = SearchStorage(self.db_path)
        storage.save(index)
        loaded = storage.load()
        storage.close()

        doc = loaded.get_document(0)
        assert doc is not None
        assert doc.title == "Hello Doc"
        assert doc.metadata["author"] == "test"
        assert doc.metadata["version"] == 42

    def test_positions_preserved(self):
        index = InvertedIndex()
        index.add_document(
            "doc1.txt",
            ["hello", "world", "hello"],
            doc_length=3,
            positions=[[0], [1], [2]],
        )

        storage = SearchStorage(self.db_path)
        storage.save(index)
        loaded = storage.load()
        storage.close()

        hello_info = loaded.get_term_info("hello")
        assert hello_info is not None
        assert hello_info.postings[0].positions == [0, 2]

    def test_close_idempotent(self):
        storage = SearchStorage(self.db_path)
        storage.close()
        storage.close()  # Should not raise

    def test_wal_mode(self):
        """Verify WAL journal mode is enabled for performance."""
        index = InvertedIndex()
        storage = SearchStorage(self.db_path)
        storage.save(index)

        import sqlite3

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        conn.close()
        storage.close()

        assert mode.lower() == "wal"
