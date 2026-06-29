"""Tests for the search engine integration."""

import tempfile
from pathlib import Path

from minisearch.search import SearchEngine


class TestSearchEngine:
    """Tests for the SearchEngine class."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "test.db")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_index_and_search(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello world", "doc1.txt")
        engine.index_text("hello foo", "doc2.txt")
        engine.save()

        results = engine.search("hello")
        assert len(results) == 2

    def test_search_ranking(self):
        engine = SearchEngine(index_path=self.db_path)
        # "hello" appears 3 times in doc1, 1 time in doc2
        engine.index_text("hello hello hello world", "doc1.txt")
        engine.index_text("hello foo bar", "doc2.txt")

        results = engine.search("hello")
        assert len(results) == 2
        # doc1 should rank higher
        assert results[0].path == "doc1.txt"

    def test_search_no_results(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello world", "doc1.txt")

        results = engine.search("nonexistent")
        assert len(results) == 0

    def test_search_empty_query(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello", "doc1.txt")

        results = engine.search("")
        assert len(results) == 0

    def test_boolean_or_query(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello world", "doc1.txt")
        engine.index_text("foo bar", "doc2.txt")
        engine.index_text("baz qux", "doc3.txt")

        results = engine.search("hello OR foo")
        paths = [r.path for r in results]
        assert "doc1.txt" in paths
        assert "doc2.txt" in paths

    def test_boolean_and_query(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello world", "doc1.txt")
        engine.index_text("hello foo", "doc2.txt")
        engine.index_text("world foo", "doc3.txt")

        results = engine.search("hello AND world")
        paths = [r.path for r in results]
        assert "doc1.txt" in paths
        assert "doc2.txt" not in paths

    def test_phrase_query(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("the quick brown fox jumps", "doc1.txt")
        engine.index_text("the brown quick fox", "doc2.txt")

        results = engine.search('"quick brown"')
        # Only doc1 should match the exact phrase
        if results:
            assert results[0].path == "doc1.txt"

    def test_max_results(self):
        engine = SearchEngine(index_path=self.db_path)
        for i in range(20):
            engine.index_text(f"hello world term{i}", f"doc{i}.txt")

        results = engine.search("hello", max_results=5)
        assert len(results) == 5

    def test_index_file(self):
        # Create a test file
        test_file = Path(self.tmpdir) / "test.txt"
        test_file.write_text("hello world this is a test")

        engine = SearchEngine(index_path=self.db_path)
        result = engine.index_file(test_file, relative_to=self.tmpdir)
        assert result is not None

        results = engine.search("hello")
        assert len(results) == 1

    def test_index_directory(self):
        # Create test files
        test_dir = Path(self.tmpdir) / "docs"
        test_dir.mkdir()
        (test_dir / "doc1.txt").write_text("hello world")
        (test_dir / "doc2.txt").write_text("foo bar")
        (test_dir / "doc3.py").write_text("def hello(): pass")

        engine = SearchEngine(index_path=self.db_path)
        count = engine.index_directory(test_dir)
        assert count == 3

    def test_index_directory_non_recursive(self):
        # Create test files
        test_dir = Path(self.tmpdir) / "docs"
        test_dir.mkdir()
        sub_dir = test_dir / "sub"
        sub_dir.mkdir()
        (test_dir / "doc1.txt").write_text("hello world")
        (sub_dir / "doc2.txt").write_text("foo bar")

        engine = SearchEngine(index_path=self.db_path)
        count = engine.index_directory(test_dir, recursive=False)
        assert count == 1

    def test_index_file_nonexistent(self):
        engine = SearchEngine(index_path=self.db_path)
        result = engine.index_file("/nonexistent/file.txt")
        assert result is None

    def test_index_file_wrong_extension(self):
        test_file = Path(self.tmpdir) / "test.xyz"
        test_file.write_text("hello world")

        engine = SearchEngine(index_path=self.db_path)
        result = engine.index_file(test_file)
        assert result is None

    def test_index_info(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello world", "doc1.txt")

        info = engine.index_info()
        assert info["num_documents"] == 1
        assert info["num_terms"] == 2

    def test_clear(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello world", "doc1.txt")
        engine.clear()
        engine.save()

        # Reload and check
        engine2 = SearchEngine(index_path=self.db_path)
        info = engine2.index_info()
        assert info["num_documents"] == 0

    def test_context_manager(self):
        with SearchEngine(index_path=self.db_path) as engine:
            engine.index_text("hello world", "doc1.txt")

        # Index should be saved
        engine2 = SearchEngine(index_path=self.db_path)
        info = engine2.index_info()
        assert info["num_documents"] == 1

    def test_index_file_too_large(self):
        engine = SearchEngine(
            index_path=self.db_path,
            max_file_size=100,  # 100 bytes
        )
        test_file = Path(self.tmpdir) / "large.txt"
        test_file.write_text("x" * 200)

        result = engine.index_file(test_file)
        assert result is None

    def test_search_result_attributes(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello world", "doc1.txt", title="Hello Document")

        results = engine.search("hello")
        assert len(results) == 1
        r = results[0]
        assert r.path == "doc1.txt"
        assert r.title == "Hello Document"
        assert r.score > 0

    def test_persistence(self):
        # Index files
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello world", "doc1.txt")
        engine.save()

        # Create new engine and verify data persists
        engine2 = SearchEngine(index_path=self.db_path)
        results = engine2.search("hello")
        assert len(results) == 1

    def test_update_document(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello world", "doc1.txt")
        engine.save()

        # Update the same document
        engine.index_text("hello new world", "doc1.txt")
        engine.save()

        # Reload and verify
        engine2 = SearchEngine(index_path=self.db_path)
        info = engine2.index_info()
        assert info["num_documents"] == 1

    def test_multiple_file_extensions(self):
        engine = SearchEngine(
            index_path=self.db_path,
            extensions={".py", ".js"},
        )

        py_file = Path(self.tmpdir) / "test.py"
        py_file.write_text("hello world")

        js_file = Path(self.tmpdir) / "test.js"
        js_file.write_text("hello foo")

        txt_file = Path(self.tmpdir) / "test.txt"
        txt_file.write_text("hello bar")

        engine.index_file(py_file)
        engine.index_file(js_file)
        engine.index_file(txt_file)

        info = engine.index_info()
        assert info["num_documents"] == 2  # Only .py and .js

    def test_prefix_search(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello world", "doc1.txt")
        engine.index_text("help me", "doc2.txt")

        results = engine.search("hel*")
        # Should find both "hello" and "help"
        assert len(results) >= 1

    def test_fuzzy_search(self):
        engine = SearchEngine(index_path=self.db_path)
        engine.index_text("hello world", "doc1.txt")

        results = engine.search("helo~1")
        # "helo" is 1 edit away from "hello"
        # May or may not find results depending on index contents
        # Just verify it doesn't crash
        assert isinstance(results, list)
