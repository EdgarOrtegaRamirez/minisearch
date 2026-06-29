"""Tests for the CLI module."""

import tempfile
from pathlib import Path

import pytest

from minisearch.cli import main


class TestCLI:
    """Tests for the CLI commands."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = str(Path(self.tmpdir) / "test.db")

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_test_files(self):
        """Create test files for indexing."""
        test_dir = Path(self.tmpdir) / "docs"
        test_dir.mkdir()
        (test_dir / "doc1.txt").write_text("hello world this is a test")
        (test_dir / "doc2.txt").write_text("hello foo bar baz")
        (test_dir / "doc3.py").write_text("def hello(): return 'world'")
        return test_dir

    def test_help(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_no_command(self):
        result = main([])
        assert result == 0

    def test_index_command(self):
        test_dir = self._create_test_files()
        result = main(["index", str(test_dir), "-d", self.db_path])
        assert result == 0

    def test_search_command(self):
        test_dir = self._create_test_files()
        main(["index", str(test_dir), "-d", self.db_path])

        result = main(["search", "hello", "-d", self.db_path])
        assert result == 0

    def test_search_json_output(self):
        test_dir = self._create_test_files()
        main(["index", str(test_dir), "-d", self.db_path])

        result = main(["search", "hello", "-d", self.db_path, "--json"])
        assert result == 0

    def test_info_command(self):
        test_dir = self._create_test_files()
        main(["index", str(test_dir), "-d", self.db_path])

        result = main(["info", "-d", self.db_path])
        assert result == 0

    def test_clear_command_with_yes(self):
        test_dir = self._create_test_files()
        main(["index", str(test_dir), "-d", self.db_path])

        result = main(["clear", "-d", self.db_path, "-y"])
        assert result == 0

    def test_export_command(self):
        test_dir = self._create_test_files()
        main(["index", str(test_dir), "-d", self.db_path])

        output_path = str(Path(self.tmpdir) / "export.json")
        result = main(["export", "-d", self.db_path, "-o", output_path])
        assert result == 0
        assert Path(output_path).exists()

    def test_index_with_no_stem(self):
        test_dir = self._create_test_files()
        result = main(["index", str(test_dir), "-d", self.db_path, "--no-stem"])
        assert result == 0

    def test_index_with_custom_extensions(self):
        test_dir = self._create_test_files()
        result = main([
            "index", str(test_dir), "-d", self.db_path,
            "--extensions", ".py",
        ])
        assert result == 0

    def test_search_with_max_results(self):
        test_dir = self._create_test_files()
        main(["index", str(test_dir), "-d", self.db_path])

        result = main(["search", "hello", "-d", self.db_path, "-n", "1"])
        assert result == 0

    def test_index_single_file(self):
        test_file = Path(self.tmpdir) / "single.txt"
        test_file.write_text("hello world test")

        result = main(["index", str(test_file), "-d", self.db_path])
        assert result == 0

    def test_search_nonexistent_index(self):
        result = main(["search", "hello", "-d", "/nonexistent/path/db.db"])
        # Should handle gracefully
        assert result in (0, 1)

    def test_info_empty_index(self):
        # Create empty index
        from minisearch.search import SearchEngine
        engine = SearchEngine(index_path=self.db_path)
        engine.save()

        result = main(["info", "-d", self.db_path])
        assert result == 0
