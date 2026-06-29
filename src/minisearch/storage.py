"""
SQLite-backed persistent storage for the search index.

Provides efficient serialization and deserialization of the inverted index,
document metadata, and index configuration to/from SQLite databases.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from minisearch.index import (
    DocumentInfo,
    InvertedIndex,
    PostingEntry,
    TermInfo,
)


class SearchStorage:
    """
    SQLite-backed storage for the search index.

    Provides methods to save and load the complete index state
    to/from a SQLite database file. Uses efficient batch inserts
    and transactions for performance.

    Schema:
        - documents: doc_id, path, length, title, metadata_json
        - terms: term, doc_freq, postings_json
        - config: key, value (for storing index metadata)
    """

    def __init__(self, db_path: str | Path) -> None:
        """
        Initialize storage with a database path.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._create_tables()
        return self._conn

    def _create_tables(self) -> None:
        """Create the database schema."""
        conn = self._conn
        assert conn is not None

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                doc_id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                length INTEGER NOT NULL,
                title TEXT,
                metadata_json TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS terms (
                term TEXT PRIMARY KEY,
                doc_freq INTEGER NOT NULL,
                postings_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
        """)
        conn.commit()

    def save(self, index: InvertedIndex) -> None:
        """
        Save the complete index to SQLite.

        Uses transactions and batch operations for efficiency.

        Args:
            index: The inverted index to save.
        """
        conn = self._get_conn()
        with conn:
            # Clear existing data
            conn.execute("DELETE FROM terms")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM config")

            # Save config
            conn.execute(
                "INSERT INTO config (key, value) VALUES (?, ?)",
                ("next_doc_id", str(index._next_doc_id)),
            )
            conn.execute(
                "INSERT INTO config (key, value) VALUES (?, ?)",
                ("num_documents", str(index.num_documents)),
            )
            conn.execute(
                "INSERT INTO config (key, value) VALUES (?, ?)",
                ("total_tokens", str(index.total_tokens)),
            )

            # Save documents in batch
            doc_data = [
                (
                    info.doc_id,
                    info.path,
                    info.length,
                    info.title,
                    json.dumps(info.metadata),
                )
                for info in index._documents.values()
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO documents (doc_id, path, length, title, metadata_json) "
                "VALUES (?, ?, ?, ?, ?)",
                doc_data,
            )

            # Save terms in batch
            term_data = [
                (
                    term,
                    ti.doc_freq,
                    json.dumps([
                        {
                            "doc_id": p.doc_id,
                            "term_freq": p.term_freq,
                            "positions": p.positions,
                        }
                        for p in ti.postings
                    ]),
                )
                for term, ti in index._terms.items()
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO terms (term, doc_freq, postings_json) "
                "VALUES (?, ?, ?)",
                term_data,
            )

    def load(self) -> InvertedIndex:
        """
        Load the index from SQLite.

        Returns:
            The loaded InvertedIndex.

        Raises:
            FileNotFoundError: If the database file doesn't exist.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"Index database not found: {self.db_path}")

        conn = self._get_conn()
        index = InvertedIndex()

        # Load config
        cursor = conn.execute("SELECT key, value FROM config")
        config = {row[0]: row[1] for row in cursor}
        index._next_doc_id = int(config.get("next_doc_id", "0"))

        # Load documents
        cursor = conn.execute("SELECT doc_id, path, length, title, metadata_json FROM documents")
        for doc_id, path, length, title, metadata_json in cursor:
            info = DocumentInfo(
                doc_id=doc_id,
                path=path,
                length=length,
                title=title,
                metadata=json.loads(metadata_json) if metadata_json else {},
            )
            index._documents[doc_id] = info
            index._path_to_id[path] = doc_id
            index._total_docs += 1
            index._total_tokens += length

        # Load terms
        cursor = conn.execute("SELECT term, doc_freq, postings_json FROM terms")
        for term, doc_freq, postings_json in cursor:
            postings_data = json.loads(postings_json)
            ti = TermInfo(doc_freq=doc_freq)
            for p_data in postings_data:
                ti.postings.append(PostingEntry(
                    doc_id=p_data["doc_id"],
                    term_freq=p_data["term_freq"],
                    positions=p_data.get("positions", []),
                ))
            index._terms[term] = ti

        return index

    def exists(self) -> bool:
        """Check if the database file exists."""
        return self.db_path.exists()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SearchStorage:
        return self

    def __exit__(self, *args) -> None:
        self.close()
