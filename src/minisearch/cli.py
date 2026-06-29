"""
CLI interface for MiniSearch.

Commands:
    index   - Index files or directories
    search  - Search the index
    info    - Show index statistics
    clear   - Clear the index
    export  - Export index as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from minisearch.search import SearchEngine


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="minisearch",
        description="MiniSearch: Lightweight full-text search engine with BM25 ranking",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index files or directories")
    index_parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to index",
    )
    index_parser.add_argument(
        "-d", "--db",
        default=".minisearch.db",
        help="Path to the index database (default: .minisearch.db)",
    )
    index_parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        default=True,
        help="Recurse into subdirectories (default: True)",
    )
    index_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't recurse into subdirectories",
    )
    index_parser.add_argument(
        "--no-stem",
        action="store_true",
        help="Disable Porter stemming",
    )
    index_parser.add_argument(
        "--no-stop-words",
        action="store_true",
        help="Don't remove stop words",
    )
    index_parser.add_argument(
        "--extensions",
        nargs="+",
        help="File extensions to index (e.g., .py .js .txt)",
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument(
        "query",
        nargs="+",
        help="Search query",
    )
    search_parser.add_argument(
        "-d", "--db",
        default=".minisearch.db",
        help="Path to the index database (default: .minisearch.db)",
    )
    search_parser.add_argument(
        "-n", "--max-results",
        type=int,
        default=10,
        help="Maximum number of results (default: 10)",
    )
    search_parser.add_argument(
        "-s", "--min-score",
        type=float,
        default=0.0,
        help="Minimum score threshold (default: 0.0)",
    )
    search_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    search_parser.add_argument(
        "--context",
        type=int,
        default=0,
        help="Number of context lines to show (default: 0)",
    )
    search_parser.add_argument(
        "--no-stem",
        action="store_true",
        help="Disable stemming for this search",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show index statistics")
    info_parser.add_argument(
        "-d", "--db",
        default=".minisearch.db",
        help="Path to the index database (default: .minisearch.db)",
    )

    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the index")
    clear_parser.add_argument(
        "-d", "--db",
        default=".minisearch.db",
        help="Path to the index database (default: .minisearch.db)",
    )
    clear_parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export index as JSON")
    export_parser.add_argument(
        "-d", "--db",
        default=".minisearch.db",
        help="Path to the index database (default: .minisearch.db)",
    )
    export_parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)",
    )

    return parser


def cmd_index(args: argparse.Namespace) -> int:
    """Handle the index command."""
    from minisearch.tokenizer import Tokenizer

    tokenizer = Tokenizer(
        remove_stop_words=not args.no_stop_words,
        stem=not args.no_stem,
    )

    extensions = None
    if args.extensions:
        extensions = set(args.extensions)

    engine = SearchEngine(
        index_path=args.db,
        tokenizer=tokenizer,
        extensions=extensions,
    )

    total_files = 0
    start_time = time.time()

    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file():
            result = engine.index_file(path)
            if result is not None:
                total_files += 1
                print(f"  Indexed: {path}")
        elif path.is_dir():
            count = engine.index_directory(path, recursive=not args.no_recursive)
            total_files += count
            print(f"  Indexed {count} files from: {path}")
        else:
            print(f"  Warning: {path} not found, skipping", file=sys.stderr)

    elapsed = time.time() - start_time
    engine.save()

    print(f"\nIndexed {total_files} files in {elapsed:.2f}s")
    print(f"Index: {args.db}")
    stats = engine.index_info()
    print(f"  Documents: {stats['num_documents']}")
    print(f"  Unique terms: {stats['num_terms']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Avg doc length: {stats['avg_doc_length']}")

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Handle the search command."""
    from minisearch.tokenizer import Tokenizer

    tokenizer = Tokenizer(stem=not args.no_stem)

    engine = SearchEngine(
        index_path=args.db,
        tokenizer=tokenizer,
    )

    query = " ".join(args.query)
    start_time = time.time()
    results = engine.search(
        query,
        max_results=args.max_results,
        min_score=args.min_score,
    )
    elapsed = time.time() - start_time

    if args.json:
        output = [
            {
                "path": r.path,
                "score": r.score,
                "title": r.title,
                "snippet": r.snippet,
                "line_number": r.line_number,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))
    else:
        if not results:
            print("No results found.")
            return 0

        print(f"Found {len(results)} results ({elapsed*1000:.1f}ms):\n")

        for i, result in enumerate(results, 1):
            location = result.path
            if result.line_number is not None:
                location += f":{result.line_number}"

            print(f"  {i}. [{result.score:.3f}] {location}")
            if result.title and result.title != result.path:
                print(f"     Title: {result.title}")
            if result.snippet:
                print(f"     {result.snippet}")

            # Show context lines if requested
            if args.context > 0 and result.line_number is not None:
                _show_context(result.path, result.line_number, args.context)

            print()

    return 0


def _show_context(path: str, line_number: int, context_lines: int) -> None:
    """Show context lines around a match."""
    try:
        file_path = Path(path)
        if not file_path.exists():
            return
        lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, line_number - 1 - context_lines)
        end = min(len(lines), line_number + context_lines)
        for i in range(start, end):
            marker = ">>>" if i == line_number - 1 else "   "
            print(f"     {marker} {i+1:4d} | {lines[i]}")
    except (OSError, UnicodeDecodeError):
        pass


def cmd_info(args: argparse.Namespace) -> int:
    """Handle the info command."""
    engine = SearchEngine(index_path=args.db)

    stats = engine.index_info()

    if stats["num_documents"] == 0:
        print("Index is empty.")
        return 0

    print(f"Index: {args.db}")
    print(f"  Documents: {stats['num_documents']}")
    print(f"  Unique terms: {stats['num_terms']}")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Avg doc length: {stats['avg_doc_length']} tokens")
    print()

    # Show top terms
    terms = engine.index.get_all_terms()
    term_freqs = []
    for term in terms:
        ti = engine.index.get_term_info(term)
        if ti:
            term_freqs.append((term, ti.doc_freq, ti.total_freq))

    term_freqs.sort(key=lambda x: x[2], reverse=True)

    print("Top 20 terms by frequency:")
    print(f"  {'Term':<20} {'Doc Freq':>10} {'Total Freq':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12}")
    for term, doc_freq, total_freq in term_freqs[:20]:
        print(f"  {term:<20} {doc_freq:>10} {total_freq:>12}")

    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    """Handle the clear command."""
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Index not found: {args.db}")
        return 1

    if not args.yes:
        response = input(f"Are you sure you want to clear the index '{args.db}'? [y/N] ")
        if response.lower() not in ("y", "yes"):
            print("Cancelled.")
            return 0

    engine = SearchEngine(index_path=args.db)
    engine.clear()
    engine.save()
    print(f"Index cleared: {args.db}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Handle the export command."""
    engine = SearchEngine(index_path=args.db)
    json_str = engine.index.export_json()

    if args.output:
        Path(args.output).write_text(json_str, encoding="utf-8")
        print(f"Index exported to: {args.output}")
    else:
        print(json_str)

    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "index": cmd_index,
        "search": cmd_search,
        "info": cmd_info,
        "clear": cmd_clear,
        "export": cmd_export,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
