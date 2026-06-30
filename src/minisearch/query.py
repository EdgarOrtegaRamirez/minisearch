"""
Query parser for boolean, phrase, and fuzzy queries.

Supports:
- Simple term queries: "hello world"
- Boolean operators: "hello AND world", "hello OR world", "NOT hello"
- Parentheses: "(hello OR world) AND test"
- Phrase queries: '"exact phrase match"'
- Fuzzy queries: "hello~2" (Levenshtein distance tolerance of 2)
- Prefix queries: "hel*" (matches terms starting with "hel")
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class QueryNodeType(enum.Enum):
    """Types of query nodes in the AST."""

    TERM = "term"
    PHRASE = "phrase"
    PREFIX = "prefix"
    FUZZY = "fuzzy"
    AND = "and"
    OR = "or"
    NOT = "not"


@dataclass
class QueryNode:
    """
    A node in the query AST.

    Represents different query operations:
    - TERM: a single search term
    - PHRASE: an exact phrase query
    - PREFIX: a prefix wildcard query
    - FUZZY: a fuzzy/approximate match query
    - AND/OR/NOT: boolean operators with children
    """

    type: QueryNodeType
    term: str | None = None  # for TERM, PREFIX
    phrase: list[str] | None = None  # for PHRASE (list of stemmed tokens)
    prefix: str | None = None  # for PREFIX
    fuzzy_term: str | None = None  # for FUZZY
    max_distance: int = 0  # for FUZZY (max Levenshtein distance)
    children: list[QueryNode] = field(default_factory=list)  # for AND, OR, NOT


class QueryParseError(Exception):
    """Raised when a query cannot be parsed."""

    pass


class QueryParser:
    """
    Recursive descent parser for search queries.

    Grammar:
        query     → or_expr
        or_expr   → and_expr (OR and_expr)*
        and_expr  → unary (AND? unary)*
        unary     → NOT unary | primary
        primary   → PHRASE | FUZZY | PREFIX | TERM | '(' or_expr ')'
        PHRASE    → '"' word+ '"'
        FUZZY     → word '~' DIGITS
        PREFIX    → word '*'
        TERM      → word
        word      → [a-zA-Z0-9_-]+

    Examples:
        "hello world"           → AND(hello, world)
        "hello OR world"        → OR(hello, world)
        "hello AND NOT world"   → AND(hello, NOT(world))
        '"exact phrase"'        → PHRASE(exact, phrase)
        "hello~2"               → FUZZY(hello, max_distance=2)
        "hel*"                  → PREFIX(hel)
        "(a OR b) AND c"        → AND(OR(a, b), c)
    """

    def __init__(self, query: str):
        self.query = query
        self.pos = 0
        self.length = len(query)

    def parse(self) -> QueryNode:
        """
        Parse the query string into a QueryNode AST.

        Returns:
            Root QueryNode of the parsed query.

        Raises:
            QueryParseError: If the query syntax is invalid.
        """
        self.pos = 0
        self._skip_whitespace()

        # Check for leading binary operators
        if self._is_keyword_at(self.pos):
            kw = self.query[self.pos : self.pos + 3].upper()
            if kw in ("AND", "OR "):
                raise QueryParseError(f"Unexpected keyword '{kw}' at start of query")

        node = self._parse_or()
        self._skip_whitespace()
        if self.pos < self.length:
            raise QueryParseError(
                f"Unexpected character '{self.query[self.pos]}' at position {self.pos}"
            )
        return node

    def _skip_whitespace(self) -> None:
        """Skip whitespace characters."""
        while self.pos < self.length and self.query[self.pos].isspace():
            self.pos += 1

    def _peek(self) -> str | None:
        """Peek at the current character without consuming it."""
        self._skip_whitespace()
        if self.pos < self.length:
            return self.query[self.pos]
        return None

    def _consume(self, ch: str) -> None:
        """Consume a specific character."""
        self._skip_whitespace()
        if self.pos >= self.length or self.query[self.pos] != ch:
            expected = f"'{ch}'"
            got = f"'{self.query[self.pos]}'" if self.pos < self.length else "end of input"
            raise QueryParseError(f"Expected {expected} but got {got} at position {self.pos}")
        self.pos += 1

    def _parse_or(self) -> QueryNode:
        """Parse OR expressions (lowest precedence)."""
        left = self._parse_and()

        while True:
            self._skip_whitespace()
            if self._match_keyword("OR"):
                right = self._parse_and()
                left = QueryNode(type=QueryNodeType.OR, children=[left, right])
            else:
                break

        return left

    def _parse_and(self) -> QueryNode:
        """Parse AND expressions (medium precedence)."""
        left = self._parse_unary()

        while True:
            self._skip_whitespace()
            # Check for explicit AND keyword
            if self._match_keyword("AND"):
                right = self._parse_unary()
                left = QueryNode(type=QueryNodeType.AND, children=[left, right])
            else:
                # Implicit AND: two consecutive terms
                self._skip_whitespace()
                if self.pos < self.length and self.query[self.pos] not in (
                    "(",
                    '"',
                    ")",
                ):
                    # Check if next token is a keyword
                    if not self._is_keyword_at(self.pos):
                        right = self._parse_unary()
                        left = QueryNode(type=QueryNodeType.AND, children=[left, right])
                    else:
                        break
                else:
                    break

        return left

    def _parse_unary(self) -> QueryNode:
        """Parse unary operators (NOT)."""
        self._skip_whitespace()
        if self._match_keyword("NOT"):
            operand = self._parse_unary()
            return QueryNode(type=QueryNodeType.NOT, children=[operand])
        return self._parse_primary()

    def _parse_primary(self) -> QueryNode:
        """Parse primary expressions: terms, phrases, parenthesized expressions."""
        self._skip_whitespace()

        if self.pos >= self.length:
            raise QueryParseError("Unexpected end of query")

        ch = self.query[self.pos]

        # Parenthesized expression
        if ch == "(":
            self.pos += 1
            node = self._parse_or()
            self._consume(")")
            return node

        # Phrase query
        if ch == '"':
            return self._parse_phrase()

        # Term (possibly fuzzy or prefix)
        return self._parse_term()

    def _parse_phrase(self) -> QueryNode:
        """Parse a phrase query: "exact phrase match"."""
        self._consume('"')
        words: list[str] = []

        while self.pos < self.length:
            self._skip_whitespace()
            if self.pos >= self.length or self.query[self.pos] == '"':
                break
            word = self._read_word()
            if word:
                words.append(word)

        if self.pos < self.length and self.query[self.pos] == '"':
            self.pos += 1
        else:
            raise QueryParseError("Unterminated phrase query (missing closing quote)")

        if not words:
            raise QueryParseError("Empty phrase query")

        return QueryNode(type=QueryNodeType.PHRASE, phrase=words)

    def _parse_term(self) -> QueryNode:
        """Parse a term, possibly with fuzzy (~) or prefix (*) modifiers."""
        word = self._read_word()
        if not word:
            raise QueryParseError(f"Expected term at position {self.pos}")

        self._skip_whitespace()

        # Check for fuzzy modifier: term~2
        if self.pos < self.length and self.query[self.pos] == "~":
            self.pos += 1
            dist_str = self._read_digits()
            max_distance = int(dist_str) if dist_str else 1
            return QueryNode(
                type=QueryNodeType.FUZZY,
                fuzzy_term=word,
                max_distance=max_distance,
            )

        # Check for prefix modifier: term*
        if self.pos < self.length and self.query[self.pos] == "*":
            self.pos += 1
            return QueryNode(type=QueryNodeType.PREFIX, prefix=word)

        return QueryNode(type=QueryNodeType.TERM, term=word)

    def _read_word(self) -> str:
        """Read a word token."""
        start = self.pos
        while self.pos < self.length and (
            self.query[self.pos].isalnum() or self.query[self.pos] in "_-"
        ):
            self.pos += 1
        return self.query[start : self.pos]

    def _read_digits(self) -> str:
        """Read a sequence of digits."""
        start = self.pos
        while self.pos < self.length and self.query[self.pos].isdigit():
            self.pos += 1
        return self.query[start : self.pos]

    def _match_keyword(self, keyword: str) -> bool:
        """
        Check if the current position matches a keyword.

        Keywords must be followed by a space, parenthesis, or end of input
        to avoid matching prefixes of terms.
        """
        self._skip_whitespace()
        remaining = self.query[self.pos :]
        if remaining.upper().startswith(keyword):
            after = self.pos + len(keyword)
            if after >= self.length or not remaining[len(keyword)].isalnum():
                self.pos += len(keyword)
                return True
        return False

    def _is_keyword_at(self, pos: int) -> bool:
        """Check if a keyword starts at the given position."""
        remaining = self.query[pos:].upper()
        for kw in ("AND", "OR", "NOT"):
            if remaining.startswith(kw):
                after = pos + len(kw)
                if after >= self.length or not self.query[after].isalnum():
                    return True
        return False


def normalize_query_term(term: str, stemmer=None) -> str:
    """
    Normalize a query term for matching against the index.

    Applies the same normalization as the tokenizer:
    1. Lowercase
    2. Stem (if stemmer provided)

    Args:
        term: Raw query term.
        stemmer: Optional PorterStemmer instance.

    Returns:
        Normalized term.
    """
    normalized = term.lower()
    if stemmer is not None:
        normalized = stemmer.stem(normalized)
    return normalized
