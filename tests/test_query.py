"""Tests for the query parser module."""

import pytest

from minisearch.query import QueryNodeType, QueryParseError, QueryParser, normalize_query_term


class TestQueryParser:
    """Tests for the QueryParser class."""

    def test_simple_term(self):
        parser = QueryParser("hello")
        node = parser.parse()
        assert node.type == QueryNodeType.TERM
        assert node.term == "hello"

    def test_multiple_terms_implicit_and(self):
        parser = QueryParser("hello world")
        node = parser.parse()
        assert node.type == QueryNodeType.AND
        assert len(node.children) == 2
        assert node.children[0].term == "hello"
        assert node.children[1].term == "world"

    def test_explicit_and(self):
        parser = QueryParser("hello AND world")
        node = parser.parse()
        assert node.type == QueryNodeType.AND

    def test_or_operator(self):
        parser = QueryParser("hello OR world")
        node = parser.parse()
        assert node.type == QueryNodeType.OR
        assert len(node.children) == 2

    def test_not_operator(self):
        parser = QueryParser("NOT hello")
        node = parser.parse()
        assert node.type == QueryNodeType.NOT
        assert node.children[0].term == "hello"

    def test_complex_boolean(self):
        parser = QueryParser("hello AND NOT world")
        node = parser.parse()
        assert node.type == QueryNodeType.AND
        assert node.children[0].term == "hello"
        assert node.children[1].type == QueryNodeType.NOT

    def test_parentheses(self):
        parser = QueryParser("(hello OR world) AND test")
        node = parser.parse()
        assert node.type == QueryNodeType.AND
        assert node.children[0].type == QueryNodeType.OR
        assert node.children[1].term == "test"

    def test_nested_parentheses(self):
        parser = QueryParser("((a OR b) AND c) OR d")
        node = parser.parse()
        assert node.type == QueryNodeType.OR

    def test_phrase_query(self):
        parser = QueryParser('"hello world"')
        node = parser.parse()
        assert node.type == QueryNodeType.PHRASE
        assert node.phrase == ["hello", "world"]

    def test_phrase_single_word(self):
        parser = QueryParser('"hello"')
        node = parser.parse()
        assert node.type == QueryNodeType.PHRASE
        assert node.phrase == ["hello"]

    def test_unterminated_phrase(self):
        parser = QueryParser('"hello world')
        with pytest.raises(QueryParseError):
            parser.parse()

    def test_empty_phrase(self):
        parser = QueryParser('""')
        with pytest.raises(QueryParseError):
            parser.parse()

    def test_fuzzy_query(self):
        parser = QueryParser("hello~2")
        node = parser.parse()
        assert node.type == QueryNodeType.FUZZY
        assert node.fuzzy_term == "hello"
        assert node.max_distance == 2

    def test_fuzzy_default_distance(self):
        parser = QueryParser("hello~")
        node = parser.parse()
        assert node.type == QueryNodeType.FUZZY
        assert node.max_distance == 1

    def test_prefix_query(self):
        parser = QueryParser("hel*")
        node = parser.parse()
        assert node.type == QueryNodeType.PREFIX
        assert node.prefix == "hel"

    def test_combined_phrase_and_boolean(self):
        parser = QueryParser('"hello world" OR test')
        node = parser.parse()
        assert node.type == QueryNodeType.OR
        assert node.children[0].type == QueryNodeType.PHRASE
        assert node.children[1].term == "test"

    def test_operator_precedence(self):
        # AND has higher precedence than OR
        parser = QueryParser("a OR b AND c")
        node = parser.parse()
        assert node.type == QueryNodeType.OR
        assert node.children[0].term == "a"
        assert node.children[1].type == QueryNodeType.AND

    def test_complex_query(self):
        parser = QueryParser('(hello OR "world test") AND NOT fuzzy~2')
        node = parser.parse()
        assert node.type == QueryNodeType.AND
        assert node.children[0].type == QueryNodeType.OR
        assert node.children[1].type == QueryNodeType.NOT

    def test_invalid_syntax(self):
        parser = QueryParser("AND hello")
        with pytest.raises(QueryParseError):
            parser.parse()

    def test_empty_parentheses(self):
        parser = QueryParser("()")
        with pytest.raises(QueryParseError):
            parser.parse()

    def test_hyphenated_terms(self):
        parser = QueryParser("machine-learning")
        node = parser.parse()
        assert node.type == QueryNodeType.TERM
        assert node.term == "machine-learning"

    def test_underscored_terms(self):
        parser = QueryParser("my_function")
        node = parser.parse()
        assert node.type == QueryNodeType.TERM
        assert node.term == "my_function"


class TestNormalizeQueryTerm:
    """Tests for query term normalization."""

    def test_lowercase(self):
        assert normalize_query_term("Hello") == "hello"

    def test_with_stemmer(self):
        from minisearch.tokenizer import PorterStemmer
        stemmer = PorterStemmer()
        assert normalize_query_term("Running", stemmer) == "run"

    def test_without_stemmer(self):
        assert normalize_query_term("Running") == "running"
