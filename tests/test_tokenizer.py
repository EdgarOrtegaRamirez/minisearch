"""Tests for the tokenizer module."""

import pytest

from minisearch.tokenizer import PorterStemmer, Tokenizer, TokenizeResult


class TestPorterStemmer:
    """Tests for the Porter stemming algorithm."""

    def setup_method(self):
        self.stemmer = PorterStemmer()

    @pytest.mark.parametrize("word,expected", [
        ("running", "run"),
        ("flies", "fli"),
        ("tries", "tri"),
        ("caress", "caress"),
        ("cats", "cat"),
        (" ponies", "poni"),
        ("studies", "studi"),
        ("hopping", "hop"),
        ("sing", "sing"),
        ("troubled", "troubl"),
        ("amazing", "amaz"),
        ("connection", "connect"),
        ("relational", "rel"),
        ("sensitivity", "sensit"),
        ("organization", "organ"),
        ("computer", "comput"),
        ("walking", "walk"),
        ("jumping", "jump"),
        ("beautiful", "beauti"),
        ("quickly", "quick"),
    ])
    def test_stemming(self, word, expected):
        result = self.stemmer.stem(word)
        assert result == expected

    def test_short_words_unchanged(self):
        assert self.stemmer.stem("a") == "a"
        assert self.stemmer.stem("is") == "is"

    def test_already_stemmed(self):
        # Words that are already short enough
        assert self.stemmer.stem("cat") == "cat"
        assert self.stemmer.stem("run") == "run"


class TestTokenizer:
    """Tests for the Tokenizer class."""

    def setup_method(self):
        self.tokenizer = Tokenizer()

    def test_basic_tokenization(self):
        result = self.tokenizer.tokenize("hello world")
        assert len(result.tokens) == 2
        assert result.tokens[0].text == "hello"
        assert result.tokens[1].text == "world"

    def test_lowercasing(self):
        result = self.tokenizer.tokenize("Hello World")
        assert result.tokens[0].text.lower() == "hello"
        assert result.tokens[1].text.lower() == "world"

    def test_stop_word_removal(self):
        result = self.tokenizer.tokenize("the quick brown fox")
        # "the" should be removed as a stop word
        texts = [t.text.lower() for t in result.tokens]
        assert "the" not in texts
        assert "quick" in texts
        assert "brown" in texts
        assert "fox" in texts

    def test_no_stop_word_removal(self):
        tokenizer = Tokenizer(remove_stop_words=False)
        result = tokenizer.tokenize("the quick brown fox")
        texts = [t.text.lower() for t in result.tokens]
        assert "the" in texts

    def test_stemming(self):
        result = self.tokenizer.tokenize("running dogs jump")
        keys = [t.key for t in result.tokens]
        assert "run" in keys
        assert "dog" in keys
        assert "jump" in keys

    def test_no_stemming(self):
        tokenizer = Tokenizer(stem=False)
        result = tokenizer.tokenize("running dogs jump")
        keys = [t.key for t in result.tokens]
        assert "running" in keys
        assert "dogs" in keys
        assert "jump" in keys

    def test_camel_case_splitting(self):
        result = self.tokenizer.tokenize("camelCase snake_case")
        texts = [t.text.lower() for t in result.tokens]
        assert "camel" in texts
        assert "case" in texts
        assert "snake" in texts
        assert "case" in texts

    def test_no_camel_case_splitting(self):
        tokenizer = Tokenizer(split_camel_case=False)
        result = tokenizer.tokenize("camelCase")
        texts = [t.text for t in result.tokens]
        assert "camelCase" in texts

    def test_min_token_length(self):
        tokenizer = Tokenizer(min_token_length=3)
        result = tokenizer.tokenize("a big the is cat")
        # Only "big" and "cat" should survive (3+ chars, non-stop)
        texts = [t.text.lower() for t in result.tokens]
        for t in texts:
            assert len(t) >= 3

    def test_position_tracking(self):
        result = self.tokenizer.tokenize("hello beautiful world")
        positions = [t.position for t in result.tokens]
        # Positions should be increasing
        for i in range(1, len(positions)):
            assert positions[i] >= positions[i - 1]

    def test_empty_text(self):
        result = self.tokenizer.tokenize("")
        assert len(result.tokens) == 0
        assert result.doc_length == 0

    def test_only_stop_words(self):
        result = self.tokenizer.tokenize("the is a an")
        assert len(result.tokens) == 0

    def test_numbers(self):
        result = self.tokenizer.tokenize("version 2 is number 42")
        texts = [t.text for t in result.tokens]
        assert "2" in texts
        assert "42" in texts

    def test_mixed_content(self):
        text = "The quick-Brown_Fox jumps over 42 lazy dogs."
        result = self.tokenizer.tokenize(text)
        # Should tokenize into meaningful tokens
        assert len(result.tokens) > 0

    def test_custom_stop_words(self):
        tokenizer = Tokenizer(custom_stop_words={"quick", "brown"})
        result = tokenizer.tokenize("the quick brown fox")
        texts = [t.text.lower() for t in result.tokens]
        assert "quick" not in texts
        assert "brown" not in texts
        assert "fox" in texts

    def test_tokenize_batch(self):
        texts = ["hello world", "foo bar baz"]
        results = self.tokenizer.tokenize_batch(texts)
        assert len(results) == 2
        assert len(results[0].tokens) == 2
        assert len(results[1].tokens) == 3


class TestTokenizeResult:
    """Tests for TokenizeResult."""

    def test_len(self):
        result = TokenizeResult()
        assert len(result) == 0

    def test_doc_length(self):
        result = TokenizeResult(doc_length=42)
        assert result.doc_length == 42
