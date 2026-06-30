"""
Text tokenizer with stop word removal, Porter stemming, and position tracking.

The tokenizer splits text into tokens, normalizes them (lowercase, stem),
and tracks their positions for phrase query support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Common English stop words
STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "his",
        "her",
        "not",
        "no",
        "nor",
        "so",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "all",
        "also",
        "any",
        "because",
        "before",
        "below",
        "between",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "then",
        "there",
        "here",
        "when",
        "where",
        "why",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "if",
        "while",
        "during",
        "into",
        "through",
        "up",
        "out",
        "off",
        "over",
        "under",
        "further",
        "once",
        "only",
    }
)


@dataclass
class Token:
    """A single token with its text, position, and optional stem."""

    text: str
    position: int  # character offset in original text
    stem: str | None = None  # stemmed form if available

    @property
    def key(self) -> str:
        """The normalized key used for indexing (stemmed if available, else lowercase)."""
        return self.stem if self.stem is not None else self.text.lower()


@dataclass
class TokenizeResult:
    """Result of tokenizing a document."""

    tokens: list[Token] = field(default_factory=list)
    doc_length: int = 0  # number of non-stop-word tokens

    def __len__(self) -> int:
        return len(self.tokens)


class PorterStemmer:
    """
    Simplified Porter stemmer implementation.

    Implements the core steps of the Porter stemming algorithm:
    1. Handle plurals and -ed/-ing
    2. Turn terminal y to i when there is another vowel in the stem
    3. Map double suffixes to single ones
    4. Map doubles to single
    5. Remove suffixes: -ational, -tional, -enci, -anci, -izer, -alli,
       -entli, -eli, -ousli, -ization, -ation, -ator, -alism, -iveness,
       -fulness, -ousness, -aliti, -iviti, -biliti, -fulness, -ousness
    6. Remove trailing -e
    """

    def stem(self, word: str) -> str:
        """Apply Porter stemming algorithm to a word."""
        word = word.lower().strip()
        if len(word) <= 2:
            return word

        # Step 1: Handle common suffixes
        word = self._step1(word)
        # Step 2: Handle -ational, -tional, etc.
        word = self._step2(word)
        # Step 3: Handle -al, -ence, -ance, etc.
        word = self._step3(word)
        # Step 4: Handle double suffixes
        word = self._step4(word)
        # Step 5: Remove trailing -e
        word = self._step5(word)

        return word

    def _has_vowel(self, stem: str) -> bool:
        """Check if stem contains a vowel."""
        return bool(re.search(r"[aeiouy]", stem))

    def _step1(self, word: str) -> str:
        """Handle plurals and -ed/-ing."""
        if word.endswith("sses"):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-2]
        if word.endswith("ss"):
            return word
        if word.endswith("s"):
            return word[:-1]
        if word.endswith("ed") and self._has_vowel(word[:-2]):
            base = word[:-2]
            # Remove doubled consonant at end (e.g., "troubled" -> "troubl")
            if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in "aeiouy":
                return base[:-1]
            return base
        if word.endswith("ing") and self._has_vowel(word[:-3]):
            base = word[:-3]
            # Remove doubled consonant at end (e.g., "running" -> "runn" -> "run")
            if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in "aeiouy":
                return base[:-1]
            return base
        return word

    def _step2(self, word: str) -> str:
        """Map double suffixes to single."""
        suffixes = {
            "ational": "ate",
            "tional": "tion",
            "enci": "ence",
            "anci": "ance",
            "izer": "ize",
            "abli": "able",
            "alli": "al",
            "entli": "ent",
            "eli": "e",
            "ousli": "ous",
            "ization": "ize",
            "ation": "ate",
            "ator": "ate",
            "alism": "al",
            "iveness": "ive",
            "fulness": "ful",
            "ousness": "ous",
            "aliti": "al",
            "iviti": "ive",
            "biliti": "ble",
            "logi": "log",
            "ivity": "ive",
            "ly": "",
        }
        for suffix, replacement in suffixes.items():
            if word.endswith(suffix) and self._has_vowel(word[: -len(suffix)]):
                return word[: -len(suffix)] + replacement
        return word

    def _step3(self, word: str) -> str:
        """Remove suffixes: -al, -ence, -ance, -er, -able, etc."""
        suffixes = ["al", "ance", "ence", "er", "able", "ible", "ment"]
        for suffix in suffixes:
            if word.endswith(suffix) and self._has_vowel(word[: -len(suffix)]):
                return word[: -len(suffix)]
        return word

    def _step4(self, word: str) -> str:
        """Handle various suffixes."""
        suffixes = [
            "ement",
            "ment",
            "ence",
            "ance",
            "able",
            "ible",
            "ant",
            "ent",
            "ion",
            "ism",
            "ate",
            "iti",
            "ous",
            "ive",
            "ize",
            "ful",
        ]
        for suffix in suffixes:
            if word.endswith(suffix) and self._has_vowel(word[: -len(suffix)]):
                # For "ion", only strip if preceded by s or t
                if suffix == "ion":
                    prev_char_pos = len(word) - len(suffix) - 1
                    if len(word) > len(suffix) + 1 and word[prev_char_pos] in "st":
                        return word[: -len(suffix)]
                else:
                    return word[: -len(suffix)]
        return word

    def _step5(self, word: str) -> str:
        """Remove trailing -e."""
        if word.endswith("e") and len(word) > 1:
            stem = word[:-1]
            if self._has_vowel(stem):
                # Only remove if stem has >1 vowel or ends in double consonant
                vowel_count = len(re.findall(r"[aeiouy]", stem))
                ends_double = len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiouy"
                if vowel_count > 1 or ends_double:
                    return stem
        return word


class Tokenizer:
    """
    Text tokenizer with stop word removal and Porter stemming.

    Splits text into tokens, tracks positions, optionally removes stop words,
    and applies Porter stemming for normalized indexing.
    """

    # Pattern to match words (letters, digits, underscores, hyphens within words)
    _WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*")

    # Pattern to split camelCase and PascalCase
    _CAMEL_RE = re.compile(r"([a-z])([A-Z])")

    # Pattern to split snake_case and kebab-case
    _SEPARATOR_RE = re.compile(r"[-_]")

    def __init__(
        self,
        remove_stop_words: bool = True,
        stem: bool = True,
        split_camel_case: bool = True,
        min_token_length: int = 1,
        custom_stop_words: set[str] | None = None,
    ):
        self.remove_stop_words = remove_stop_words
        self.stem = stem
        self.split_camel_case = split_camel_case
        self.min_token_length = min_token_length
        self.stop_words = custom_stop_words if custom_stop_words is not None else STOP_WORDS
        self._stemmer = PorterStemmer() if stem else None

    def tokenize(self, text: str) -> TokenizeResult:
        """
        Tokenize text into a list of Token objects.

        Args:
            text: Input text to tokenize.

        Returns:
            TokenizeResult with tokens and document length.
        """
        tokens: list[Token] = []

        # Find all word matches
        for match in self._WORD_RE.finditer(text):
            raw = match.group()
            start_pos = match.start()

            # Optionally split camelCase
            sub_tokens = self._split_token(raw, start_pos)
            for sub_text, sub_pos in sub_tokens:
                # Skip stop words
                if self.remove_stop_words and sub_text.lower() in self.stop_words:
                    continue

                # Skip short tokens
                if len(sub_text) < self.min_token_length:
                    continue

                # Apply stemming
                stem = self._stemmer.stem(sub_text) if self._stemmer and self.stem else None

                tokens.append(
                    Token(
                        text=sub_text,
                        position=sub_pos,
                        stem=stem,
                    )
                )

        return TokenizeResult(tokens=tokens, doc_length=len(tokens))

    def _split_token(self, raw: str, start_pos: int) -> list[tuple[str, int]]:
        """
        Split a raw token into sub-tokens.

        Handles camelCase splitting and snake_case/kebab-case splitting.
        Returns list of (token_text, position) tuples.
        """
        result: list[tuple[str, int]] = []

        if self.split_camel_case:
            # Split camelCase: "camelCase" -> ["camel", "Case"]
            split_text = self._CAMEL_RE.sub(r"\1 \2", raw)
            parts = split_text.split()
        else:
            parts = [raw]

        pos = start_pos
        for part in parts:
            # Further split by separators if needed
            if self._SEPARATOR_RE.search(part):
                sub_parts = self._SEPARATOR_RE.split(part)
                for sp in sub_parts:
                    if sp:
                        result.append((sp, pos))
                        pos += len(sp) + 1  # +1 for separator
            else:
                result.append((part, pos))
                pos += len(part) + 1  # +1 for potential separator

        return result

    def tokenize_batch(self, texts: list[str]) -> list[TokenizeResult]:
        """Tokenize multiple texts."""
        return [self.tokenize(text) for text in texts]
