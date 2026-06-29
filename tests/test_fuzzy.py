"""Tests for the fuzzy matching (Levenshtein distance) module."""


from minisearch.fuzzy import (
    damerau_levenshtein_distance,
    find_closest_matches,
    levenshtein_distance,
    levenshtein_within_distance,
)


class TestLevenshteinDistance:
    """Tests for Levenshtein distance computation."""

    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0
        assert levenshtein_distance("hello", "") == 5
        assert levenshtein_distance("", "hello") == 5

    def test_single_insertion(self):
        assert levenshtein_distance("cat", "cats") == 1
        assert levenshtein_distance("cat", "scat") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("cats", "cat") == 1
        assert levenshtein_distance("scat", "cat") == 1

    def test_single_substitution(self):
        assert levenshtein_distance("cat", "bat") == 1
        assert levenshtein_distance("cat", "car") == 1

    def test_multiple_edits(self):
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("saturday", "sunday") == 3

    def test_completely_different(self):
        assert levenshtein_distance("abc", "xyz") == 3

    def test_single_character(self):
        assert levenshtein_distance("a", "b") == 1
        assert levenshtein_distance("a", "ab") == 1

    def test_symmetry(self):
        assert levenshtein_distance("hello", "world") == levenshtein_distance("world", "hello")

    def test_long_strings(self):
        s1 = "algorithm" * 10
        s2 = "altruistic" * 10
        dist = levenshtein_distance(s1, s2)
        assert dist > 0

    def test_unicode(self):
        assert levenshtein_distance("café", "cafe") == 1
        assert levenshtein_distance("hello", "héllo") == 1


class TestLevenshteinWithinDistance:
    """Tests for within-distance checking."""

    def test_within_distance(self):
        assert levenshtein_within_distance("cat", "cats", 1) is True
        assert levenshtein_within_distance("cat", "dog", 1) is False

    def test_exact_match(self):
        assert levenshtein_within_distance("hello", "hello", 0) is True

    def test_zero_distance(self):
        assert levenshtein_within_distance("hello", "hell", 0) is False

    def test_large_distance(self):
        assert levenshtein_within_distance("hello", "world", 10) is True

    def test_early_termination(self):
        # This should terminate early due to length difference
        assert levenshtein_within_distance("a", "abcdefghij", 3) is False

    def test_symmetry(self):
        assert levenshtein_within_distance("cat", "cats", 1) == \
            levenshtein_within_distance("cats", "cat", 1)


class TestFindClosestMatches:
    """Tests for finding closest matches from a list of candidates."""

    def test_exact_match(self):
        matches = find_closest_matches("hello", ["hello", "world", "foo"])
        assert len(matches) == 1
        assert matches[0] == ("hello", 0)

    def test_close_matches(self):
        matches = find_closest_matches("hell", ["hello", "world", "foo"], max_distance=2)
        assert len(matches) >= 1
        assert matches[0][0] == "hello"

    def test_no_matches(self):
        matches = find_closest_matches("xyz", ["hello", "world"], max_distance=1)
        assert len(matches) == 0

    def test_sorted_by_distance(self):
        matches = find_closest_matches("cat", ["cats", "category", "car"], max_distance=3)
        distances = [m[1] for m in matches]
        assert distances == sorted(distances)

    def test_max_results(self):
        candidates = [f"test{i}" for i in range(20)]
        matches = find_closest_matches("test", candidates, max_distance=5, max_results=5)
        assert len(matches) <= 5

    def test_empty_candidates(self):
        matches = find_closest_matches("hello", [], max_distance=2)
        assert len(matches) == 0


class TestDamerauLevenshteinDistance:
    """Tests for Damerau-Levenshtein distance (with transpositions)."""

    def test_identical(self):
        assert damerau_levenshtein_distance("hello", "hello") == 0

    def test_transposition(self):
        # Damerau-Levenshtein counts transposition as 1 edit
        assert damerau_levenshtein_distance("ab", "ba") == 1

    def test_insertion(self):
        assert damerau_levenshtein_distance("cat", "cats") == 1

    def test_deletion(self):
        assert damerau_levenshtein_distance("cats", "cat") == 1

    def test_substitution(self):
        assert damerau_levenshtein_distance("cat", "bat") == 1

    def test_complex(self):
        # "ca" -> "abc" requires 3 edits
        assert damerau_levenshtein_distance("ca", "abc") == 3

    def test_empty(self):
        assert damerau_levenshtein_distance("", "") == 0
        assert damerau_levenshtein_distance("abc", "") == 3

    def test_symmetry(self):
        assert damerau_levenshtein_distance("hello", "world") == \
            damerau_levenshtein_distance("world", "hello")
