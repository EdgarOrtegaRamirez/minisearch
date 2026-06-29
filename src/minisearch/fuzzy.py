"""
Levenshtein distance for fuzzy matching.

Implements the Levenshtein edit distance algorithm using dynamic programming.
Used for fuzzy/approximate string matching in search queries.
"""

from __future__ import annotations


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.

    The Levenshtein distance is the minimum number of single-character
    edits (insertions, deletions, or substitutions) required to change
    one string into the other.

    Uses the standard dynamic programming approach with O(n*m) time
    and O(min(n,m)) space complexity.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Edit distance (non-negative integer).
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    # Optimize by using shorter string for column dimension
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    len1 = len(s1)
    len2 = len(s2)

    # Previous and current row of distances
    prev_row = list(range(len1 + 1))
    curr_row = [0] * (len1 + 1)

    for j in range(1, len2 + 1):
        curr_row[0] = j
        for i in range(1, len1 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[i] = min(
                prev_row[i] + 1,       # deletion
                curr_row[i - 1] + 1,   # insertion
                prev_row[i - 1] + cost, # substitution
            )
        prev_row, curr_row = curr_row, prev_row

    return prev_row[len1]


def levenshtein_within_distance(s1: str, s2: str, max_distance: int) -> bool:
    """
    Check if two strings are within a given Levenshtein distance.

    Early termination optimization: stops computing as soon as
    any row exceeds the maximum distance.

    Args:
        s1: First string.
        s2: Second string.
        max_distance: Maximum allowed edit distance.

    Returns:
        True if edit distance <= max_distance.
    """
    if abs(len(s1) - len(s2)) > max_distance:
        return False
    if s1 == s2:
        return True

    # Optimize by using shorter string for column dimension
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    len1 = len(s1)
    len2 = len(s2)

    prev_row = list(range(len1 + 1))
    curr_row = [0] * (len1 + 1)

    for j in range(1, len2 + 1):
        curr_row[0] = j
        min_in_row = curr_row[0]

        for i in range(1, len1 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[i] = min(
                prev_row[i] + 1,
                curr_row[i - 1] + 1,
                prev_row[i - 1] + cost,
            )
            min_in_row = min(min_in_row, curr_row[i])

        # Early termination: if minimum in this row exceeds max_distance,
        # no need to continue
        if min_in_row > max_distance:
            return False

        prev_row, curr_row = curr_row, prev_row

    return prev_row[len1] <= max_distance


def find_closest_matches(
    target: str,
    candidates: list[str],
    max_distance: int = 2,
    max_results: int = 5,
) -> list[tuple[str, int]]:
    """
    Find the closest matches to a target string from a list of candidates.

    Args:
        target: The string to match against.
        candidates: List of candidate strings.
        max_distance: Maximum allowed edit distance.
        max_results: Maximum number of results to return.

    Returns:
        List of (candidate, distance) tuples, sorted by distance ascending.
    """
    matches: list[tuple[str, int]] = []

    for candidate in candidates:
        dist = levenshtein_distance(target, candidate)
        if dist <= max_distance:
            matches.append((candidate, dist))

    matches.sort(key=lambda x: (x[1], x[0]))
    return matches[:max_results]


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Damerau-Levenshtein distance between two strings.

    Extends Levenshtein distance by also allowing transpositions of
    two adjacent characters as a single edit operation.

    Uses the optimal string alignment algorithm with O(n*m) time
    and O(min(n,m)) space complexity.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        Damerau-Levenshtein distance (non-negative integer).
    """
    if s1 == s2:
        return 0
    if not s1:
        return len(s2)
    if not s2:
        return len(s1)

    len1 = len(s1)
    len2 = len(s2)

    # Ensure s1 is the shorter string for space optimization
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    # Use three rows for the DP table
    # prev_prev_row[i] = dp[i-2] (for transposition check)
    # prev_row[i] = dp[i-1]
    # curr_row[i] = dp[i]
    prev_prev = list(range(len1 + 1))
    prev_row = list(range(len1 + 1))
    curr_row = [0] * (len1 + 1)

    for j in range(1, len2 + 1):
        curr_row[0] = j
        for i in range(1, len1 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr_row[i] = min(
                prev_row[i] + 1,           # deletion
                curr_row[i - 1] + 1,       # insertion
                prev_row[i - 1] + cost,    # substitution
            )
            # Transposition (optimal string alignment variant)
            if (
                i > 1
                and j > 1
                and s1[i - 1] == s2[j - 2]
                and s1[i - 2] == s2[j - 1]
            ):
                curr_row[i] = min(curr_row[i], prev_prev[i - 2] + 1)

        # Rotate rows
        prev_prev = prev_row[:]
        prev_row = curr_row[:]
        curr_row = [0] * (len1 + 1)

    return prev_row[len1]
