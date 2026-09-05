import pytest

from musicagent.core.camelot import compatible, key_affinity, parse_camelot


@pytest.mark.parametrize(
    "raw,expected",
    [
        # Original brief cases
        ("8A", "8A"),
        ("Am", "8A"),
        ("A minor", "8A"),
        ("C", "8B"),
        ("F# min", "11A"),
        # Critical fix: A#/Bb minor enharmonic aliases (required)
        ("A# minor", "3A"),
        ("Bb min", "3A"),
        # Enharmonic aliases with lowercase (required lowercase-input case)
        ("a# minor", "3A"),
        ("bb min", "3A"),
        # Lowercase inputs (required lowercase-input case)
        ("a minor", "8A"),
        ("c", "8B"),
        # Explicit maj/major qualifiers (required explicit qualifier case)
        ("C major", "8B"),
        ("A maj", "11B"),
        ("f minor", "4A"),
        ("f major", "7B"),
        # All 24 wheel positions - Minor (A column) - comprehensive round-trip coverage
        ("G# min", "1A"),
        ("D# min", "2A"),
        ("A# min", "3A"),
        ("F min", "4A"),
        ("C min", "5A"),
        ("G min", "6A"),
        ("D min", "7A"),
        ("E min", "9A"),
        ("B min", "10A"),
        ("C# min", "12A"),
        # Enharmonic equivalents for minor - ensure all table entries are tested
        ("Ab min", "1A"),
        ("Eb min", "2A"),
        ("Db min", "12A"),
        ("Gb min", "11A"),
        # All 24 wheel positions - Major (B column) - comprehensive round-trip coverage
        ("B maj", "1B"),
        ("F# maj", "2B"),
        ("C# maj", "3B"),
        ("G# maj", "4B"),
        ("D# maj", "5B"),
        ("A# maj", "6B"),
        ("F maj", "7B"),
        ("C maj", "8B"),
        ("G maj", "9B"),
        ("D maj", "10B"),
        ("E maj", "12B"),
        # Enharmonic equivalents for major - ensure all table entries are tested
        ("Gb maj", "2B"),
        ("Db maj", "3B"),
        ("Ab maj", "4B"),
        ("Eb maj", "5B"),
        ("Bb maj", "6B"),
    ],
)
def test_parse_camelot(raw, expected):
    assert parse_camelot(raw) == expected


def test_parse_unknown_raises():
    with pytest.raises(ValueError):
        parse_camelot("H#")


@pytest.mark.parametrize(
    "a,b,ok",
    [
        ("8A", "8A", True),  # same
        ("8A", "9A", True),  # +1
        ("8A", "7A", True),  # -1
        ("12A", "1A", True),  # wheel wraps
        ("8A", "8B", True),  # relative
        ("8A", "10A", False),
        ("8A", "9B", False),
    ],
)
def test_compatible(a, b, ok):
    assert compatible(a, b) is ok


def test_affinity_ordering():
    assert (
        key_affinity("8A", "8A") > key_affinity("8A", "9A") > key_affinity("8A", "3B")
    )
