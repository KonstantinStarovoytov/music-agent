import pytest

from musicagent.core.camelot import (
    NONE,
    compatible,
    key_affinity,
    parse_camelot,
    transition,
)


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
        ("8A", "9A", True),  # +1 (boost +)
        ("8A", "7A", True),  # -1 (drop -)
        ("12A", "1A", True),  # wheel wraps
        ("8A", "8B", True),  # boost + (not the relative key!)
        ("8A", "7B", True),  # relative major
        ("8A", "10A", True),  # boost +++ (diagonal)
        ("8A", "9B", False),
        ("8A", "3B", False),
        ("8A", "2A", False),  # +6: the far side of the wheel
    ],
)
def test_compatible(a, b, ok):
    assert compatible(a, b) is ok


def test_affinity_ordering():
    assert (
        key_affinity("8A", "8A")
        > key_affinity("8A", "9A")
        > key_affinity("8A", "5A")
        > key_affinity("8A", "10A")
        > key_affinity("8A", "3B")
        == 0.0
    )


# The full harmonic-mixing table, one A row and one B row, exactly as printed.
# Every other row is the same pattern rotated around the wheel (checked below).
_ROW_1A = {
    "perfect match": ["1A", "12B"],
    "energy boost +": ["1B", "2A"],
    "energy boost ++": ["10A"],
    "energy boost +++": ["3A", "8A"],
    "energy drop -": ["12A"],
    "energy drop --": ["4A"],
    "energy drop ---": ["11A", "6A"],
    "mood change": ["4B"],
}
_ROW_1B = {
    "perfect match": ["1B", "2A"],
    "energy boost +": ["2B"],
    "energy boost ++": ["10B"],
    "energy boost +++": ["3B", "8B"],
    "energy drop -": ["1A", "12B"],
    "energy drop --": ["4B"],
    "energy drop ---": ["11B", "6B"],
    "mood change": ["10A"],
}
_DELTA = {
    "perfect match": 0,
    "energy boost +": 1,
    "energy boost ++": 2,
    "energy boost +++": 3,
    "energy drop -": -1,
    "energy drop --": -2,
    "energy drop ---": -3,
    "mood change": 0,
}


def _rotate(code: str, by: int) -> str:
    n, letter = int(code[:-1]), code[-1]
    return f"{(n - 1 + by) % 12 + 1}{letter}"


@pytest.mark.parametrize("source,row", [("1A", _ROW_1A), ("1B", _ROW_1B)])
@pytest.mark.parametrize("rotation", range(12))
def test_transition_table_matches_the_printed_chart(source, row, rotation):
    src = _rotate(source, rotation)
    listed = set()
    for label, targets in row.items():
        for target in targets:
            t = transition(src, _rotate(target, rotation))
            assert t.label == label, (src, target, t)
            assert t.energy_delta == _DELTA[label]
            assert t.affinity > 0.0
            listed.add(_rotate(target, rotation))
    # Everything the chart leaves out is not a transition at all.
    for n in range(1, 13):
        for letter in "AB":
            other = f"{n}{letter}"
            if other not in listed:
                assert transition(src, other) == NONE, (src, other)


def test_transition_is_directed():
    assert transition("8A", "9A").label == "energy boost +"
    assert transition("9A", "8A").label == "energy drop -"
    assert transition("8A", "8B").label == "energy boost +"
    assert transition("8B", "8A").label == "energy drop -"


def test_parenthesised_alternatives_are_weaker():
    assert transition("8A", "10A").affinity > transition("8A", "3A").affinity
    assert transition("8A", "6A").affinity > transition("8A", "1A").affinity
