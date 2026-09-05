import pytest

from musicagent.core.camelot import compatible, key_affinity, parse_camelot


@pytest.mark.parametrize(
    "raw,expected",
    [("8A", "8A"), ("Am", "8A"), ("A minor", "8A"), ("C", "8B"), ("F# min", "11A")],
)
def test_parse_camelot(raw, expected):
    assert parse_camelot(raw) == expected


def test_parse_unknown_raises():
    with pytest.raises(ValueError):
        parse_camelot("H#")


@pytest.mark.parametrize(
    "a,b,ok",
    [
        ("8A", "8A", True),   # same
        ("8A", "9A", True),   # +1
        ("8A", "7A", True),   # -1
        ("12A", "1A", True),  # wheel wraps
        ("8A", "8B", True),   # relative
        ("8A", "10A", False),
        ("8A", "9B", False),
    ],
)
def test_compatible(a, b, ok):
    assert compatible(a, b) is ok


def test_affinity_ordering():
    assert key_affinity("8A", "8A") > key_affinity("8A", "9A") > key_affinity("8A", "3B")
