import re
from typing import NamedTuple

_NOTE_TO_CAMELOT_MINOR = {
    "A": "8A",
    "A#": "3A",
    "BB": "3A",
    "B": "10A",
    "C": "5A",
    "C#": "12A",
    "DB": "12A",
    "D": "7A",
    "D#": "2A",
    "EB": "2A",
    "E": "9A",
    "F": "4A",
    "F#": "11A",
    "GB": "11A",
    "G": "6A",
    "G#": "1A",
    "AB": "1A",
}
_NOTE_TO_CAMELOT_MAJOR = {
    "C": "8B",
    "C#": "3B",
    "DB": "3B",
    "D": "10B",
    "D#": "5B",
    "EB": "5B",
    "E": "12B",
    "F": "7B",
    "F#": "2B",
    "GB": "2B",
    "G": "9B",
    "G#": "4B",
    "AB": "4B",
    "A": "11B",
    "A#": "6B",
    "BB": "6B",
    "B": "1B",
}
_CAMELOT_RE = re.compile(r"^(1[0-2]|[1-9])([AB])$")
_KEY_RE = re.compile(r"^([A-Ga-g][#b]?)\s*(m|min|minor|maj|major)?$")


def parse_camelot(key: str) -> str:
    raw = key.strip()
    if m := _CAMELOT_RE.match(raw.upper()):
        return f"{m.group(1)}{m.group(2)}"
    if m := _KEY_RE.match(raw):
        note = m.group(1).upper()
        qual = (m.group(2) or "major").lower()
        table = (
            _NOTE_TO_CAMELOT_MINOR
            if qual.startswith("m") and qual not in ("maj", "major")
            else _NOTE_TO_CAMELOT_MAJOR
        )
        if note in table:
            return table[note]
    raise ValueError(f"unrecognized key: {key!r}")


def _split(code: str) -> tuple[int, str]:
    m = _CAMELOT_RE.match(code)
    if not m:
        raise ValueError(f"not a camelot code: {code!r}")
    return int(m.group(1)), m.group(2)


class Transition(NamedTuple):
    """Harmonic meaning of moving from one Camelot key to another.

    affinity: how smooth the change sounds (0 = not a recognised transition).
    energy_delta: the push the key change gives the floor, -3..+3.
    label: the table's name for it ("perfect match", "energy boost ++", ...).
    """

    affinity: float
    energy_delta: int
    label: str


NONE = Transition(0.0, 0, "none")

# The standard harmonic-mixing transition table (spec section 3), keyed on
# (letter of the source key, letter of the target key, (target - source) mod 12).
# Directed: 8A -> 9A is a boost, 9A -> 8A is a drop. The rows for A and B keys
# differ (the chart pairs nA with (n-1)B but nB with (n+1)A), so both
# are spelled out rather than derived from one formula.
_TABLE: dict[tuple[str, str, int], Transition] = {
    # from a minor (A) key
    ("A", "A", 0): Transition(1.0, 0, "perfect match"),
    ("A", "B", 11): Transition(1.0, 0, "perfect match"),  # e.g. 8A -> 7B
    ("A", "B", 0): Transition(0.85, 1, "energy boost +"),
    ("A", "A", 1): Transition(0.85, 1, "energy boost +"),
    ("A", "A", 9): Transition(0.6, 2, "energy boost ++"),
    ("A", "A", 2): Transition(0.5, 3, "energy boost +++"),
    ("A", "A", 7): Transition(0.4, 3, "energy boost +++"),
    ("A", "A", 11): Transition(0.85, -1, "energy drop -"),
    ("A", "A", 3): Transition(0.6, -2, "energy drop --"),
    ("A", "A", 10): Transition(0.5, -3, "energy drop ---"),
    ("A", "A", 5): Transition(0.4, -3, "energy drop ---"),
    ("A", "B", 3): Transition(0.5, 0, "mood change"),
    # from a major (B) key
    ("B", "B", 0): Transition(1.0, 0, "perfect match"),
    ("B", "A", 1): Transition(1.0, 0, "perfect match"),  # e.g. 7B -> 8A
    ("B", "B", 1): Transition(0.85, 1, "energy boost +"),
    ("B", "B", 9): Transition(0.6, 2, "energy boost ++"),
    ("B", "B", 2): Transition(0.5, 3, "energy boost +++"),
    ("B", "B", 7): Transition(0.4, 3, "energy boost +++"),
    ("B", "A", 0): Transition(0.85, -1, "energy drop -"),
    ("B", "B", 11): Transition(0.85, -1, "energy drop -"),
    ("B", "B", 3): Transition(0.6, -2, "energy drop --"),
    ("B", "B", 10): Transition(0.5, -3, "energy drop ---"),
    ("B", "B", 5): Transition(0.4, -3, "energy drop ---"),
    ("B", "A", 9): Transition(0.5, 0, "mood change"),
}


def transition(a: str, b: str) -> Transition:
    """Directed transition from key `a` to key `b`; NONE if not in the table."""
    na, la = _split(a)
    nb, lb = _split(b)
    return _TABLE.get((la, lb, (nb - na) % 12), NONE)


def compatible(a: str, b: str) -> bool:
    return transition(a, b).affinity > 0.0


def key_affinity(a: str, b: str) -> float:
    return transition(a, b).affinity
