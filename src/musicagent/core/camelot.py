import re

_NOTE_TO_CAMELOT_MINOR = {
    "A": "8A", "A#": "9A", "BB": "9A", "B": "10A", "C": "5A", "C#": "12A",
    "DB": "12A", "D": "7A", "D#": "2A", "EB": "2A", "E": "9A", "F": "4A",
    "F#": "11A", "GB": "11A", "G": "6A", "G#": "1A", "AB": "1A",
}
_NOTE_TO_CAMELOT_MAJOR = {
    "C": "8B", "C#": "3B", "DB": "3B", "D": "10B", "D#": "5B", "EB": "5B",
    "E": "12B", "F": "7B", "F#": "2B", "GB": "2B", "G": "9B", "G#": "4B",
    "AB": "4B", "A": "11B", "A#": "6B", "BB": "6B", "B": "1B",
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
        table = _NOTE_TO_CAMELOT_MINOR if qual.startswith("m") and qual not in ("maj", "major") else _NOTE_TO_CAMELOT_MAJOR
        if note in table:
            return table[note]
    raise ValueError(f"unrecognized key: {key!r}")


def _split(code: str) -> tuple[int, str]:
    m = _CAMELOT_RE.match(code)
    if not m:
        raise ValueError(f"not a camelot code: {code!r}")
    return int(m.group(1)), m.group(2)


def compatible(a: str, b: str) -> bool:
    return key_affinity(a, b) > 0.0


def key_affinity(a: str, b: str) -> float:
    na, la = _split(a)
    nb, lb = _split(b)
    if (na, la) == (nb, lb):
        return 1.0
    if la == lb and (na - nb) % 12 in (1, 11):
        return 0.8
    if na == nb and la != lb:
        return 0.8
    return 0.0
