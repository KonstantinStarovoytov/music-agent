from musicagent.core.camelot import key_affinity
from musicagent.models import Edge, Track

BPM_TOLERANCE = 0.06


def bpm_ok(a: float, b: float) -> bool:
    return abs(a - b) / max(a, b) <= BPM_TOLERANCE


def edge_score(a: Track, b: Track) -> float:
    key = key_affinity(a.camelot, b.camelot)
    if key == 0.0 or not bpm_ok(a.bpm, b.bpm):
        return 0.0
    bpm_closeness = 1.0 - (abs(a.bpm - b.bpm) / max(a.bpm, b.bpm)) / BPM_TOLERANCE
    energy_smoothness = 1.0 - min(abs(a.energy - b.energy), 1.0)
    return 0.5 * key + 0.3 * bpm_closeness + 0.2 * energy_smoothness


def build_edges(tracks: list[Track]) -> list[Edge]:
    return [
        Edge(a=i, b=j, score=s)
        for i, a in enumerate(tracks)
        for j, b in enumerate(tracks)
        if i != j and (s := edge_score(a, b)) > 0.0
    ]
