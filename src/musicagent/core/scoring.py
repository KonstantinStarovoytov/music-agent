from musicagent.core.camelot import transition
from musicagent.models import Edge, Track

BPM_TOLERANCE = 0.06


def bpm_ok(a: float, b: float) -> bool:
    return abs(a - b) / max(a, b) <= BPM_TOLERANCE


def edge_score(a: Track, b: Track) -> float:
    key = transition(a.camelot, b.camelot).affinity
    if key == 0.0 or not bpm_ok(a.bpm, b.bpm):
        return 0.0
    bpm_closeness = 1.0 - (abs(a.bpm - b.bpm) / max(a.bpm, b.bpm)) / BPM_TOLERANCE
    energy_smoothness = 1.0 - min(abs(a.energy - b.energy), 1.0)
    return 0.5 * key + 0.3 * bpm_closeness + 0.2 * energy_smoothness


def build_edges(tracks: list[Track]) -> list[Edge]:
    edges = []
    for i, a in enumerate(tracks):
        for j, b in enumerate(tracks):
            if i == j:
                continue
            score = edge_score(a, b)
            if score <= 0.0:
                continue
            t = transition(a.camelot, b.camelot)
            edges.append(
                Edge(a=i, b=j, score=score, energy_delta=t.energy_delta, label=t.label)
            )
    return edges
