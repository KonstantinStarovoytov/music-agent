import math

from musicagent.core.scoring import build_edges
from musicagent.models import Edge, SetPath, Track

ENERGY_WEIGHT = 0.4
# Weight of the harmonic push term: does the key change (boost/drop from the
# transition table) point the way the target curve moves at this step?
KEY_SHIFT_WEIGHT = 0.2
# Target-curve moves smaller than this per step count as a plateau, where any
# boost/drop is mildly out of place.
PLATEAU_EPS = 0.02


def target_energy(shape: str, pos: int, total: int) -> float:
    frac = pos / max(total - 1, 1)
    if shape == "build":
        return 0.2 + 0.8 * frac
    if shape == "peak_end":
        peak = 0.85
        return 0.4 + 0.6 * (frac / peak) if frac <= peak else 1.0
    if shape == "wave":
        return 0.6 + 0.3 * math.sin(2 * math.pi * frac)
    raise ValueError(f"unknown shape: {shape!r}")


def shift_fit(energy_delta: int, shape: str, pos: int, total: int) -> float:
    """How well a key change's harmonic push (-3..+3) matches the direction the
    target curve moves into position `pos`. +1 when fully aligned (a +++
    boost on a rising step), -1 when fully opposed, 0 for a neutral
    transition; on a plateau any push is a small negative."""
    if pos <= 0 or energy_delta == 0:
        return 0.0
    want = target_energy(shape, pos, total) - target_energy(shape, pos - 1, total)
    strength = energy_delta / 3.0
    if abs(want) < PLATEAU_EPS:
        return -0.5 * abs(strength)
    return strength if want > 0 else -strength


def find_path(
    tracks: list[Track],
    shape: str,
    beam_width: int = 8,
    *,
    edges: list[Edge] | None = None,
) -> SetPath:
    n = len(tracks)
    if n == 0:
        return SetPath(tracks=[], edge_scores=[])

    if edges is None:
        edges = build_edges(tracks)

    adj: dict[int, dict[int, float]] = {}
    delta: dict[tuple[int, int], int] = {}
    for e in edges:
        adj.setdefault(e.a, {})[e.b] = e.score
        delta[(e.a, e.b)] = e.energy_delta

    def fit(i: int, pos: int) -> float:
        # Known MVP limitation: normalizes against the input pool size (n), not
        # the eventual path length, so the curve can look compressed/stretched
        # when the returned path is shorter than the full pool.
        return 1.0 - abs(tracks[i].energy - target_energy(shape, pos, n))

    # beam items: (total_score, path, used)
    beam = [(fit(i, 0) * ENERGY_WEIGHT, [i], {i}) for i in range(n)]
    best = max(beam, key=lambda x: (len(x[1]), x[0]))
    while beam:
        nxt = []
        for score, path, used in beam:
            for j, s in adj.get(path[-1], {}).items():
                if j in used:
                    continue
                pos = len(path)
                nscore = (
                    score
                    + s
                    + ENERGY_WEIGHT * fit(j, pos)
                    + KEY_SHIFT_WEIGHT * shift_fit(delta[(path[-1], j)], shape, pos, n)
                )
                nxt.append((nscore, path + [j], used | {j}))
        if nxt:
            # Update best from the pre-truncation candidate list: a longest-so-far
            # path could score below the top beam_width candidates this round and
            # be dropped by the beam_width slice below, which would otherwise lose
            # the "longer path wins" guarantee once a round is dense enough.
            cand = max(nxt, key=lambda x: (len(x[1]), x[0]))
            if (len(cand[1]), cand[0]) > (len(best[1]), best[0]):
                best = cand
        nxt.sort(key=lambda x: x[0], reverse=True)
        beam = nxt[:beam_width]

    _, path, _ = best
    scores = [adj[path[k]][path[k + 1]] for k in range(len(path) - 1)]
    return SetPath(tracks=[tracks[i] for i in path], edge_scores=scores)
