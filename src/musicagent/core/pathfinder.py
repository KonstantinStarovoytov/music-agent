import math

from musicagent.core.scoring import build_edges
from musicagent.models import Edge, SetPath, Track

ENERGY_WEIGHT = 0.4


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
    for e in edges:
        adj.setdefault(e.a, {})[e.b] = e.score

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
                nscore = score + s + ENERGY_WEIGHT * fit(j, len(path))
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
