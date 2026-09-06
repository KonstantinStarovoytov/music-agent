from typing import Literal

from pydantic import BaseModel


class TrackRef(BaseModel):
    artist: str
    title: str


class UnresolvedTrack(BaseModel):
    """A requested track that never became a usable `Track`, with a
    machine-readable reason plus a human-readable sentence explaining it
    (surfaced to the user via the API and, in plain language, the
    explain_set summary)."""

    artist: str
    title: str
    # not_found: no provider recognised the track at all.
    # no_key: bpm was found but no provider supplied a musical key.
    # no_bpm: a musical key was found but no provider supplied a tempo.
    # timeout: the per-track deadline (ENRICH_DEADLINE_S) expired first.
    # error: an unexpected exception occurred while enriching this track.
    reason: Literal["not_found", "no_key", "no_bpm", "timeout", "error"]
    message: str


class Track(BaseModel):
    ref: TrackRef
    bpm: float
    camelot: str
    energy: float = 0.5
    duration_s: int | None = None
    tags: list[str] = []
    source: str = "unknown"
    # Algorithmic key-detection confidence (AcousticBrainz `tonal.key_strength`,
    # 0..1) when the camelot key came from that provider; None otherwise.
    key_confidence: float | None = None


class SetRequest(BaseModel):
    tracks: list[TrackRef]
    duration_min: int | None = None
    energy_shape: Literal["build", "peak_end", "wave"] = "peak_end"


class Edge(BaseModel):
    a: int
    b: int
    score: float
    # Harmonic push of the key change a -> b (-3..+3) and the transition
    # table's name for it; see core/camelot.py::transition.
    energy_delta: int = 0
    label: str = ""


class TransitionGraph(BaseModel):
    edges: list[Edge]


class SetPath(BaseModel):
    tracks: list[Track]
    edge_scores: list[float]


class Transition(BaseModel):
    from_track: TrackRef
    to_track: TrackRef
    explanation: str


class SetResult(BaseModel):
    transitions: list[Transition]
    summary: str
    unresolved: list[UnresolvedTrack] = []
    # Tracks that enriched fine (so they're not in `unresolved`) but ended up
    # with no place in the final path -- either no compatible edge to any
    # other track, or trimmed from the end to fit a requested duration_min.
    omitted: list[TrackRef] = []
