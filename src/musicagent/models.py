from typing import Literal

from pydantic import BaseModel


class TrackRef(BaseModel):
    artist: str
    title: str


class Track(BaseModel):
    ref: TrackRef
    bpm: float
    camelot: str
    energy: float = 0.5
    duration_s: int | None = None
    tags: list[str] = []
    source: str = "unknown"


class SetRequest(BaseModel):
    tracks: list[TrackRef]
    duration_min: int | None = None
    energy_shape: Literal["build", "peak_end", "wave"] = "peak_end"


class Edge(BaseModel):
    a: int
    b: int
    score: float


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
    unresolved: list[TrackRef] = []
    # Tracks that enriched fine (so they're not in `unresolved`) but ended up
    # with no place in the final path -- either no compatible edge to any
    # other track, or trimmed from the end to fit a requested duration_min.
    omitted: list[TrackRef] = []
