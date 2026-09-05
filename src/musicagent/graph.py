import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from musicagent.core.pathfinder import find_path
from musicagent.core.scoring import build_edges
from musicagent.enrichment import enrich_all
from musicagent.llm import explain_set, parse_input
from musicagent.models import (
    SetPath,
    SetRequest,
    SetResult,
    Track,
    TrackRef,
    TransitionGraph,
    UnresolvedTrack,
)

MAX_TRACKS = 30


class SetState(TypedDict, total=False):
    text: str
    request: SetRequest
    tracks: list[Track]
    unresolved: list[UnresolvedTrack]
    transition_graph: TransitionGraph
    path: SetPath
    result: SetResult
    notice: str | None


def get_langfuse_handler() -> list:
    if not os.environ.get("LANGFUSE_SECRET_KEY"):
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def _dedupe_track_refs(refs: list[TrackRef]) -> list[TrackRef]:
    """Drop later duplicates of the same track, normalized on (artist, title)
    (stripped + lowercased). Two copies of the same track would otherwise
    score a perfect 1.0 edge against each other and get placed back-to-back
    by the pathfinder. Keeps the first occurrence's casing/spacing."""
    seen: set[tuple[str, str]] = set()
    deduped = []
    for ref in refs:
        key = (ref.artist.strip().lower(), ref.title.strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ref)
    return deduped


def _trim_to_duration(path: SetPath, duration_min: int) -> SetPath:
    """Trim tracks from the end of the path while the summed duration exceeds
    duration_min minutes, keeping at least 2 tracks. A track with no known
    duration_s counts as 0s. Trimmed tracks are still enriched tracks the
    caller knows about (they remain in state["tracks"]), so n_explain's
    "enriched minus placed" diff picks them up as omitted automatically."""
    budget_s = duration_min * 60
    tracks = list(path.tracks)
    edge_scores = list(path.edge_scores)
    while len(tracks) > 2 and sum(t.duration_s or 0 for t in tracks) > budget_s:
        tracks.pop()
        if edge_scores:
            edge_scores.pop()
    return SetPath(tracks=tracks, edge_scores=edge_scores)


def build_graph(cache=None, llm=None):
    def n_parse(state: SetState) -> SetState:
        request = parse_input(state["text"], llm=llm)
        deduped = _dedupe_track_refs(request.tracks)
        if len(deduped) != len(request.tracks):
            request = request.model_copy(update={"tracks": deduped})
        notice = None
        if len(request.tracks) > MAX_TRACKS:
            request = request.model_copy(update={"tracks": request.tracks[:MAX_TRACKS]})
            notice = (
                f"Request contained more than {MAX_TRACKS} tracks; "
                f"truncated to the first {MAX_TRACKS}."
            )
        return {"request": request, "notice": notice}

    async def n_enrich(state: SetState) -> SetState:
        tracks, unresolved = await enrich_all(state["request"].tracks, cache)
        return {"tracks": tracks, "unresolved": unresolved}

    def n_build_transition_graph(state: SetState) -> SetState:
        edges = build_edges(state["tracks"])
        return {"transition_graph": TransitionGraph(edges=edges)}

    def n_path(state: SetState) -> SetState:
        path = find_path(
            state["tracks"],
            state["request"].energy_shape,
            edges=state["transition_graph"].edges,
        )
        duration_min = state["request"].duration_min
        if duration_min is not None and duration_min > 0:
            path = _trim_to_duration(path, duration_min)
        return {"path": path}

    def n_explain(state: SetState) -> SetState:
        path = state["path"]
        placed = {(t.ref.artist, t.ref.title) for t in path.tracks}
        omitted = [
            t.ref for t in state["tracks"] if (t.ref.artist, t.ref.title) not in placed
        ]
        return {
            "result": explain_set(path, state["unresolved"], omitted=omitted, llm=llm)
        }

    g = StateGraph(SetState)
    g.add_node("parse_input", n_parse)
    g.add_node("enrich_tracks", n_enrich)
    g.add_node("build_transition_graph", n_build_transition_graph)
    g.add_node("find_set_path", n_path)
    g.add_node("explain_set", n_explain)
    g.set_entry_point("parse_input")
    g.add_edge("parse_input", "enrich_tracks")
    g.add_edge("enrich_tracks", "build_transition_graph")
    g.add_edge("build_transition_graph", "find_set_path")
    g.add_edge("find_set_path", "explain_set")
    g.add_edge("explain_set", END)
    return g.compile()


async def run_set(text: str, cache=None, llm=None, callbacks=None) -> SetResult:
    graph = build_graph(cache=cache, llm=llm)
    if callbacks is None:
        callbacks = get_langfuse_handler()
    state = await graph.ainvoke({"text": text}, config={"callbacks": callbacks})
    return state["result"]
