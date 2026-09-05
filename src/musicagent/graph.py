import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from musicagent.core.pathfinder import find_path
from musicagent.enrichment import enrich_all
from musicagent.llm import explain_set, parse_input
from musicagent.models import SetPath, SetRequest, SetResult, Track, TrackRef


class SetState(TypedDict, total=False):
    text: str
    request: SetRequest
    tracks: list[Track]
    unresolved: list[TrackRef]
    path: SetPath
    result: SetResult


def get_langfuse_handler() -> list:
    if not os.environ.get("LANGFUSE_SECRET_KEY"):
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def build_graph(cache=None, llm=None):
    def n_parse(state: SetState) -> SetState:
        return {"request": parse_input(state["text"], llm=llm)}

    async def n_enrich(state: SetState) -> SetState:
        tracks, unresolved = await enrich_all(state["request"].tracks, cache)
        return {"tracks": tracks, "unresolved": unresolved}

    def n_path(state: SetState) -> SetState:
        return {"path": find_path(state["tracks"], state["request"].energy_shape)}

    def n_explain(state: SetState) -> SetState:
        return {"result": explain_set(state["path"], state["unresolved"], llm=llm)}

    g = StateGraph(SetState)
    g.add_node("parse_input", n_parse)
    g.add_node("enrich_tracks", n_enrich)
    g.add_node("find_set_path", n_path)
    g.add_node("explain_set", n_explain)
    g.set_entry_point("parse_input")
    g.add_edge("parse_input", "enrich_tracks")
    g.add_edge("enrich_tracks", "find_set_path")
    g.add_edge("find_set_path", "explain_set")
    g.add_edge("explain_set", END)
    return g.compile()


async def run_set(text: str, cache=None, llm=None, callbacks=None) -> SetResult:
    graph = build_graph(cache=cache, llm=llm)
    state = await graph.ainvoke({"text": text}, config={"callbacks": callbacks or []})
    return state["result"]
