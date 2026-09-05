import sys

import pytest

from musicagent.db import TrackCache, get_engine, init_db
from musicagent.graph import build_graph, get_langfuse_handler, run_set
from musicagent.llm import _Explanations
from musicagent.models import SetRequest, Track, TrackRef


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        if self.schema is SetRequest:
            return SetRequest(
                tracks=[
                    TrackRef(artist="a", title="t1"),
                    TrackRef(artist="b", title="t2"),
                ],
                energy_shape="build",
            )
        return _Explanations(explanations=["works"], summary="ok")


class AllUnresolvedLLM:
    """Parses to two tracks that are never present in the cache, so both fail
    enrichment (no network access happens in tests, so unknown refs cannot resolve)."""

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        if self.schema is SetRequest:
            return SetRequest(
                tracks=[
                    TrackRef(artist="ghost", title="nowhere"),
                    TrackRef(artist="phantom", title="nothing"),
                ],
                energy_shape="build",
            )
        return _Explanations(explanations=[], summary="no tracks resolved")


@pytest.mark.asyncio
async def test_run_set_end_to_end_offline():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)
    cache.put(
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A", energy=0.3)
    )
    cache.put(
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A", energy=0.7)
    )
    result = await run_set("a t1, b t2, build it up", cache=cache, llm=FakeLLM())
    assert len(result.transitions) == 1
    assert result.transitions[0].explanation == "works"
    assert result.unresolved == []


@pytest.mark.asyncio
async def test_run_set_all_tracks_unresolved_still_returns_valid_result():
    """When every track fails enrichment, the pathfinder and explain node must
    handle the empty track list gracefully instead of crashing."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)  # empty cache; refs below are not in it

    result = await run_set(
        "ghost nowhere, phantom nothing, build it up",
        cache=cache,
        llm=AllUnresolvedLLM(),
    )

    assert result.transitions == []
    assert len(result.unresolved) == 2
    assert {r.artist for r in result.unresolved} == {"ghost", "phantom"}


def test_get_langfuse_handler_returns_empty_list_without_key(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    sys.modules.pop("langfuse", None)
    sys.modules.pop("langfuse.langchain", None)

    result = get_langfuse_handler()

    assert result == []
    assert "langfuse" not in sys.modules
    assert "langfuse.langchain" not in sys.modules


def test_build_graph_constructible_with_no_env_vars(monkeypatch):
    monkeypatch.setattr("os.environ", {}, raising=False)
    graph = build_graph()
    assert graph is not None
