import sys

import pytest
import respx

from musicagent.db import TrackCache, get_engine, init_db
from musicagent.graph import build_graph, get_langfuse_handler, run_set
from musicagent.llm import LLMOutputError, _Explanations
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
    enrichment (the test stubs the provider HTTP routes to return empty results,
    so this never touches a real socket)."""

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


class FailingParseLLM:
    """Always fails structured output for SetRequest, twice (past the one repair
    retry), so parse_input raises LLMOutputError."""

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        raise RuntimeError("parse always fails")


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
@respx.mock
async def test_run_set_all_tracks_unresolved_still_returns_valid_result(monkeypatch):
    """When every track fails enrichment, the pathfinder and explain node must
    handle the empty track list gracefully instead of crashing. All provider
    routes are stubbed so this test never touches a real network socket."""
    monkeypatch.setenv("GETSONGBPM_API_KEY", "test-key")
    deezer_route = respx.get(url__regex=r"api\.deezer\.com/search.*").respond(
        json={"data": []}
    )
    gsb_route = respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(
        json={"search": []}
    )

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
    # Prove the path actually went through enrichment against the stubbed
    # providers (not e.g. short-circuited some other way).
    assert deezer_route.call_count == 2
    assert gsb_route.call_count == 2
    # And that no unstubbed host was contacted: respx raises on any request
    # to a route it doesn't recognize when there's no catch-all, so reaching
    # this point with only the two routes above registered is itself proof.
    assert all(
        call.request.url.host in {"api.deezer.com", "api.getsongbpm.com"}
        for call in respx.calls
    )


@pytest.mark.asyncio
async def test_run_set_propagates_parse_failure():
    """A parse_input failure (LLMOutputError, after the one repair retry) must
    propagate out of run_set, not be swallowed, so callers have a defined
    failure mode."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)
    with pytest.raises(LLMOutputError):
        await run_set("anything", cache=cache, llm=FailingParseLLM())


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


def test_build_graph_has_all_five_spec_nodes():
    """The compiled graph must expose all five nodes from spec.md section 3:
    parse_input -> enrich_tracks -> build_transition_graph -> find_set_path ->
    explain_set -> END."""
    graph = build_graph()
    node_names = set(graph.get_graph().nodes.keys())
    assert {
        "parse_input",
        "enrich_tracks",
        "build_transition_graph",
        "find_set_path",
        "explain_set",
    } <= node_names


class _FakeCompiledGraph:
    """Stands in for the compiled LangGraph so these tests exercise only
    run_set's callback-defaulting logic, not the real graph machinery (which
    would reject a plain string/sentinel as a callback handler)."""

    def __init__(self):
        self.captured_configs = []

    async def ainvoke(self, state, config=None):
        self.captured_configs.append(config)
        return {"result": "fake-result"}


@pytest.mark.asyncio
async def test_run_set_defaults_callbacks_to_langfuse_handler(monkeypatch):
    """When callbacks is not passed, run_set must default to get_langfuse_handler()
    so the whole graph run is traced (spec.md section 6)."""
    sentinel = ["sentinel-handler"]
    fake_graph = _FakeCompiledGraph()

    monkeypatch.setattr("musicagent.graph.get_langfuse_handler", lambda: sentinel)
    monkeypatch.setattr("musicagent.graph.build_graph", lambda **kw: fake_graph)

    result = await run_set("a t1, b t2, build it up")

    assert result == "fake-result"
    assert fake_graph.captured_configs[-1]["callbacks"] == sentinel


@pytest.mark.asyncio
async def test_run_set_explicit_callbacks_win_over_langfuse_default(monkeypatch):
    """An explicitly passed callbacks list (including an empty one) must win over
    the langfuse default, and get_langfuse_handler must not even be called."""
    handler_calls = 0
    fake_graph = _FakeCompiledGraph()

    def fake_handler():
        nonlocal handler_calls
        handler_calls += 1
        return ["should-not-be-used"]

    monkeypatch.setattr("musicagent.graph.get_langfuse_handler", fake_handler)
    monkeypatch.setattr("musicagent.graph.build_graph", lambda **kw: fake_graph)

    result = await run_set("a t1, b t2, build it up", callbacks=[])

    assert result == "fake-result"
    assert fake_graph.captured_configs[-1]["callbacks"] == []
    assert handler_calls == 0
