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


class DuplicateTracksLLM:
    """Parses to two entries for the same track (differing only in casing and
    surrounding whitespace) plus one distinct track, to exercise I4's dedup
    in n_parse."""

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        if self.schema is SetRequest:
            return SetRequest(
                tracks=[
                    TrackRef(artist="a", title="t1"),
                    TrackRef(artist=" A ", title="T1"),
                    TrackRef(artist="b", title="t2"),
                ],
                energy_shape="build",
            )
        return _Explanations(explanations=["works"], summary="ok")


class ThreeTrackWithIslandLLM:
    """Parses to three tracks, one of which (island) is BPM-incompatible with
    the other two and so can never share an edge with them."""

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        if self.schema is SetRequest:
            return SetRequest(
                tracks=[
                    TrackRef(artist="a", title="t1"),
                    TrackRef(artist="b", title="t2"),
                    TrackRef(artist="c", title="island"),
                ],
                energy_shape="peak_end",
            )
        return _Explanations(explanations=["works"], summary="ok")


class DurationCappedLLM:
    """Parses to three compatible tracks plus a duration_min that only fits
    two of them."""

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        if self.schema is SetRequest:
            return SetRequest(
                tracks=[
                    TrackRef(artist="a", title="t1"),
                    TrackRef(artist="b", title="t2"),
                    TrackRef(artist="c", title="t3"),
                ],
                duration_min=5,
                energy_shape="build",
            )
        return _Explanations(explanations=["works", "works too"], summary="ok")


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
    gsb_route = respx.get(url__regex=r"api\.getsong\.co.*").respond(json={"search": []})

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
    # A GetSongBPM miss costs two calls per track: `both`, then the title-only fallback.
    assert gsb_route.call_count == 4
    # And that no unstubbed host was contacted: respx raises on any request
    # to a route it doesn't recognize when there's no catch-all, so reaching
    # this point with only the two routes above registered is itself proof.
    assert all(
        call.request.url.host in {"api.deezer.com", "api.getsong.co"}
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
    for name in (
        "LANGFUSE_SECRET_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "GETSONGBPM_API_KEY",
        "LASTFM_API_KEY",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
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


@pytest.mark.asyncio
async def test_duplicate_tracks_are_deduped_before_enrichment(monkeypatch):
    """Two copies of the same track (normalized on artist/title) must be
    deduped in n_parse before enrichment -- otherwise they'd score a perfect
    1.0 edge against each other and get placed back-to-back (I4)."""
    seen: dict = {}

    async def fake_enrich_all(refs, cache):
        seen["refs"] = list(refs)
        tracks = [Track(ref=r, bpm=120, camelot="8A", energy=0.5) for r in refs]
        return tracks, []

    monkeypatch.setattr("musicagent.graph.enrich_all", fake_enrich_all)

    await run_set("a t1, a t1 again, b t2", llm=DuplicateTracksLLM())

    assert len(seen["refs"]) == 2
    assert {(r.artist, r.title) for r in seen["refs"]} == {("a", "t1"), ("b", "t2")}


@pytest.mark.asyncio
async def test_resolved_but_unplaced_track_reported_as_omitted():
    """A track that enriches fine but has no compatible edge to any other
    track must show up in `omitted`, not silently vanish (I3)."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)
    cache.put(
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A", energy=0.3)
    )
    cache.put(
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A", energy=0.7)
    )
    cache.put(
        Track(
            ref=TrackRef(artist="c", title="island"),
            bpm=175,
            camelot="3B",
            energy=0.5,
        )
    )

    result = await run_set(
        "a t1, b t2, c island", cache=cache, llm=ThreeTrackWithIslandLLM()
    )

    assert result.unresolved == []
    assert [r.title for r in result.omitted] == ["island"]


class NonPositiveDurationLLM:
    """Parses to three compatible tracks with a non-positive duration_min,
    which must be treated as absent (no trimming)."""

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        if self.schema is SetRequest:
            return SetRequest(
                tracks=[
                    TrackRef(artist="a", title="t1"),
                    TrackRef(artist="b", title="t2"),
                    TrackRef(artist="c", title="t3"),
                ],
                duration_min=0,
                energy_shape="build",
            )
        return _Explanations(explanations=["works", "works too"], summary="ok")


@pytest.mark.asyncio
async def test_non_positive_duration_min_is_treated_as_absent_no_trimming():
    """A model-produced duration_min of 0 (or negative) must not trim the set
    down to the 2-track floor -- it should be treated the same as no
    duration_min at all."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)
    cache.put(
        Track(
            ref=TrackRef(artist="a", title="t1"),
            bpm=120,
            camelot="8A",
            energy=0.2,
            duration_s=200,
        )
    )
    cache.put(
        Track(
            ref=TrackRef(artist="b", title="t2"),
            bpm=120,
            camelot="8A",
            energy=0.5,
            duration_s=200,
        )
    )
    cache.put(
        Track(
            ref=TrackRef(artist="c", title="t3"),
            bpm=120,
            camelot="8A",
            energy=0.8,
            duration_s=200,
        )
    )

    result = await run_set(
        "a t1, b t2, c t3, no time limit",
        cache=cache,
        llm=NonPositiveDurationLLM(),
    )

    assert len(result.transitions) == 2
    assert result.omitted == []


@pytest.mark.asyncio
async def test_duration_min_trims_set_from_end_and_reports_omitted():
    """duration_min trims tracks from the end of the path once the summed
    duration exceeds the budget, keeping at least 2 tracks; trimmed tracks
    are reported as omitted, not silently dropped (I6)."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)
    cache.put(
        Track(
            ref=TrackRef(artist="a", title="t1"),
            bpm=120,
            camelot="8A",
            energy=0.2,
            duration_s=200,
        )
    )
    cache.put(
        Track(
            ref=TrackRef(artist="b", title="t2"),
            bpm=120,
            camelot="8A",
            energy=0.5,
            duration_s=200,
        )
    )
    cache.put(
        Track(
            ref=TrackRef(artist="c", title="t3"),
            bpm=120,
            camelot="8A",
            energy=0.8,
            duration_s=200,
        )
    )

    # 3 * 200s = 600s = 10min, well over the 5min budget.
    result = await run_set(
        "a t1, b t2, c t3, keep it around 5 minutes",
        cache=cache,
        llm=DurationCappedLLM(),
    )

    assert len(result.transitions) == 1
    assert [r.title for r in result.omitted] == ["t3"]
