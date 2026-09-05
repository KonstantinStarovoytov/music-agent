import json

import httpx
import pytest

from musicagent.api import create_app
from musicagent.db import TrackCache, get_engine, init_db
from musicagent.llm import _Explanations
from musicagent.models import SetRequest, Track, TrackRef
from tests.test_graph import FakeLLM


@pytest.mark.asyncio
async def test_health_and_post_sets_stream():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)
    cache.put(
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A", energy=0.3)
    )
    cache.put(
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A", energy=0.7)
    )
    app = create_app(engine=engine, llm=FakeLLM())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/health")).status_code == 200
        async with client.stream("POST", "/sets", json={"text": "a t1, b t2"}) as r:
            body = "".join([chunk async for chunk in r.aiter_text()])
    assert "event: progress" in body and "event: result" in body
    final = [line for line in body.splitlines() if line.startswith("data: {")][-1]
    payload = json.loads(final.removeprefix("data: "))
    assert payload["result"]["summary"] == "ok" and payload["set_id"]


@pytest.mark.asyncio
async def test_get_set_roundtrip():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)
    cache.put(
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A", energy=0.3)
    )
    cache.put(
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A", energy=0.7)
    )
    app = create_app(engine=engine, llm=FakeLLM())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/sets/nope")).status_code == 404

        async with client.stream("POST", "/sets", json={"text": "a t1, b t2"}) as r:
            body = "".join([chunk async for chunk in r.aiter_text()])
        final = [line for line in body.splitlines() if line.startswith("data: {")][-1]
        set_id = json.loads(final.removeprefix("data: "))["set_id"]

        got = await client.get(f"/sets/{set_id}")
    assert got.status_code == 200
    payload = got.json()
    assert payload["set_id"] == set_id
    assert payload["result"]["summary"] == "ok"


class FailingLLM:
    """Always raises when invoked, so parse_input surfaces LLMOutputError
    (after the one repair retry inside musicagent.llm)."""

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_llm_output_error_streams_error_event_not_500():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    app = create_app(engine=engine, llm=FailingLLM())
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/sets", json={"text": "a t1, b t2"}) as r,
    ):
        assert r.status_code == 200
        body = "".join([chunk async for chunk in r.aiter_text()])
    assert "event: error" in body
    assert "event: result" not in body


@pytest.mark.asyncio
async def test_llm_output_error_hides_internal_exception_text():
    """The SSE error payload must carry only the fixed generic message -- never
    the underlying provider/client exception text (e.g. 'boom', 'RuntimeError'),
    which could leak internal error details to an anonymous caller."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    app = create_app(engine=engine, llm=FailingLLM())
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/sets", json={"text": "a t1, b t2"}) as r,
    ):
        body = "".join([chunk async for chunk in r.aiter_text()])
    from musicagent.api import LLM_ERROR_MESSAGE

    assert LLM_ERROR_MESSAGE in body
    assert "boom" not in body
    assert "RuntimeError" not in body


def test_create_app_requires_no_env_vars(monkeypatch):
    monkeypatch.setattr("os.environ", {}, raising=False)
    engine = get_engine("sqlite:///:memory:")
    app = create_app(engine=engine, llm=FakeLLM())
    assert app is not None


@pytest.mark.asyncio
async def test_cors_allows_configured_site_origin(monkeypatch):
    monkeypatch.setenv("SITE_ORIGIN", "https://example.com")
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    app = create_app(engine=engine, llm=FakeLLM())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health", headers={"Origin": "https://example.com"})
    assert resp.headers["access-control-allow-origin"] == "https://example.com"


class BoomIfInvokedLLM:
    """Fails the test if the LLM is ever invoked -- used to prove an oversized
    request body is rejected before any LLM work starts."""

    def with_structured_output(self, schema):
        raise AssertionError("LLM must not be invoked for an oversized request body")


@pytest.mark.asyncio
async def test_oversized_text_returns_422_without_invoking_llm():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    app = create_app(engine=engine, llm=BoomIfInvokedLLM())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/sets", json={"text": "x" * 4001})
    assert resp.status_code == 422


class ManyTracksLLM:
    """Parses to 40 tracks, so the API's MAX_TRACKS cap must truncate before
    enrichment fans out."""

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def invoke(self, prompt):
        if self.schema is SetRequest:
            return SetRequest(
                tracks=[
                    TrackRef(artist=f"artist{i}", title=f"track{i}") for i in range(40)
                ],
                energy_shape="build",
            )
        return _Explanations(explanations=["ok"], summary="ok")


@pytest.mark.asyncio
async def test_track_list_over_cap_is_truncated_with_notice(monkeypatch):
    calls: dict = {}

    async def fake_enrich_all(refs, cache):
        calls["refs"] = list(refs)
        tracks = [Track(ref=r, bpm=120, camelot="8A", energy=0.5) for r in refs[:2]]
        return tracks, [r for r in refs[2:]]

    monkeypatch.setattr("musicagent.graph.enrich_all", fake_enrich_all)

    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    app = create_app(engine=engine, llm=ManyTracksLLM())
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/sets", json={"text": "many tracks"}) as r,
    ):
        body = "".join([chunk async for chunk in r.aiter_text()])

    assert len(calls["refs"]) == 30
    final = [line for line in body.splitlines() if line.startswith("data: {")][-1]
    payload = json.loads(final.removeprefix("data: "))
    assert payload.get("notice")


@pytest.mark.asyncio
async def test_track_list_under_cap_has_no_notice(monkeypatch):
    async def fake_enrich_all(refs, cache):
        tracks = [Track(ref=r, bpm=120, camelot="8A", energy=0.5) for r in refs]
        return tracks, []

    monkeypatch.setattr("musicagent.graph.enrich_all", fake_enrich_all)

    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    app = create_app(engine=engine, llm=FakeLLM())
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/sets", json={"text": "a t1, b t2"}) as r,
    ):
        body = "".join([chunk async for chunk in r.aiter_text()])

    final = [line for line in body.splitlines() if line.startswith("data: {")][-1]
    payload = json.loads(final.removeprefix("data: "))
    assert "notice" not in payload


@pytest.mark.asyncio
async def test_client_disconnect_mid_stream_does_not_raise():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)
    cache.put(
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A", energy=0.3)
    )
    cache.put(
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A", energy=0.7)
    )
    app = create_app(engine=engine, llm=FakeLLM())
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/sets", json={"text": "a t1, b t2"}) as r,
    ):
        async for _chunk in r.aiter_text():
            break  # simulate the client disconnecting after the first event
    # Reaching here without an exception propagating out of the ASGI app is the
    # assertion: an early client disconnect must not surface as a server error.
