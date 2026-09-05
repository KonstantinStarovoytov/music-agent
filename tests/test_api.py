import asyncio
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
    # The truncation notice (None here, since this request is under the cap)
    # is persisted with the saved set and replayed on GET.
    assert payload["request"]["notice"] is None


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


@pytest.mark.asyncio
async def test_generic_exception_mid_run_streams_error_event_not_500(monkeypatch):
    """A plain (non-LLMOutputError) exception raised mid-run -- e.g. a bug or
    network failure in a non-LLM node -- must still produce a graceful `error`
    SSE event, never an aborted stream or a leaked internal exception message.
    """

    async def boom_enrich_all(refs, cache):
        raise RuntimeError("boom: unexpected failure detail")

    monkeypatch.setattr("musicagent.graph.enrich_all", boom_enrich_all)

    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    app = create_app(engine=engine, llm=FakeLLM())
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/sets", json={"text": "a t1, b t2"}) as r,
    ):
        assert r.status_code == 200
        body = "".join([chunk async for chunk in r.aiter_text()])

    from musicagent.api import GENERIC_ERROR_MESSAGE

    assert "event: error" in body
    assert GENERIC_ERROR_MESSAGE in body
    assert "event: result" not in body
    assert "boom" not in body
    assert "unexpected failure detail" not in body


_ENV_VARS_USED_BY_APP = [
    "SITE_ORIGIN",
    "DATABASE_URL",
    "LANGFUSE_SECRET_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "GETSONGBPM_API_KEY",
    "LASTFM_API_KEY",
]


def test_create_app_requires_no_env_vars(monkeypatch):
    for name in _ENV_VARS_USED_BY_APP:
        monkeypatch.delenv(name, raising=False)
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


@pytest.mark.asyncio
async def test_cors_closed_by_default_without_site_origin(monkeypatch):
    """CORS must fail closed (no cross-origin access allowed) rather than open
    to '*' when SITE_ORIGIN is unset -- this is a public unauthenticated API."""
    monkeypatch.delenv("SITE_ORIGIN", raising=False)
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    app = create_app(engine=engine, llm=FakeLLM())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(
            "/health",
            headers={"Origin": "https://evil.example"},
        )
    assert "access-control-allow-origin" not in resp.headers


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
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        async with client.stream("POST", "/sets", json={"text": "many tracks"}) as r:
            body = "".join([chunk async for chunk in r.aiter_text()])

        assert len(calls["refs"]) == 30
        final = [line for line in body.splitlines() if line.startswith("data: {")][-1]
        payload = json.loads(final.removeprefix("data: "))
        assert payload.get("notice")

        # The notice must be persisted with the saved set, not just streamed
        # once, so GET /sets/{id} replays it too.
        got = await client.get(f"/sets/{payload['set_id']}")
    assert got.json()["request"]["notice"] == payload["notice"]


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


## test_client_disconnect_mid_stream_does_not_raise was removed here: it had
## no assertion, and because httpx.ASGITransport fully buffers the ASGI
## response before the client reads it, breaking out of `aiter_text()` early
## never actually disconnects anything server-side -- the generator has
## already run to completion by the time the test's `break` executes. It
## tested nothing. A genuine disconnect test would need a transport that
## streams incrementally (e.g. a real uvicorn server + a raw socket client),
## which is a bigger lift than this ticket's scope; not attempted here.


@pytest.mark.asyncio
async def test_rate_limit_returns_429_after_max_requests_per_minute():
    """POST /sets is rate limited per client host; the request past
    RATE_LIMIT_MAX_REQUESTS within the window must be rejected with 429
    before any graph work runs."""
    from musicagent.api import RATE_LIMIT_MAX_REQUESTS

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
        for _ in range(RATE_LIMIT_MAX_REQUESTS):
            resp = await client.post("/sets", json={"text": "a t1, b t2"})
            assert resp.status_code == 200
        resp = await client.post("/sets", json={"text": "a t1, b t2"})
    assert resp.status_code == 429


@pytest.mark.asyncio
async def test_overall_timeout_streams_error_event_not_hang(monkeypatch):
    """If the whole graph run exceeds OVERALL_DEADLINE_S, the stream must end
    with a generic error event rather than hang or crash silently."""
    monkeypatch.setattr("musicagent.api.OVERALL_DEADLINE_S", 0.01)

    async def slow_enrich_all(refs, cache):
        await asyncio.sleep(1)
        return [], []

    monkeypatch.setattr("musicagent.graph.enrich_all", slow_enrich_all)

    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    app = create_app(engine=engine, llm=FakeLLM())
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("POST", "/sets", json={"text": "a t1, b t2"}) as r,
    ):
        body = "".join([chunk async for chunk in r.aiter_text()])

    from musicagent.api import GENERIC_ERROR_MESSAGE

    assert "event: error" in body
    assert GENERIC_ERROR_MESSAGE in body
    assert "event: result" not in body


@pytest.mark.asyncio
async def test_store_save_failure_streams_error_event_not_crash(monkeypatch):
    """A DB blip on store.save() (result serialization + persistence, which
    used to sit outside the SSE error handling) must still produce a
    terminal error event instead of silently killing the stream."""

    def boom_save(self, request_json, result):
        raise RuntimeError("db blip: connection reset")

    monkeypatch.setattr("musicagent.db.SetStore.save", boom_save)

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
        body = "".join([chunk async for chunk in r.aiter_text()])

    from musicagent.api import GENERIC_ERROR_MESSAGE

    assert "event: error" in body
    assert GENERIC_ERROR_MESSAGE in body
    assert "event: result" not in body
    assert "db blip" not in body
