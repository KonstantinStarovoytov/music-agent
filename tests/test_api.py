import json

import httpx
import pytest

from musicagent.api import create_app
from musicagent.db import TrackCache, get_engine, init_db
from musicagent.models import Track, TrackRef
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
    error_line = [line for line in body.splitlines() if line.startswith("data: ")][-1]
    assert "LLM" in error_line or "parse_input" in error_line


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
