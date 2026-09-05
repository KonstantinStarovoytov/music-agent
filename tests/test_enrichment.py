import httpx
import pytest
import respx

from musicagent.db import TrackCache, get_engine, init_db
from musicagent.enrichment import enrich_all, enrich_one
from musicagent.models import Track, TrackRef

DEEZER = {"data": [{"bpm": 126.0, "gain": -8.0, "duration": 240}]}
GSB = {"search": [{"tempo": "126", "key_of": "Am"}]}


@pytest.fixture(autouse=True)
def api_keys(monkeypatch):
    """Default: both provider keys present. Individual tests can delenv to test degradation."""
    monkeypatch.setenv("GETSONGBPM_API_KEY", "test-key")
    monkeypatch.setenv("LASTFM_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Skip real backoff sleeps so retry tests stay fast."""
    async def _no_sleep(_):
        return None

    monkeypatch.setattr("musicagent.enrichment.asyncio.sleep", _no_sleep)


@pytest.mark.asyncio
@respx.mock
async def test_enrich_one_deezer_bpm_gsb_key():
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER)
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(json={"toptags": {"tag": [{"name": "electronic"}]}})
    async with httpx.AsyncClient() as client:
        track = await enrich_one(TrackRef(artist="Bicep", title="Glue"), client, cache=None)
    assert track and track.bpm == 126.0 and track.camelot == "8A"
    assert "electronic" in track.tags


@pytest.mark.asyncio
@respx.mock
async def test_unresolvable_goes_to_unresolved():
    respx.get(url__regex=r".*").respond(json={"data": [], "search": None})
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    resolved, unresolved = await enrich_all([TrackRef(artist="x", title="y")], TrackCache(engine))
    assert resolved == [] and len(unresolved) == 1


@pytest.mark.asyncio
async def test_cache_hit_skips_network():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)
    ref = TrackRef(artist="a", title="b")
    cache.put(Track(ref=ref, bpm=120, camelot="8A", source="deezer"))
    resolved, unresolved = await enrich_all([ref], cache)  # no respx: network would raise
    assert resolved[0].bpm == 120 and unresolved == []


@pytest.mark.asyncio
@respx.mock
async def test_provider_500_retries_then_falls_through():
    """A provider returning HTTP 500 must retry (RETRIES=2 -> 3 attempts total) then fall
    through to the next provider rather than crashing enrichment."""
    deezer_route = respx.get(url__regex=r"api\.deezer\.com/search.*").respond(500)
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(json={"toptags": {"tag": []}})
    async with httpx.AsyncClient() as client:
        track = await enrich_one(TrackRef(artist="Bicep", title="Glue"), client, cache=None)
    assert deezer_route.call_count == 3
    assert track is not None
    assert track.bpm == 126.0
    assert track.camelot == "8A"
    assert track.source == "getsongbpm"


@pytest.mark.asyncio
@respx.mock
async def test_provider_malformed_json_retries_then_falls_through():
    """Malformed (non-JSON) response body must also retry then fall through, not crash."""
    deezer_route = respx.get(url__regex=r"api\.deezer\.com/search.*").respond(200, content="not json")
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(json={"toptags": {"tag": []}})
    async with httpx.AsyncClient() as client:
        track = await enrich_one(TrackRef(artist="Bicep", title="Glue"), client, cache=None)
    assert deezer_route.call_count == 3
    assert track is not None and track.bpm == 126.0


@pytest.mark.asyncio
@respx.mock
async def test_all_providers_fail_marks_unresolved_not_crash():
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(500)
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(500)
    async with httpx.AsyncClient() as client:
        track = await enrich_one(TrackRef(artist="Bicep", title="Glue"), client, cache=None)
    assert track is None


@pytest.mark.asyncio
@respx.mock
async def test_enrich_all_partial_failure_does_not_fail_batch():
    """One unresolvable track in a batch must not prevent the others from resolving."""
    good_ref = TrackRef(artist="Bicep", title="Glue")
    bad_ref = TrackRef(artist="Nobody", title="Nothing")

    def deezer_side_effect(request):
        if "Nobody" in str(request.url):
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json=DEEZER)

    def gsb_side_effect(request):
        if "Nobody" in str(request.url):
            return httpx.Response(200, json={"search": []})
        return httpx.Response(200, json=GSB)

    respx.get(url__regex=r"api\.deezer\.com/search.*").mock(side_effect=deezer_side_effect)
    respx.get(url__regex=r"api\.getsongbpm\.com.*").mock(side_effect=gsb_side_effect)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(json={"toptags": {"tag": []}})

    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    resolved, unresolved = await enrich_all([good_ref, bad_ref], TrackCache(engine))
    assert len(resolved) == 1 and resolved[0].bpm == 126.0
    assert unresolved == [bad_ref]


@pytest.mark.asyncio
@respx.mock
async def test_missing_getsongbpm_key_skips_provider_without_crash(monkeypatch):
    """Missing GETSONGBPM_API_KEY must not raise; the provider is skipped (no network call)."""
    monkeypatch.delenv("GETSONGBPM_API_KEY", raising=False)
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER)
    gsb_route = respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    async with httpx.AsyncClient() as client:
        track = await enrich_one(TrackRef(artist="Bicep", title="Glue"), client, cache=None)
    assert gsb_route.call_count == 0
    # Deezer alone never supplies a Camelot key, and GSB is skipped -> unresolved,
    # but critically: no exception was raised.
    assert track is None


@pytest.mark.asyncio
@respx.mock
async def test_missing_lastfm_key_yields_empty_tags_without_crash(monkeypatch):
    """Missing LASTFM_API_KEY must not raise; tags simply come back empty."""
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER)
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    lastfm_route = respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": [{"name": "electronic"}]}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(TrackRef(artist="Bicep", title="Glue"), client, cache=None)
    assert track is not None
    assert track.tags == []
    assert lastfm_route.call_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_resolved_track_written_to_cache_and_second_call_skips_network():
    """Every resolved enrichment is written to the cache exactly once, and a subsequent
    call for the same track makes no HTTP request at all."""
    deezer_route = respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER)
    gsb_route = respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    lastfm_route = respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(json={"toptags": {"tag": []}})
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)
    ref = TrackRef(artist="Bicep", title="Glue")

    async with httpx.AsyncClient() as client:
        track1 = await enrich_one(ref, client, cache)
    assert track1 is not None
    assert deezer_route.call_count == 1
    assert gsb_route.call_count == 1
    assert lastfm_route.call_count == 1

    cached = cache.get(ref)
    assert cached is not None and cached.bpm == track1.bpm

    async with httpx.AsyncClient() as client:
        track2 = await enrich_one(ref, client, cache)
    assert track2 is not None and track2.bpm == track1.bpm
    # No new network calls happened on the cache hit.
    assert deezer_route.call_count == 1
    assert gsb_route.call_count == 1
    assert lastfm_route.call_count == 1
