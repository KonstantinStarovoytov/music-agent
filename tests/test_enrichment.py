import asyncio

import httpx
import pytest
import respx

import musicagent.enrichment as enrichment_mod
from musicagent.db import TrackCache, get_engine, init_db
from musicagent.enrichment import enrich_all, enrich_one
from musicagent.models import Track, TrackRef

DEEZER = {"data": [{"bpm": 126.0, "gain": -8.0, "duration": 240}]}
GSB = {"search": [{"tempo": "126", "key_of": "Am"}]}

# Captured at import time, before any test/fixture monkeypatches asyncio.sleep
# (the autouse fast_retries fixture replaces it with a no-op) -- needed by the
# concurrency-bound test below, which requires a *real* sleep to make
# concurrently-scheduled tasks actually overlap in time.
_REAL_SLEEP = asyncio.sleep


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
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": [{"name": "electronic"}]}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track and track.bpm == 126.0 and track.camelot == "8A"
    assert "electronic" in track.tags


@pytest.mark.asyncio
@respx.mock
async def test_unresolvable_goes_to_unresolved():
    respx.get(url__regex=r".*").respond(json={"data": [], "search": None})
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    resolved, unresolved = await enrich_all(
        [TrackRef(artist="x", title="y")], TrackCache(engine)
    )
    assert resolved == [] and len(unresolved) == 1


@pytest.mark.asyncio
async def test_cache_hit_skips_network():
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    cache = TrackCache(engine)
    ref = TrackRef(artist="a", title="b")
    cache.put(Track(ref=ref, bpm=120, camelot="8A", source="deezer"))
    resolved, unresolved = await enrich_all(
        [ref], cache
    )  # no respx: network would raise
    assert resolved[0].bpm == 120 and unresolved == []


@pytest.mark.asyncio
@respx.mock
async def test_provider_500_retries_then_falls_through():
    """A provider returning HTTP 500 must retry (RETRIES=2 -> 3 attempts total) then fall
    through to the next provider rather than crashing enrichment."""
    deezer_route = respx.get(url__regex=r"api\.deezer\.com/search.*").respond(500)
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": []}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert deezer_route.call_count == 3
    assert track is not None
    assert track.bpm == 126.0
    assert track.camelot == "8A"
    assert track.source == "getsongbpm"


@pytest.mark.asyncio
@respx.mock
async def test_provider_malformed_json_retries_then_falls_through():
    """Malformed (non-JSON) response body must also retry then fall through, not crash."""
    deezer_route = respx.get(url__regex=r"api\.deezer\.com/search.*").respond(
        200, content="not json"
    )
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": []}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert deezer_route.call_count == 3
    assert track is not None and track.bpm == 126.0


@pytest.mark.asyncio
@respx.mock
async def test_all_providers_fail_marks_unresolved_not_crash():
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(500)
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(500)
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
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

    respx.get(url__regex=r"api\.deezer\.com/search.*").mock(
        side_effect=deezer_side_effect
    )
    respx.get(url__regex=r"api\.getsongbpm\.com.*").mock(side_effect=gsb_side_effect)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": []}}
    )

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
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
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
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track is not None
    assert track.tags == []
    assert lastfm_route.call_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_resolved_track_written_to_cache_and_second_call_skips_network():
    """Every resolved enrichment is written to the cache exactly once, and a subsequent
    call for the same track makes no HTTP request at all."""
    deezer_route = respx.get(url__regex=r"api\.deezer\.com/search.*").respond(
        json=DEEZER
    )
    gsb_route = respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    lastfm_route = respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": []}}
    )
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


# --- Finding 1: wrong-shape JSON must never crash the batch -----------------


@pytest.mark.asyncio
@respx.mock
async def test_provider_returns_json_list_falls_through_without_crash():
    """A provider returning a valid JSON list (not an object) must be treated like a
    failed response, not crash with AttributeError."""
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(200, json=[1, 2, 3])
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": []}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track is not None
    assert track.bpm == 126.0
    assert track.camelot == "8A"


@pytest.mark.asyncio
@respx.mock
async def test_provider_returns_json_string_falls_through_without_crash():
    """A provider returning a bare JSON string must also be treated as a failed
    response rather than crashing when callers do `.get(...)` on it."""
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(200, json="oops")
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": []}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track is not None
    assert track.bpm == 126.0
    assert track.camelot == "8A"


@pytest.mark.asyncio
@respx.mock
async def test_lastfm_single_tag_returned_as_bare_dict():
    """Last.fm's real API quirk: a single tag comes back as a dict, not a list."""
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER)
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": {"name": "electronic"}}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track is not None
    assert track.tags == ["electronic"]


@pytest.mark.asyncio
@respx.mock
async def test_lastfm_items_missing_name_are_skipped():
    """Tag items lacking a usable 'name' must be skipped, not raise KeyError."""
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER)
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": [{"count": 5}, {"name": "electronic"}, "not-a-dict"]}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track is not None
    assert track.tags == ["electronic"]


@pytest.mark.asyncio
async def test_enrich_all_survives_internal_exception_in_one_track(monkeypatch):
    """If one track's enrichment raises an unexpected exception, the other tracks in
    the batch must still resolve and the batch call must not raise."""
    good_ref = TrackRef(artist="Bicep", title="Glue")
    bad_ref = TrackRef(artist="Boom", title="Crash")
    other_ref = TrackRef(artist="Four Tet", title="Baby")

    good_track = Track(ref=good_ref, bpm=120, camelot="8A", source="deezer")
    other_track = Track(ref=other_ref, bpm=125, camelot="9A", source="deezer")

    async def fake_inner(ref, client, cache):
        if ref == bad_ref:
            raise RuntimeError("boom: unexpected bug in provider path")
        if ref == good_ref:
            return good_track
        return other_track

    monkeypatch.setattr(enrichment_mod, "_enrich_one_inner", fake_inner)

    resolved, unresolved = await enrich_all([good_ref, bad_ref, other_ref], cache=None)
    assert unresolved == [bad_ref]
    resolved_refs = [t.ref for t in resolved]
    assert good_ref in resolved_refs and other_ref in resolved_refs
    assert len(resolved) == 2


# --- Finding 2: source must reflect true provenance of bpm/camelot ----------


@pytest.mark.asyncio
@respx.mock
async def test_source_reflects_mixed_provenance_not_plain_deezer():
    """Deezer supplies bpm+duration but no key; GetSongBPM supplies the key. The
    resulting source must not claim plain 'deezer' since Deezer never supplied the
    Camelot key that made the track resolvable."""
    deezer_no_key = {"data": [{"bpm": 126.0, "gain": -8.0, "duration": 240}]}
    gsb_key_only = {"search": [{"key_of": "Am"}]}
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=deezer_no_key)
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=gsb_key_only)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": []}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track is not None
    assert track.bpm == 126.0
    assert track.camelot == "8A"
    assert track.duration_s == 240
    assert track.source != "deezer"
    assert "deezer" in track.source and "getsongbpm" in track.source


# --- Finding 3: overall per-track deadline -----------------------------------


@pytest.mark.asyncio
async def test_hanging_track_times_out_and_is_reported_unresolved(monkeypatch):
    """A track whose provider lookups hang must be returned as unresolved once the
    overall per-track deadline elapses, without raising or failing the batch."""
    monkeypatch.setattr(enrichment_mod, "ENRICH_DEADLINE_S", 0.05)

    async def hang_forever(ref, client, cache):
        # Note: asyncio.sleep is patched to a no-op by the autouse fast_retries
        # fixture (it patches the real asyncio module, shared by this test), so
        # an Event that never gets set is used instead to simulate a genuine hang.
        await asyncio.Event().wait()
        raise AssertionError("should have been cancelled by the deadline")

    monkeypatch.setattr(enrichment_mod, "_enrich_one_inner", hang_forever)

    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track is None

    resolved, unresolved = await enrich_all(
        [TrackRef(artist="Bicep", title="Glue")], cache=None
    )
    assert resolved == []
    assert len(unresolved) == 1


# --- Finding 2c: bounded enrichment fan-out -----------------------------------


@pytest.mark.asyncio
async def test_enrich_all_bounds_concurrency(monkeypatch):
    """enrich_all must never have more than ENRICH_CONCURRENCY tracks' worth of
    per-track work in flight at once, even when given a much larger batch."""
    current = 0
    max_seen = 0
    lock = asyncio.Lock()

    async def fake_enrich_one(ref, client, cache):
        nonlocal current, max_seen
        async with lock:
            current += 1
            max_seen = max(max_seen, current)
        await _REAL_SLEEP(0.02)
        async with lock:
            current -= 1
        return Track(ref=ref, bpm=120, camelot="8A")

    monkeypatch.setattr(enrichment_mod, "enrich_one", fake_enrich_one)

    refs = [TrackRef(artist=f"artist{i}", title=f"track{i}") for i in range(20)]
    resolved, unresolved = await enrich_all(refs, cache=None)

    assert len(resolved) == 20
    assert unresolved == []
    assert max_seen <= enrichment_mod.ENRICH_CONCURRENCY
    assert max_seen == enrichment_mod.ENRICH_CONCURRENCY  # actually saturates the bound
