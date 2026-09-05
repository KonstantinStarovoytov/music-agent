import asyncio
import time

import httpx
import pytest
import respx

import musicagent.enrichment as enrichment_mod
from musicagent.db import TrackCache, get_engine, init_db
from musicagent.enrichment import enrich_all, enrich_one
from musicagent.models import Track, TrackRef

# Search hit shape verified against the live api.deezer.com/search endpoint:
# a hit carries only album, artist, duration, explicit_*, id, isrc, link,
# md5_image, preview, rank, readable, title, title_short, title_version,
# type -- notably no bpm/gain. Trimmed here to the fields _deezer reads.
DEEZER_SEARCH = {"data": [{"id": 3135556, "duration": 240}]}
# Full track object shape verified against the live api.deezer.com/track/{id}
# endpoint, which does carry bpm/gain.
DEEZER_TRACK = {"id": 3135556, "bpm": 126.0, "gain": -8.0, "duration": 240}
GSB = {"search": [{"tempo": "126", "key_of": "Am"}]}

DEEZER_TRACK_ROUTE = r"api\.deezer\.com/track/\d+"


def mock_deezer_track(json=DEEZER_TRACK, status_code=200):
    """Register the /track/{id} route the two-step Deezer lookup makes after
    a successful /search. Most tests don't care about its call_count, so this
    is a thin helper rather than something every test wires up by hand."""
    return respx.get(url__regex=DEEZER_TRACK_ROUTE).respond(status_code, json=json)


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
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER_SEARCH)
    mock_deezer_track()
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
    respx.get(url__regex=r"musicbrainz\.org.*").respond(500)
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
        return httpx.Response(200, json=DEEZER_SEARCH)

    def gsb_side_effect(request):
        if "Nobody" in str(request.url):
            return httpx.Response(200, json={"search": []})
        return httpx.Response(200, json=GSB)

    respx.get(url__regex=r"api\.deezer\.com/search.*").mock(
        side_effect=deezer_side_effect
    )
    mock_deezer_track()
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
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER_SEARCH)
    mock_deezer_track()
    gsb_route = respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    respx.get(url__regex=r"musicbrainz\.org.*").respond(json={"recordings": []})
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert gsb_route.call_count == 0
    # Deezer alone never supplies a Camelot key, GSB is skipped, and
    # MusicBrainz finds no recordings -> unresolved, but critically: no
    # exception was raised.
    assert track is None


@pytest.mark.asyncio
@respx.mock
async def test_missing_lastfm_key_yields_empty_tags_without_crash(monkeypatch):
    """Missing LASTFM_API_KEY must not raise; tags simply come back empty."""
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER_SEARCH)
    mock_deezer_track()
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
        json=DEEZER_SEARCH
    )
    deezer_track_route = mock_deezer_track()
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
    assert deezer_track_route.call_count == 1
    assert gsb_route.call_count == 1
    assert lastfm_route.call_count == 1

    cached = cache.get(ref)
    assert cached is not None and cached.bpm == track1.bpm

    async with httpx.AsyncClient() as client:
        track2 = await enrich_one(ref, client, cache)
    assert track2 is not None and track2.bpm == track1.bpm
    # No new network calls happened on the cache hit.
    assert deezer_route.call_count == 1
    assert deezer_track_route.call_count == 1
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
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER_SEARCH)
    mock_deezer_track()
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
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER_SEARCH)
    mock_deezer_track()
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
    """Deezer supplies bpm+duration but no key (it never does); GetSongBPM supplies
    the key. The resulting source must not claim plain 'deezer' since Deezer never
    supplied the Camelot key that made the track resolvable."""
    gsb_key_only = {"search": [{"key_of": "Am"}]}
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER_SEARCH)
    mock_deezer_track()
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


# --- C1: Deezer two-step lookup (search -> /track/{id}) ---------------------


@pytest.mark.asyncio
@respx.mock
async def test_deezer_track_bpm_zero_falls_through_to_next_provider():
    """Deezer returns bpm=0 when the tempo is unknown (verified against the live
    api.deezer.com/track/{id} endpoint). That must be treated like a missing bpm
    and fall through to GetSongBPM, not taken as a real 0 BPM reading."""
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER_SEARCH)
    mock_deezer_track(json={"id": 3135556, "bpm": 0, "gain": -8.0, "duration": 240})
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
    assert track.source == "getsongbpm"
    # duration_s is independent of bpm and still comes from Deezer's track lookup.
    assert track.duration_s == 240


@pytest.mark.asyncio
@respx.mock
async def test_deezer_track_lookup_failure_falls_through_without_crash():
    """If /search succeeds but /track/{id} fails (network error, 500, etc), Deezer
    must contribute nothing and the cascade must fall through to GetSongBPM."""
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER_SEARCH)
    deezer_track_route = mock_deezer_track(status_code=500)
    respx.get(url__regex=r"api\.getsongbpm\.com.*").respond(json=GSB)
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": []}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert deezer_track_route.call_count == 3  # retried like any other provider call
    assert track is not None
    assert track.bpm == 126.0
    assert track.camelot == "8A"
    assert track.source == "getsongbpm"
    assert track.duration_s is None


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


# --- MusicBrainz/AcousticBrainz fallback ------------------------------------

MB_ONE_RECORDING = {"recordings": [{"id": "mbid-1"}]}


def _ab_doc(**overrides):
    doc = {
        "tonal": {"key_key": "A#", "key_scale": "minor", "key_strength": 0.8},
        "rhythm": {"bpm": 128.0},
        "lowlevel": {"average_loudness": 0.7},
    }
    doc.update(overrides)
    return doc


@pytest.mark.asyncio
@respx.mock
async def test_acousticbrainz_supplies_key_deezer_supplies_bpm(monkeypatch):
    """Deezer gives bpm but no key, GetSongBPM is unavailable (no key configured),
    and MusicBrainz/AcousticBrainz supplies the missing camelot key."""
    monkeypatch.delenv("GETSONGBPM_API_KEY", raising=False)
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json=DEEZER_SEARCH)
    mock_deezer_track()
    respx.get(url__regex=r"musicbrainz\.org.*").respond(json=MB_ONE_RECORDING)
    respx.get(url__regex=r"acousticbrainz\.org.*").respond(
        json={"mbid-1": {"0": _ab_doc()}}
    )
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": []}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track is not None
    assert track.camelot == "3A"  # A# minor
    assert track.bpm == 126.0  # from Deezer, not overwritten by AcousticBrainz
    assert "deezer" in track.source and "acousticbrainz" in track.source


@pytest.mark.asyncio
@respx.mock
async def test_acousticbrainz_batch_lookup_hits_third_mbid_in_one_request(
    monkeypatch,
):
    """MusicBrainz returns 3 MBIDs; only the third has AcousticBrainz data. The
    track must still resolve, and only ONE AcousticBrainz request (the batch)
    must be made -- not one per MBID."""
    monkeypatch.delenv("GETSONGBPM_API_KEY", raising=False)
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json={"data": []})
    respx.get(url__regex=r"musicbrainz\.org.*").respond(
        json={"recordings": [{"id": "mbid-a"}, {"id": "mbid-b"}, {"id": "mbid-c"}]}
    )
    ab_route = respx.get(url__regex=r"acousticbrainz\.org.*").respond(
        json={"mbid-c": {"0": _ab_doc()}}
    )
    respx.get(url__regex=r"ws\.audioscrobbler\.com.*").respond(
        json={"toptags": {"tag": []}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track is not None
    assert track.camelot == "3A"
    assert track.bpm == 128.0
    assert ab_route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_musicbrainz_no_recordings_skips_acousticbrainz_call(monkeypatch):
    """When MusicBrainz returns no recordings, AcousticBrainz must not be called
    at all, and the track ends up unresolved."""
    monkeypatch.delenv("GETSONGBPM_API_KEY", raising=False)
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json={"data": []})
    respx.get(url__regex=r"musicbrainz\.org.*").respond(json={"recordings": []})
    ab_route = respx.get(url__regex=r"acousticbrainz\.org.*").respond(json={})
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track is None
    assert ab_route.call_count == 0


@pytest.mark.asyncio
@respx.mock
async def test_acousticbrainz_unparseable_key_falls_back_to_unresolved(monkeypatch):
    """A document whose tonal.key_key can't be parsed as a musical key must not
    raise; the track simply stays unresolved (no other provider supplies bpm+key
    here)."""
    monkeypatch.delenv("GETSONGBPM_API_KEY", raising=False)
    respx.get(url__regex=r"api\.deezer\.com/search.*").respond(json={"data": []})
    respx.get(url__regex=r"musicbrainz\.org.*").respond(json=MB_ONE_RECORDING)
    respx.get(url__regex=r"acousticbrainz\.org.*").respond(
        json={"mbid-1": {"0": _ab_doc(tonal={"key_key": "H#", "key_scale": "minor"})}}
    )
    async with httpx.AsyncClient() as client:
        track = await enrich_one(
            TrackRef(artist="Bicep", title="Glue"), client, cache=None
        )
    assert track is None


@pytest.mark.asyncio
@respx.mock
async def test_musicbrainz_throttle_enforces_minimum_interval(monkeypatch):
    """Two concurrent enrichments must not fire their MusicBrainz requests less
    than MUSICBRAINZ_MIN_INTERVAL_S apart."""
    monkeypatch.setattr(enrichment_mod, "MUSICBRAINZ_MIN_INTERVAL_S", 0.2)
    monkeypatch.setattr(enrichment_mod, "_musicbrainz_last_request_at", 0.0)
    # Restore real sleep for just this test: the autouse fast_retries fixture
    # patches it to a no-op, which would defeat the throttle being tested.
    monkeypatch.setattr("musicagent.enrichment.asyncio.sleep", _REAL_SLEEP)

    timestamps: list[float] = []

    def mb_side_effect(request):
        timestamps.append(time.monotonic())
        return httpx.Response(200, json={"recordings": []})

    respx.get(url__regex=r"musicbrainz\.org.*").mock(side_effect=mb_side_effect)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            enrichment_mod._musicbrainz_search(client, TrackRef(artist="A", title="B")),
            enrichment_mod._musicbrainz_search(client, TrackRef(artist="C", title="D")),
        )

    assert len(timestamps) == 2
    assert timestamps[1] - timestamps[0] >= 0.2 - 0.02
