from musicagent.db import SetStore, TrackCache, get_engine, init_db
from musicagent.models import SetResult, Track, TrackRef


def make_engine():
    e = get_engine("sqlite:///:memory:")
    init_db(e)
    return e


def test_track_cache_roundtrip_and_case_insensitive():
    """Test from brief: basic roundtrip with case insensitivity."""
    cache = TrackCache(make_engine())
    ref = TrackRef(artist="Bicep", title="Glue")
    assert cache.get(ref) is None
    cache.put(Track(ref=ref, bpm=120, camelot="8A", energy=0.6, source="deezer"))
    hit = cache.get(TrackRef(artist="bicep", title="GLUE"))
    assert hit and hit.bpm == 120 and hit.source == "deezer"


def test_set_store_roundtrip():
    """Test from brief: basic set store roundtrip."""
    store = SetStore(make_engine())
    result = SetResult(transitions=[], summary="empty", unresolved=[])
    set_id = store.save({"tracks": []}, result)
    loaded = store.load(set_id)
    assert loaded and loaded["result"]["summary"] == "empty"


def test_track_cache_upsert():
    """Verify TrackCache.put is a genuine upsert: two puts for same artist/title leaves exactly one row."""
    cache = TrackCache(make_engine())
    ref = TrackRef(artist="Bicep", title="Glue")

    # First put
    track1 = Track(ref=ref, bpm=120, camelot="8A", energy=0.6, source="deezer")
    cache.put(track1)
    hit1 = cache.get(ref)
    assert hit1 and hit1.bpm == 120

    # Second put with same artist/title but different values
    track2 = Track(ref=ref, bpm=130, camelot="8B", energy=0.7, source="spotify")
    cache.put(track2)
    hit2 = cache.get(ref)

    # Should have exactly one row with the latest values
    assert hit2 and hit2.bpm == 130
    assert hit2.camelot == "8B"
    assert hit2.energy == 0.7
    assert hit2.source == "spotify"


def test_track_cache_whitespace_insensitive():
    """Verify cache lookups are whitespace-insensitive."""
    cache = TrackCache(make_engine())
    ref = TrackRef(artist="  Bicep  ", title="  Glue  ")
    cache.put(Track(ref=ref, bpm=120, camelot="8A", energy=0.6, source="deezer"))

    # Try with different whitespace
    hit = cache.get(TrackRef(artist="bicep", title="glue"))
    assert hit and hit.bpm == 120

    # Try with extra spaces
    hit2 = cache.get(TrackRef(artist="  BICEP  ", title="  GLUE  "))
    assert hit2 and hit2.bpm == 120


def test_set_store_load_unknown_id():
    """Verify SetStore.load returns None for unknown id rather than raising."""
    store = SetStore(make_engine())
    result = store.load("nonexistent-id")
    assert result is None


def test_track_cache_empty_tags_and_none_duration():
    """Verify Track with empty tags list and None duration_s round-trips unchanged."""
    cache = TrackCache(make_engine())
    ref = TrackRef(artist="Test Artist", title="Test Track")

    # Create track with empty tags and None duration
    original = Track(ref=ref, bpm=120, camelot="8A", energy=0.5, duration_s=None, tags=[], source="test")
    cache.put(original)

    # Retrieve and verify fields match exactly
    retrieved = cache.get(ref)
    assert retrieved is not None
    assert retrieved.ref.artist == original.ref.artist
    assert retrieved.ref.title == original.ref.title
    assert retrieved.bpm == original.bpm
    assert retrieved.camelot == original.camelot
    assert retrieved.energy == original.energy
    assert retrieved.duration_s is None  # Should be None, not some default
    assert retrieved.tags == []  # Should be empty list, not None
    assert retrieved.source == original.source
