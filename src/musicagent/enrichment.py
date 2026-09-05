import asyncio
import os
import time

import httpx

from musicagent.core.camelot import parse_camelot
from musicagent.db import TrackCache
from musicagent.models import Track, TrackRef, UnresolvedTrack

# Machine-readable reason -> human-readable sentence, used to build
# UnresolvedTrack entries. Keep in sync with UnresolvedTrack.reason's Literal.
_REASON_MESSAGES = {
    "not_found": "No provider recognised this track.",
    "no_key": "Found the track and its tempo, but no provider supplied a musical key.",
    "no_bpm": "Found a musical key for this track, but no provider supplied its tempo.",
    "timeout": "Lookup for this track did not finish before the per-track deadline.",
    "error": "An unexpected error occurred while looking up this track.",
}


def _unresolved(ref: TrackRef, reason: str) -> UnresolvedTrack:
    return UnresolvedTrack(
        artist=ref.artist,
        title=ref.title,
        reason=reason,
        message=_REASON_MESSAGES[reason],
    )


TIMEOUT = 10.0
RETRIES = 2

# MusicBrainz allows 1 request/second per client and starts returning 503 once
# exceeded. This is a hard external constraint (not configurable per-request),
# so it's enforced with a module-level throttle shared by every concurrent
# enrichment, applied ONLY to MusicBrainz calls -- Deezer/GetSongBPM/Last.fm
# stay fully concurrent.
MUSICBRAINZ_MIN_INTERVAL_S = 1.1
MUSICBRAINZ_USER_AGENT = "musicagent/0.1 ( https://github.com/ )"
_musicbrainz_lock = asyncio.Lock()
_musicbrainz_last_request_at = 0.0
# Per-track deadline. It is wall-clock, so it also covers the time a track
# spends queued behind others for a MusicBrainz send slot: with the throttle
# above, the Nth track in a batch waits ~N * MUSICBRAINZ_MIN_INTERVAL_S before
# its own request even starts. Sized so a full MAX_TRACKS (30) request still
# fits; the request as a whole is bounded separately by OVERALL_DEADLINE_S in
# api.py, and a warm cache skips all of this.
#
# Arithmetic for the lowered value below: the audio-analysis path (now the
# primary key/bpm source, see _audio) costs ~0.75s/track (download + decode +
# analyse, measured), so a cold-cache batch that resolves entirely through it
# finishes in a couple of seconds regardless of MAX_TRACKS. The MusicBrainz/
# AcousticBrainz fallback is still needed for the tracks Deezer never found at
# all (no preview to analyse), and it alone can cost ~MAX_TRACKS *
# MUSICBRAINZ_MIN_INTERVAL_S ~= 30 * 1.1 ~= 33s if literally every track in a
# full-size (30-track) request falls all the way through to it. 20s of
# headroom on top of that covers per-track HTTP latency/retries, so 55s stays
# comfortably above the worst case while being well below the old 45s+heavy-
# margin figure that assumed AcousticBrainz was the *primary* path rather than
# a rarely-hit fallback.
ENRICH_DEADLINE_S = 55.0
ENRICH_CONCURRENCY = 8

# Preview clips are small (typically ~470KB); refuse anything drastically
# larger rather than buffering an unbounded response in memory.
MAX_PREVIEW_BYTES = 5 * 1024 * 1024

# Analysis (ffmpeg decode + essentia) is blocking CPU work run via
# asyncio.to_thread, gated by its own semaphore separate from
# ENRICH_CONCURRENCY (8): that constant bounds concurrent *I/O-bound* per-track
# pipelines (mostly waiting on HTTP), which is fine to set high, but letting 8
# CPU-bound essentia analyses run at once on what is typically a single shared
# vCPU (free-tier hosting) would thrash rather than speed anything up. Capped
# much lower, independently of how many tracks are in flight overall.
AUDIO_ANALYSIS_CONCURRENCY = 2
_audio_semaphore = asyncio.Semaphore(AUDIO_ANALYSIS_CONCURRENCY)


async def _get_json(client: httpx.AsyncClient, url: str, **kw) -> dict | None:
    """GET url, retrying RETRIES times with exponential backoff (0.5s, 1s) on any
    transport/HTTP-status error or malformed (non-JSON) response body. Never raises;
    returns None if all attempts fail or the parsed payload is not a JSON object
    (a list/string/number is treated the same as a failed response, since callers
    only know how to use a dict)."""
    for attempt in range(RETRIES + 1):
        try:
            r = await client.get(url, timeout=TIMEOUT, **kw)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, dict):
                if attempt == RETRIES:
                    return None
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            return data
        except (httpx.HTTPError, ValueError):
            if attempt == RETRIES:
                return None
            await asyncio.sleep(0.5 * 2**attempt)
    return None


async def _get_bytes(
    client: httpx.AsyncClient, url: str, max_bytes: int = MAX_PREVIEW_BYTES
) -> bytes | None:
    """GET url and return the raw response body, retrying like _get_json but
    without assuming a JSON payload (a preview clip is audio, not JSON, so
    _get_json's r.json() parse doesn't fit). Streams the body so an
    over-large response is rejected as soon as it exceeds max_bytes rather
    than being fully buffered first; returns None on any transport/HTTP-status
    error, or if the body exceeds max_bytes, after RETRIES retries."""
    for attempt in range(RETRIES + 1):
        try:
            async with client.stream("GET", url, timeout=TIMEOUT) as r:
                r.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                async for chunk in r.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        return None
                    chunks.append(chunk)
                return b"".join(chunks)
        except httpx.HTTPError:
            if attempt == RETRIES:
                return None
            await asyncio.sleep(0.5 * 2**attempt)
    return None


async def _deezer(client: httpx.AsyncClient, ref: TrackRef) -> dict:
    """Deezer is a two-step lookup: /search finds the track id (a search hit
    carries only album, artist, duration, explicit_*, id, isrc, link,
    md5_image, preview, rank, readable, title, title_short, title_version,
    type -- no bpm/gain, verified against the live API), then /track/{id}
    returns the full track object, which does carry bpm/gain."""
    q = f'artist:"{ref.artist}" track:"{ref.title}"'
    data = await _get_json(client, "https://api.deezer.com/search", params={"q": q})
    items = (data or {}).get("data") or []
    if not items:
        return {}
    hit = items[0]
    track_id = hit.get("id")
    # The preview clip URL lives on the *search* hit, not the /track/{id}
    # response -- captured here so the audio-analysis provider downstream can
    # reuse it without searching Deezer a second time.
    preview_url = hit.get("preview")
    if not track_id:
        return {}

    track_data = await _get_json(client, f"https://api.deezer.com/track/{track_id}")
    if not track_data:
        return {"preview_url": preview_url} if preview_url else {}

    out: dict = {"duration_s": track_data.get("duration")}
    if preview_url:
        out["preview_url"] = preview_url
    # Deezer returns bpm=0 when the tempo is unknown; treat that like a
    # missing value so the cascade falls through to the next provider
    # instead of taking 0 BPM as a real reading.
    if track_data.get("bpm"):
        out["bpm"] = float(track_data["bpm"])
    # Deezer returns gain=0 when loudness is unknown (verified against the
    # live API), the same sentinel convention as bpm=0 above; treat it as
    # missing rather than as the maximum-energy real gain value of 0 dB.
    if track_data.get("gain"):
        out["energy"] = min(max((track_data["gain"] + 20) / 20, 0.0), 1.0)
    return out


async def _getsongbpm(client: httpx.AsyncClient, ref: TrackRef) -> dict:
    api_key = os.environ.get("GETSONGBPM_API_KEY")
    if not api_key:
        # Missing key: skip the provider entirely rather than making a doomed request.
        return {}
    data = await _get_json(
        client,
        "https://api.getsongbpm.com/search/",
        params={
            "type": "both",
            "lookup": f"song:{ref.title} artist:{ref.artist}",
            "api_key": api_key,
        },
    )
    items = (data or {}).get("search") or []
    if not items:
        return {}
    hit = items[0]
    out: dict = {}
    if hit.get("tempo"):
        out["bpm"] = float(hit["tempo"])
    if hit.get("key_of"):
        try:
            out["camelot"] = parse_camelot(hit["key_of"])
        except ValueError:
            pass
    return out


async def _audio(client: httpx.AsyncClient, preview_url: str | None) -> dict:
    """Analyse the Deezer preview clip (if any) for key/bpm. Takes the
    preview URL rather than a TrackRef -- unlike every other provider here --
    because it was already extracted from _deezer's search hit above, and
    re-searching Deezer just to get the same URL again would be wasteful.

    Coverage argument (measured on a real underground-techno playlist, see
    README/spec): MusicBrainz/AcousticBrainz resolved 0/15 tracks, Deezer
    metadata supplied bpm for 7 and never a key, but 13/15 had a public
    preview clip -- analysing it gives key + bpm for anything Deezer carries
    at all, mainstream or not.
    """
    if not preview_url:
        return {}
    mp3_bytes = await _get_bytes(client, preview_url)
    if not mp3_bytes:
        return {}
    async with _audio_semaphore:
        from musicagent.audio import analyze_preview

        return await asyncio.to_thread(analyze_preview, mp3_bytes)


async def _musicbrainz_search(client: httpx.AsyncClient, ref: TrackRef) -> list[str]:
    """Query MusicBrainz recording search for candidate MBIDs, respecting the
    module-level 1 req/sec throttle (MusicBrainz starts returning 503 above
    that rate). Returns every id in the response: a track commonly has several
    recording MBIDs and only some were ever analysed by AcousticBrainz, so all
    candidates are tried together in one AcousticBrainz batch call."""
    global _musicbrainz_last_request_at
    query = f'artist:"{ref.artist}" AND recording:"{ref.title}"'
    # Reserve this caller's send slot under the lock, then release it before
    # sleeping and before the request itself. Holding the lock across the HTTP
    # call would serialize whole request durations, not just space the sends
    # one interval apart, and every queued track would pay for the ones ahead.
    async with _musicbrainz_lock:
        slot = max(
            time.monotonic(), _musicbrainz_last_request_at + MUSICBRAINZ_MIN_INTERVAL_S
        )
        _musicbrainz_last_request_at = slot
    wait = slot - time.monotonic()
    if wait > 0:
        await asyncio.sleep(wait)
    data = await _get_json(
        client,
        "https://musicbrainz.org/ws/2/recording",
        params={"query": query, "fmt": "json", "limit": 15},
        headers={"User-Agent": MUSICBRAINZ_USER_AGENT},
    )
    recordings = (data or {}).get("recordings") or []
    return [r["id"] for r in recordings if isinstance(r, dict) and r.get("id")]


async def _musicbrainz_acousticbrainz(client: httpx.AsyncClient, ref: TrackRef) -> dict:
    """Fallback with no API key needed: MusicBrainz recording search finds
    candidate MBIDs, then a single batch AcousticBrainz lookup supplies key,
    bpm and an energy proxy for whichever of them were ever analysed
    (one-mbid-at-a-time lookups see roughly a 1/6 hit rate, hence the batch).

    Several MBIDs of the same track may carry analyses that disagree — the key
    is estimated from audio, not read from metadata — so the candidate with the
    highest `key_strength` wins, ties broken by MBID for stability. Picking
    whichever happened to come back first made the same track resolve to a
    different key between runs."""
    mbids = await _musicbrainz_search(client, ref)
    if not mbids:
        return {}

    ab_data = await _get_json(
        client,
        "https://acousticbrainz.org/api/v1/low-level",
        params={"recording_ids": ";".join(mbids)},
    )
    if not ab_data:
        return {}

    candidates = []
    for mbid in mbids:
        entry = ab_data.get(mbid)
        candidate = (entry or {}).get("0") if isinstance(entry, dict) else None
        if isinstance(candidate, dict) and isinstance(candidate.get("tonal"), dict):
            strength = candidate["tonal"].get("key_strength")
            strength = float(strength) if isinstance(strength, (int, float)) else 0.0
            candidates.append((strength, mbid, candidate))
    if not candidates:
        return {}
    doc = max(candidates, key=lambda c: (c[0], c[1]))[2]

    out: dict = {}
    tonal = doc["tonal"]
    key_key = tonal.get("key_key")
    key_scale = tonal.get("key_scale")
    if key_key and key_scale:
        try:
            out["camelot"] = parse_camelot(f"{key_key} {key_scale}")
        except ValueError:
            pass
    key_strength = tonal.get("key_strength")
    if isinstance(key_strength, (int, float)):
        out["key_confidence"] = float(key_strength)

    rhythm = doc.get("rhythm")
    if isinstance(rhythm, dict) and rhythm.get("bpm"):
        out["bpm"] = float(rhythm["bpm"])

    lowlevel = doc.get("lowlevel")
    if isinstance(lowlevel, dict) and lowlevel.get("average_loudness") is not None:
        out["energy"] = min(max(float(lowlevel["average_loudness"]), 0.0), 1.0)

    return out


async def _lastfm_tags(client: httpx.AsyncClient, ref: TrackRef) -> list[str]:
    api_key = os.environ.get("LASTFM_API_KEY")
    if not api_key:
        return []
    data = await _get_json(
        client,
        "https://ws.audioscrobbler.com/2.0/",
        params={
            "method": "track.getTopTags",
            "artist": ref.artist,
            "track": ref.title,
            "api_key": api_key,
            "format": "json",
        },
    )
    tags = ((data or {}).get("toptags") or {}).get("tag") or []
    # Last.fm returns a single tag as a bare dict instead of a one-element list.
    if isinstance(tags, dict):
        tags = [tags]
    names = []
    for t in tags:
        if isinstance(t, dict) and t.get("name"):
            names.append(t["name"])
        if len(names) == 5:
            break
    return names


_PROVIDER_NAMES = {
    _deezer: "deezer",
    _getsongbpm: "getsongbpm",
    _audio: "audio",
    _musicbrainz_acousticbrainz: "acousticbrainz",
}


async def _enrich_one_inner(
    ref: TrackRef, client: httpx.AsyncClient, cache: TrackCache | None
) -> Track | UnresolvedTrack:
    merged: dict = {}
    # Which provider supplied the musically load-bearing fields, in the order
    # they were first set (so "source" reflects true provenance, never a
    # provider that merely returned an incomplete hit).
    field_source: dict[str, str] = {}
    # Captured from _deezer's search hit (see _deezer) and threaded through to
    # _audio below, which takes it directly instead of a TrackRef -- it has no
    # need to search Deezer a second time for the same URL.
    preview_url: str | None = None
    for provider in (_deezer, _getsongbpm, _audio, _musicbrainz_acousticbrainz):
        if provider is _audio:
            got = await _audio(client, preview_url)
        else:
            got = await provider(client, ref)
            if provider is _deezer:
                preview_url = got.pop("preview_url", None)
        name = _PROVIDER_NAMES[provider]
        for k, v in got.items():
            # Every field but `energy` follows first-writer-wins: the cascade
            # order already encodes precedence (e.g. Deezer's bpm is
            # authoritative metadata and must not be displaced by a later
            # provider's guess). `energy` is the deliberate exception --
            # Deezer's gain-derived value and AcousticBrainz's loudness proxy
            # are both crude stand-ins, whereas `_audio` is a direct
            # measurement of the actual preview clip, so it must win
            # regardless of what an earlier provider already set.
            if k == "energy":
                if provider is _audio or "energy" not in merged:
                    merged["energy"] = v
            elif k not in merged:
                merged[k] = v
                if k in ("bpm", "camelot"):
                    field_source[k] = name
        if "bpm" in merged and "camelot" in merged:
            break

    if "bpm" not in merged or "camelot" not in merged:
        if "bpm" not in merged and "camelot" not in merged:
            reason = "not_found"
        elif "camelot" not in merged:
            reason = "no_key"
        else:
            reason = "no_bpm"
        return _unresolved(ref, reason)

    providers = []
    for key in ("bpm", "camelot"):
        p = field_source.get(key)
        if p and p not in providers:
            providers.append(p)
    source = "+".join(providers) if providers else "unknown"

    track = Track(
        ref=ref,
        bpm=merged["bpm"],
        camelot=merged["camelot"],
        energy=merged.get("energy", 0.5),
        duration_s=merged.get("duration_s"),
        tags=await _lastfm_tags(client, ref),
        source=source,
        key_confidence=merged.get("key_confidence"),
    )
    if cache:
        cache.put(track)
    return track


async def enrich_one(
    ref: TrackRef, client: httpx.AsyncClient, cache: TrackCache | None
) -> Track | UnresolvedTrack:
    """Resolve bpm/camelot/tags for a single track via the provider cascade
    (Deezer -> GetSongBPM -> MusicBrainz/AcousticBrainz), checking the cache
    first and writing back on success.
    Returns an `UnresolvedTrack` (carrying a machine-readable reason) if no
    provider combination yields both bpm and camelot, or if the whole lookup
    does not complete within ENRICH_DEADLINE_S seconds (reason "timeout")."""
    if cache and (hit := cache.get(ref)):
        return hit

    try:
        return await asyncio.wait_for(
            _enrich_one_inner(ref, client, cache), timeout=ENRICH_DEADLINE_S
        )
    except TimeoutError:
        return _unresolved(ref, "timeout")


async def enrich_all(
    refs: list[TrackRef], cache: TrackCache | None
) -> tuple[list[Track], list[UnresolvedTrack]]:
    """Enrich all refs concurrently, bounded to ENRICH_CONCURRENCY tracks in flight
    at once (a semaphore gates per-track work) so a large track list can't fan out
    into an unbounded number of simultaneous upstream HTTP calls. A single
    unresolvable/failing track never fails the whole batch; it is simply reported
    in the unresolved list, each entry carrying a reason (see UnresolvedTrack).
    Any unexpected exception from an individual track's enrichment is also
    treated as unresolved (reason "timeout" for a TimeoutError that somehow
    still escapes enrich_one, "error" for anything else) rather than
    propagated, so one buggy provider path can never take down the batch."""
    sem = asyncio.Semaphore(ENRICH_CONCURRENCY)

    async def _bounded(
        ref: TrackRef, client: httpx.AsyncClient
    ) -> Track | UnresolvedTrack:
        async with sem:
            return await enrich_one(ref, client, cache)

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(_bounded(r, client) for r in refs), return_exceptions=True
        )
    resolved = [t for t in results if isinstance(t, Track)]
    unresolved: list[UnresolvedTrack] = []
    for ref, t in zip(refs, results):
        if isinstance(t, Track):
            continue
        if isinstance(t, UnresolvedTrack):
            unresolved.append(t)
        elif isinstance(t, TimeoutError):
            unresolved.append(_unresolved(ref, "timeout"))
        else:
            unresolved.append(_unresolved(ref, "error"))
    return resolved, unresolved
