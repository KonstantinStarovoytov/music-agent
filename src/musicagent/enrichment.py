import asyncio
import os

import httpx

from musicagent.core.camelot import parse_camelot
from musicagent.db import TrackCache
from musicagent.models import Track, TrackRef

TIMEOUT = 10.0
RETRIES = 2
ENRICH_DEADLINE_S = 25.0


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


async def _deezer(client: httpx.AsyncClient, ref: TrackRef) -> dict:
    q = f'artist:"{ref.artist}" track:"{ref.title}"'
    data = await _get_json(client, "https://api.deezer.com/search", params={"q": q})
    items = (data or {}).get("data") or []
    if not items:
        return {}
    hit = items[0]
    out: dict = {"duration_s": hit.get("duration")}
    if hit.get("bpm"):
        out["bpm"] = float(hit["bpm"])
    if hit.get("gain") is not None:
        out["energy"] = min(max((hit["gain"] + 20) / 20, 0.0), 1.0)
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


_PROVIDER_NAMES = {_deezer: "deezer", _getsongbpm: "getsongbpm"}


async def _enrich_one_inner(ref: TrackRef, client: httpx.AsyncClient, cache: TrackCache | None) -> Track | None:
    merged: dict = {}
    # Which provider supplied the musically load-bearing fields, in the order
    # they were first set (so "source" reflects true provenance, never a
    # provider that merely returned an incomplete hit).
    field_source: dict[str, str] = {}
    for provider in (_deezer, _getsongbpm):
        got = await provider(client, ref)
        name = _PROVIDER_NAMES[provider]
        for k, v in got.items():
            if k not in merged:
                merged[k] = v
                if k in ("bpm", "camelot"):
                    field_source[k] = name
        if "bpm" in merged and "camelot" in merged:
            break

    if "bpm" not in merged or "camelot" not in merged:
        return None

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
    )
    if cache:
        cache.put(track)
    return track


async def enrich_one(ref: TrackRef, client: httpx.AsyncClient, cache: TrackCache | None) -> Track | None:
    """Resolve bpm/camelot/tags for a single track via the provider cascade
    (Deezer -> GetSongBPM), checking the cache first and writing back on success.
    Returns None (unresolved) if no provider combination yields both bpm and camelot,
    or if the whole lookup does not complete within ENRICH_DEADLINE_S seconds."""
    if cache and (hit := cache.get(ref)):
        return hit

    try:
        return await asyncio.wait_for(_enrich_one_inner(ref, client, cache), timeout=ENRICH_DEADLINE_S)
    except asyncio.TimeoutError:
        return None


async def enrich_all(
    refs: list[TrackRef], cache: TrackCache | None
) -> tuple[list[Track], list[TrackRef]]:
    """Enrich all refs concurrently. A single unresolvable/failing track never fails
    the whole batch; it is simply reported in the unresolved list. Any unexpected
    exception from an individual track's enrichment is also treated as unresolved
    rather than propagated, so one buggy provider path can never take down the batch."""
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(enrich_one(r, client, cache) for r in refs), return_exceptions=True
        )
    resolved = [t for t in results if isinstance(t, Track)]
    unresolved = [r for r, t in zip(refs, results) if not isinstance(t, Track)]
    return resolved, unresolved
