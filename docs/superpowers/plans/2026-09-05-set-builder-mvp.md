# Set Builder MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deployed FastAPI service that turns a free-text track list into a harmonically mixed DJ set with per-transition explanations.

**Architecture:** LangGraph pipeline `parse_input → enrich_tracks → build_transition_graph → find_set_path → explain_set`. LLM only at the ends (parse, explain); Camelot/BPM/energy math is pure, unit-tested Python. External music APIs cascade with a Postgres cache.

**Tech Stack:** Python 3.13, uv, LangGraph, langchain-openai (Luna via `OPENAI_BASE_URL`), Langfuse, SQLAlchemy + psycopg (Supabase Postgres), httpx, FastAPI + SSE, pytest + respx.

**Spec:** `docs/superpowers/specs/2026-09-05-set-and-release-agent-design.md`

## Global Constraints

- Python `>=3.12`, run everything via `uv run`.
- No LLM calls and no network in tests; HTTP mocked with respx, LLM nodes stubbed.
- All node contracts are Pydantic models in `src/musicagent/models.py`; keep them in sync with the spec table (spec-sync).
- External calls: 10s timeout, 2 retries with exponential backoff.
- BPM window ±6%; Camelot rules: same code, ±1 same letter, same number other letter.
- Keys/secrets only from env (`.env` via python-dotenv); never hardcode.
- Commit after every green task; ruff formatting is applied by hook.

---

### Task 1: Contracts (Pydantic models)

**Files:**
- Create: `src/musicagent/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `TrackRef(artist: str, title: str)`, `Track(ref: TrackRef, bpm: float, camelot: str, energy: float, duration_s: int | None, tags: list[str], source: str)`, `SetRequest(tracks: list[TrackRef], duration_min: int | None, energy_shape: Literal["build","peak_end","wave"])`, `Edge(a: int, b: int, score: float)` (indices into track list), `SetPath(tracks: list[Track], edge_scores: list[float])`, `Transition(from_track: TrackRef, to_track: TrackRef, explanation: str)`, `SetResult(transitions: list[Transition], summary: str, unresolved: list[TrackRef])`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from musicagent.models import SetRequest, TrackRef


def test_set_request_defaults():
    req = SetRequest(tracks=[TrackRef(artist="Bicep", title="Glue")])
    assert req.energy_shape == "peak_end"
    assert req.duration_min is None


def test_energy_shape_validated():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SetRequest(tracks=[], energy_shape="chaotic")
```

- [ ] **Step 2: Run test to verify it fails** — `uv run pytest tests/test_models.py -v`, expect ImportError.
- [ ] **Step 3: Implement `src/musicagent/models.py`**

```python
from typing import Literal

from pydantic import BaseModel


class TrackRef(BaseModel):
    artist: str
    title: str


class Track(BaseModel):
    ref: TrackRef
    bpm: float
    camelot: str
    energy: float = 0.5
    duration_s: int | None = None
    tags: list[str] = []
    source: str = "unknown"


class SetRequest(BaseModel):
    tracks: list[TrackRef]
    duration_min: int | None = None
    energy_shape: Literal["build", "peak_end", "wave"] = "peak_end"


class Edge(BaseModel):
    a: int
    b: int
    score: float


class SetPath(BaseModel):
    tracks: list[Track]
    edge_scores: list[float]


class Transition(BaseModel):
    from_track: TrackRef
    to_track: TrackRef
    explanation: str


class SetResult(BaseModel):
    transitions: list[Transition]
    summary: str
    unresolved: list[TrackRef] = []
```

- [ ] **Step 4: Run test to verify it passes** — same command, expect PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: node contract models"`

---

### Task 2: Camelot math

**Files:**
- Create: `src/musicagent/core/camelot.py`, `src/musicagent/core/__init__.py`
- Test: `tests/core/test_camelot.py`

**Interfaces:**
- Produces: `parse_camelot(key: str) -> str` (normalizes "Am"/"A min"/"8A" → Camelot code, raises `ValueError` on unknown), `compatible(a: str, b: str) -> bool`, `key_affinity(a: str, b: str) -> float` (1.0 same, 0.8 neighbor/relative, 0.0 otherwise).

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_camelot.py
import pytest

from musicagent.core.camelot import compatible, key_affinity, parse_camelot


@pytest.mark.parametrize(
    "raw,expected",
    [("8A", "8A"), ("Am", "8A"), ("A minor", "8A"), ("C", "8B"), ("F# min", "11A")],
)
def test_parse_camelot(raw, expected):
    assert parse_camelot(raw) == expected


def test_parse_unknown_raises():
    with pytest.raises(ValueError):
        parse_camelot("H#")


@pytest.mark.parametrize(
    "a,b,ok",
    [
        ("8A", "8A", True),   # same
        ("8A", "9A", True),   # +1
        ("8A", "7A", True),   # -1
        ("12A", "1A", True),  # wheel wraps
        ("8A", "8B", True),   # relative
        ("8A", "10A", False),
        ("8A", "9B", False),
    ],
)
def test_compatible(a, b, ok):
    assert compatible(a, b) is ok


def test_affinity_ordering():
    assert key_affinity("8A", "8A") > key_affinity("8A", "9A") > key_affinity("8A", "3B")
```

- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement**

```python
# src/musicagent/core/camelot.py
import re

_NOTE_TO_CAMELOT_MINOR = {
    "A": "8A", "A#": "9A", "BB": "9A", "B": "10A", "C": "5A", "C#": "12A",
    "DB": "12A", "D": "7A", "D#": "2A", "EB": "2A", "E": "9A", "F": "4A",
    "F#": "11A", "GB": "11A", "G": "6A", "G#": "1A", "AB": "1A",
}
_NOTE_TO_CAMELOT_MAJOR = {
    "C": "8B", "C#": "3B", "DB": "3B", "D": "10B", "D#": "5B", "EB": "5B",
    "E": "12B", "F": "7B", "F#": "2B", "GB": "2B", "G": "9B", "G#": "4B",
    "AB": "4B", "A": "11B", "A#": "6B", "BB": "6B", "B": "1B",
}
_CAMELOT_RE = re.compile(r"^(1[0-2]|[1-9])([AB])$")
_KEY_RE = re.compile(r"^([A-Ga-g][#b]?)\s*(m|min|minor|maj|major)?$")


def parse_camelot(key: str) -> str:
    raw = key.strip()
    if m := _CAMELOT_RE.match(raw.upper()):
        return f"{m.group(1)}{m.group(2)}"
    if m := _KEY_RE.match(raw):
        note = m.group(1).upper().replace("b", "B")
        # note letter itself is never 'B'-flattened; fix e.g. "Bb" -> "BB" handled above
        qual = (m.group(2) or "major").lower()
        table = _NOTE_TO_CAMELOT_MINOR if qual.startswith("m") and qual not in ("maj", "major") else _NOTE_TO_CAMELOT_MAJOR
        if note in table:
            return table[note]
    raise ValueError(f"unrecognized key: {key!r}")


def _split(code: str) -> tuple[int, str]:
    m = _CAMELOT_RE.match(code)
    if not m:
        raise ValueError(f"not a camelot code: {code!r}")
    return int(m.group(1)), m.group(2)


def compatible(a: str, b: str) -> bool:
    return key_affinity(a, b) > 0.0


def key_affinity(a: str, b: str) -> float:
    na, la = _split(a)
    nb, lb = _split(b)
    if (na, la) == (nb, lb):
        return 1.0
    if la == lb and (na - nb) % 12 in (1, 11):
        return 0.8
    if na == nb and la != lb:
        return 0.8
    return 0.0
```

- [ ] **Step 4: Run to verify PASS.** Fix the "Bb" edge case if the parametrized test catches it.
- [ ] **Step 5: Commit** — `git commit -am "feat: camelot parsing and compatibility"`

---

### Task 3: BPM window and edge scoring

**Files:**
- Create: `src/musicagent/core/scoring.py`
- Test: `tests/core/test_scoring.py`

**Interfaces:**
- Consumes: `key_affinity` from Task 2, `Track`/`Edge` from Task 1.
- Produces: `BPM_TOLERANCE = 0.06`, `bpm_ok(a: float, b: float) -> bool`, `edge_score(a: Track, b: Track) -> float` (0 when incompatible; else weighted `0.5*key + 0.3*bpm_closeness + 0.2*energy_smoothness`), `build_edges(tracks: list[Track]) -> list[Edge]` (directed, only score > 0).

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_scoring.py
from musicagent.core.scoring import bpm_ok, build_edges, edge_score
from musicagent.models import Track, TrackRef


def t(title, bpm, camelot, energy=0.5):
    return Track(ref=TrackRef(artist="x", title=title), bpm=bpm, camelot=camelot, energy=energy)


def test_bpm_window():
    assert bpm_ok(128, 130)          # ~1.6%
    assert not bpm_ok(128, 140)      # ~9%


def test_incompatible_key_scores_zero():
    assert edge_score(t("a", 128, "8A"), t("b", 128, "3B")) == 0.0


def test_same_key_beats_neighbor():
    base = t("a", 128, "8A")
    assert edge_score(base, t("b", 128, "8A")) > edge_score(base, t("c", 128, "9A"))


def test_build_edges_directed_and_filtered():
    tracks = [t("a", 128, "8A"), t("b", 129, "9A"), t("c", 90, "8A")]
    edges = build_edges(tracks)
    pairs = {(e.a, e.b) for e in edges}
    assert (0, 1) in pairs and (1, 0) in pairs
    assert not any(0 in p and 2 in p for p in pairs)  # bpm too far
```

- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement**

```python
# src/musicagent/core/scoring.py
from musicagent.core.camelot import key_affinity
from musicagent.models import Edge, Track

BPM_TOLERANCE = 0.06


def bpm_ok(a: float, b: float) -> bool:
    return abs(a - b) / max(a, b) <= BPM_TOLERANCE


def edge_score(a: Track, b: Track) -> float:
    key = key_affinity(a.camelot, b.camelot)
    if key == 0.0 or not bpm_ok(a.bpm, b.bpm):
        return 0.0
    bpm_closeness = 1.0 - (abs(a.bpm - b.bpm) / max(a.bpm, b.bpm)) / BPM_TOLERANCE
    energy_smoothness = 1.0 - min(abs(a.energy - b.energy), 1.0)
    return 0.5 * key + 0.3 * bpm_closeness + 0.2 * energy_smoothness


def build_edges(tracks: list[Track]) -> list[Edge]:
    return [
        Edge(a=i, b=j, score=s)
        for i, a in enumerate(tracks)
        for j, b in enumerate(tracks)
        if i != j and (s := edge_score(a, b)) > 0.0
    ]
```

- [ ] **Step 4: Run to verify PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat: bpm window and transition edge scoring"`

---

### Task 4: Energy curves and beam-search pathfinder

**Files:**
- Create: `src/musicagent/core/pathfinder.py`
- Test: `tests/core/test_pathfinder.py`

**Interfaces:**
- Consumes: `build_edges`, `Track`, `SetPath`.
- Produces: `target_energy(shape: str, pos: int, total: int) -> float` (build: linear 0.2→1.0; peak_end: 0.4→1.0 with peak at ~85% then hold; wave: sine between 0.3 and 0.9), `find_path(tracks: list[Track], shape: str, beam_width: int = 8) -> SetPath` (beam search maximizing edge scores + closeness of each track's energy to the target curve; visits each track at most once; returns the best-scoring longest path).

- [ ] **Step 1: Write the failing test**

```python
# tests/core/test_pathfinder.py
from musicagent.core.pathfinder import find_path, target_energy
from musicagent.models import Track, TrackRef


def t(title, bpm, camelot, energy):
    return Track(ref=TrackRef(artist="x", title=title), bpm=bpm, camelot=camelot, energy=energy)


def test_target_energy_build_is_monotonic():
    vals = [target_energy("build", i, 10) for i in range(10)]
    assert vals == sorted(vals) and vals[0] < 0.35 and vals[-1] == 1.0


def test_find_path_orders_by_energy_for_build():
    tracks = [
        t("low", 128, "8A", 0.2),
        t("mid", 128, "8A", 0.5),
        t("high", 129, "9A", 0.9),
    ]
    path = find_path(tracks, "build")
    titles = [tr.ref.title for tr in path.tracks]
    assert titles == ["low", "mid", "high"]
    assert len(path.edge_scores) == 2


def test_isolated_track_excluded():
    tracks = [t("a", 128, "8A", 0.5), t("b", 128, "9A", 0.5), t("island", 175, "3B", 0.5)]
    path = find_path(tracks, "peak_end")
    assert all(tr.ref.title != "island" for tr in path.tracks)
```

- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement**

```python
# src/musicagent/core/pathfinder.py
import math

from musicagent.core.scoring import build_edges
from musicagent.models import SetPath, Track

ENERGY_WEIGHT = 0.4


def target_energy(shape: str, pos: int, total: int) -> float:
    frac = pos / max(total - 1, 1)
    if shape == "build":
        return 0.2 + 0.8 * frac
    if shape == "peak_end":
        peak = 0.85
        return 0.4 + 0.6 * (frac / peak) if frac <= peak else 1.0
    if shape == "wave":
        return 0.6 + 0.3 * math.sin(2 * math.pi * frac)
    raise ValueError(f"unknown shape: {shape!r}")


def find_path(tracks: list[Track], shape: str, beam_width: int = 8) -> SetPath:
    n = len(tracks)
    adj: dict[int, dict[int, float]] = {}
    for e in build_edges(tracks):
        adj.setdefault(e.a, {})[e.b] = e.score

    def fit(i: int, pos: int) -> float:
        return 1.0 - abs(tracks[i].energy - target_energy(shape, pos, n))

    # beam items: (total_score, path, used)
    beam = [(fit(i, 0) * ENERGY_WEIGHT, [i], {i}) for i in range(n)]
    best = max(beam, key=lambda x: (len(x[1]), x[0]))
    while beam:
        nxt = []
        for score, path, used in beam:
            for j, s in adj.get(path[-1], {}).items():
                if j in used:
                    continue
                nscore = score + s + ENERGY_WEIGHT * fit(j, len(path))
                nxt.append((nscore, path + [j], used | {j}))
        nxt.sort(key=lambda x: x[0], reverse=True)
        beam = nxt[:beam_width]
        if beam:
            cand = max(beam, key=lambda x: (len(x[1]), x[0]))
            if (len(cand[1]), cand[0]) > (len(best[1]), best[0]):
                best = cand

    _, path, _ = best
    scores = [adj[path[k]][path[k + 1]] for k in range(len(path) - 1)]
    return SetPath(tracks=[tracks[i] for i in path], edge_scores=scores)
```

- [ ] **Step 4: Run to verify PASS** (all of `uv run pytest tests/core -v`).
- [ ] **Step 5: Commit** — `git commit -am "feat: energy curves and beam-search set pathfinder"`

---

### Task 5: Database layer (tracks cache + sets storage)

**Files:**
- Create: `src/musicagent/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `Track`, `TrackRef`, `SetResult`.
- Produces: `get_engine(url: str | None = None)` (reads `DATABASE_URL`, callers pass sqlite URL in tests), `init_db(engine)`, `TrackCache(engine)` with `.get(ref: TrackRef) -> Track | None` and `.put(track: Track) -> None` (upsert keyed on lowercased artist/title), `SetStore(engine)` with `.save(request_json: dict, result: SetResult) -> str` (returns id) and `.load(set_id: str) -> dict | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
from musicagent.db import SetStore, TrackCache, get_engine, init_db
from musicagent.models import SetResult, Track, TrackRef


def make_engine():
    e = get_engine("sqlite:///:memory:")
    init_db(e)
    return e


def test_track_cache_roundtrip_and_case_insensitive():
    cache = TrackCache(make_engine())
    ref = TrackRef(artist="Bicep", title="Glue")
    assert cache.get(ref) is None
    cache.put(Track(ref=ref, bpm=120, camelot="8A", energy=0.6, source="deezer"))
    hit = cache.get(TrackRef(artist="bicep", title="GLUE"))
    assert hit and hit.bpm == 120 and hit.source == "deezer"


def test_set_store_roundtrip():
    store = SetStore(make_engine())
    result = SetResult(transitions=[], summary="empty", unresolved=[])
    set_id = store.save({"tracks": []}, result)
    loaded = store.load(set_id)
    assert loaded and loaded["result"]["summary"] == "empty"
```

- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement**

```python
# src/musicagent/db.py
import json
import os
import uuid

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Table, MetaData, create_engine, func, select
from sqlalchemy.engine import Engine

metadata = MetaData()

tracks_table = Table(
    "tracks", metadata,
    Column("key", String, primary_key=True),  # "artist|title" lowercased
    Column("artist", String, nullable=False),
    Column("title", String, nullable=False),
    Column("bpm", Float, nullable=False),
    Column("camelot", String, nullable=False),
    Column("energy", Float, nullable=False),
    Column("duration_s", Integer),
    Column("tags", JSON, default=list),
    Column("source", String, default="unknown"),
    Column("fetched_at", DateTime, server_default=func.now()),
)

sets_table = Table(
    "sets", metadata,
    Column("id", String, primary_key=True),
    Column("request", JSON, nullable=False),
    Column("result", JSON, nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
)


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or os.environ["DATABASE_URL"])


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


def _key(artist: str, title: str) -> str:
    return f"{artist.strip().lower()}|{title.strip().lower()}"


class TrackCache:
    def __init__(self, engine: Engine):
        self.engine = engine

    def get(self, ref):
        from musicagent.models import Track, TrackRef
        with self.engine.connect() as c:
            row = c.execute(select(tracks_table).where(tracks_table.c.key == _key(ref.artist, ref.title))).mappings().first()
        if not row:
            return None
        return Track(
            ref=TrackRef(artist=row["artist"], title=row["title"]),
            bpm=row["bpm"], camelot=row["camelot"], energy=row["energy"],
            duration_s=row["duration_s"], tags=row["tags"] or [], source=row["source"],
        )

    def put(self, track) -> None:
        key = _key(track.ref.artist, track.ref.title)
        values = dict(
            key=key, artist=track.ref.artist, title=track.ref.title, bpm=track.bpm,
            camelot=track.camelot, energy=track.energy, duration_s=track.duration_s,
            tags=track.tags, source=track.source,
        )
        with self.engine.begin() as c:
            c.execute(tracks_table.delete().where(tracks_table.c.key == key))
            c.execute(tracks_table.insert().values(**values))


class SetStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def save(self, request_json: dict, result) -> str:
        set_id = uuid.uuid4().hex
        with self.engine.begin() as c:
            c.execute(sets_table.insert().values(
                id=set_id, request=request_json, result=json.loads(result.model_dump_json()),
            ))
        return set_id

    def load(self, set_id: str) -> dict | None:
        with self.engine.connect() as c:
            row = c.execute(select(sets_table).where(sets_table.c.id == set_id)).mappings().first()
        return dict(row) if row else None
```

- [ ] **Step 4: Run to verify PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat: postgres track cache and set storage"`

---

### Task 6: Enrichment cascade (Deezer → GetSongBPM → AcousticBrainz, tags via Last.fm)

**Files:**
- Create: `src/musicagent/enrichment.py`
- Test: `tests/test_enrichment.py`

**Interfaces:**
- Consumes: `TrackCache`, `parse_camelot`, models.
- Produces: `async enrich_one(ref: TrackRef, client: httpx.AsyncClient, cache: TrackCache | None) -> Track | None`, `async enrich_all(refs, cache) -> tuple[list[Track], list[TrackRef]]` (resolved, unresolved; concurrency via `asyncio.gather`). All requests use `timeout=10`, and a helper `_get_json(client, url, **kw)` retries twice with backoff (0.5s, 1s).
- Providers: Deezer `GET https://api.deezer.com/search?q=artist:"{artist}" track:"{title}"` → take `data[0]` (`bpm` > 0 required, `gain` normalized to energy via `min(max((gain + 20) / 20, 0), 1)`, key absent → try next provider for key); GetSongBPM `GET https://api.getsongbpm.com/search/?type=both&lookup=song:{title} artist:{artist}&api_key=...` → `search[0].tempo`, `.key_of`; AcousticBrainz skipped in MVP if first two fail key/bpm (mark unresolved). Last.fm `GET .../2.0/?method=track.getTopTags` for tags (optional; failure → empty tags).

- [ ] **Step 1: Write the failing test** (respx-mocked; add dep first: `uv add --dev respx`)

```python
# tests/test_enrichment.py
import httpx
import pytest
import respx

from musicagent.db import TrackCache, get_engine, init_db
from musicagent.enrichment import enrich_all, enrich_one
from musicagent.models import Track, TrackRef

DEEZER = {"data": [{"bpm": 126.0, "gain": -8.0, "duration": 240}]}
GSB = {"search": [{"tempo": "126", "key_of": "Am"}]}


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
    engine = get_engine("sqlite:///:memory:"); init_db(engine)
    resolved, unresolved = await enrich_all([TrackRef(artist="x", title="y")], TrackCache(engine))
    assert resolved == [] and len(unresolved) == 1


@pytest.mark.asyncio
async def test_cache_hit_skips_network():
    engine = get_engine("sqlite:///:memory:"); init_db(engine)
    cache = TrackCache(engine)
    ref = TrackRef(artist="a", title="b")
    cache.put(Track(ref=ref, bpm=120, camelot="8A", source="deezer"))
    resolved, unresolved = await enrich_all([ref], cache)  # no respx: network would raise
    assert resolved[0].bpm == 120 and unresolved == []
```

- [ ] **Step 2: Run to verify FAIL** (configure `asyncio_mode = "auto"` for pytest-asyncio in `pyproject.toml` under `[tool.pytest.ini_options]`).
- [ ] **Step 3: Implement**

```python
# src/musicagent/enrichment.py
import asyncio
import os

import httpx

from musicagent.core.camelot import parse_camelot
from musicagent.models import Track, TrackRef

TIMEOUT = 10.0
RETRIES = 2


async def _get_json(client: httpx.AsyncClient, url: str, **kw) -> dict | None:
    for attempt in range(RETRIES + 1):
        try:
            r = await client.get(url, timeout=TIMEOUT, **kw)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError):
            if attempt == RETRIES:
                return None
            await asyncio.sleep(0.5 * 2**attempt)
    return None


async def _deezer(client, ref) -> dict:
    q = f'artist:"{ref.artist}" track:"{ref.title}"'
    data = await _get_json(client, "https://api.deezer.com/search", params={"q": q})
    items = (data or {}).get("data") or []
    if not items:
        return {}
    hit = items[0]
    out: dict = {"duration_s": hit.get("duration"), "source": "deezer"}
    if hit.get("bpm"):
        out["bpm"] = float(hit["bpm"])
    if hit.get("gain") is not None:
        out["energy"] = min(max((hit["gain"] + 20) / 20, 0.0), 1.0)
    return out


async def _getsongbpm(client, ref) -> dict:
    data = await _get_json(
        client, "https://api.getsongbpm.com/search/",
        params={"type": "both", "lookup": f"song:{ref.title} artist:{ref.artist}",
                "api_key": os.environ.get("GETSONGBPM_API_KEY", "")},
    )
    items = (data or {}).get("search") or []
    if not items:
        return {}
    hit = items[0]
    out: dict = {"source": "getsongbpm"}
    if hit.get("tempo"):
        out["bpm"] = float(hit["tempo"])
    if hit.get("key_of"):
        try:
            out["camelot"] = parse_camelot(hit["key_of"])
        except ValueError:
            pass
    return out


async def _lastfm_tags(client, ref) -> list[str]:
    key = os.environ.get("LASTFM_API_KEY")
    if not key:
        return []
    data = await _get_json(
        client, "https://ws.audioscrobbler.com/2.0/",
        params={"method": "track.getTopTags", "artist": ref.artist, "track": ref.title,
                "api_key": key, "format": "json"},
    )
    tags = ((data or {}).get("toptags") or {}).get("tag") or []
    return [t["name"] for t in tags[:5]]


async def enrich_one(ref: TrackRef, client: httpx.AsyncClient, cache) -> Track | None:
    if cache and (hit := cache.get(ref)):
        return hit
    merged: dict = {}
    for provider in (_deezer, _getsongbpm):
        got = await provider(client, ref)
        for k, v in got.items():
            merged.setdefault(k, v)
        if "bpm" in merged and "camelot" in merged:
            break
    if "bpm" not in merged or "camelot" not in merged:
        return None
    track = Track(
        ref=ref, bpm=merged["bpm"], camelot=merged["camelot"],
        energy=merged.get("energy", 0.5), duration_s=merged.get("duration_s"),
        tags=await _lastfm_tags(client, ref), source=merged.get("source", "unknown"),
    )
    if cache:
        cache.put(track)
    return track


async def enrich_all(refs: list[TrackRef], cache) -> tuple[list[Track], list[TrackRef]]:
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*(enrich_one(r, client, cache) for r in refs))
    resolved = [t for t in results if t]
    unresolved = [r for r, t in zip(refs, results) if t is None]
    return resolved, unresolved
```

- [ ] **Step 4: Run to verify PASS.** Note: Deezer alone lacks key → test expects GetSongBPM to supply `camelot`; ensure the merge logic keeps trying providers until both bpm and camelot exist.
- [ ] **Step 5: Commit** — `git commit -am "feat: enrichment cascade with cache and retries"`

---

### Task 7: LLM nodes (parse_input, explain_set)

**Files:**
- Create: `src/musicagent/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Produces: `get_llm()` → `ChatOpenAI(model=os.environ.get("MUSICAGENT_MODEL", "gpt-4o-mini"))` (Luna picked up automatically via `OPENAI_BASE_URL`); `parse_input(text: str, llm=None) -> SetRequest` using `llm.with_structured_output(SetRequest)`; `explain_set(path: SetPath, unresolved: list[TrackRef], llm=None) -> SetResult` using `llm.with_structured_output(_Explanations)` where `_Explanations(explanations: list[str], summary: str)`; the function zips explanations onto consecutive track pairs itself (LLM never invents track order) and pads/truncates if the LLM returns a wrong count.
- Tests stub the LLM with a fake object exposing `.with_structured_output(schema)` → `.invoke(prompt)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
from musicagent.llm import explain_set, parse_input, _Explanations
from musicagent.models import SetPath, SetRequest, Track, TrackRef


class FakeLLM:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return self.result


def test_parse_input_returns_request():
    fake = FakeLLM(SetRequest(tracks=[TrackRef(artist="Bicep", title="Glue")], energy_shape="build"))
    req = parse_input("bicep glue, хочу нарастающий сет", llm=fake)
    assert req.energy_shape == "build" and req.tracks[0].title == "Glue"


def test_explain_set_zips_transitions():
    tracks = [
        Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A"),
        Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A"),
    ]
    path = SetPath(tracks=tracks, edge_scores=[0.9])
    fake = FakeLLM(_Explanations(explanations=["smooth +1 move"], summary="nice set"))
    result = explain_set(path, unresolved=[], llm=fake)
    assert len(result.transitions) == 1
    assert result.transitions[0].to_track.title == "t2"
    assert result.summary == "nice set"
```

- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement**

```python
# src/musicagent/llm.py
import os

from pydantic import BaseModel

from musicagent.models import SetPath, SetRequest, SetResult, TrackRef, Transition


def get_llm():
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=os.environ.get("MUSICAGENT_MODEL", "gpt-4o-mini"), temperature=0)


class _Explanations(BaseModel):
    explanations: list[str]
    summary: str


PARSE_PROMPT = """You parse DJ set requests. Extract the track list (artist + title)
and the desired energy shape (build / peak_end / wave; default peak_end).
User request:
{text}"""

EXPLAIN_PROMPT = """You are a DJ explaining a set. For each consecutive pair of tracks
below, write one short explanation of why the transition works (key relationship on the
Camelot wheel, BPM closeness, energy movement). Then a 1-2 sentence summary of the set arc.
Return exactly {n} explanations, in order.

Tracks (in play order, with camelot/bpm/energy):
{tracks}"""


def parse_input(text: str, llm=None) -> SetRequest:
    llm = llm or get_llm()
    return llm.with_structured_output(SetRequest).invoke(PARSE_PROMPT.format(text=text))


def explain_set(path: SetPath, unresolved: list[TrackRef], llm=None) -> SetResult:
    llm = llm or get_llm()
    pairs = list(zip(path.tracks, path.tracks[1:]))
    lines = "\n".join(
        f"{i + 1}. {t.ref.artist} - {t.ref.title} [{t.camelot}, {t.bpm:.0f} BPM, energy {t.energy:.2f}]"
        for i, t in enumerate(path.tracks)
    )
    out = llm.with_structured_output(_Explanations).invoke(
        EXPLAIN_PROMPT.format(n=len(pairs), tracks=lines)
    )
    exps = (out.explanations + [""] * len(pairs))[: len(pairs)]
    transitions = [
        Transition(from_track=a.ref, to_track=b.ref, explanation=e)
        for (a, b), e in zip(pairs, exps)
    ]
    return SetResult(transitions=transitions, summary=out.summary, unresolved=unresolved)
```

- [ ] **Step 4: Run to verify PASS.**
- [ ] **Step 5: Commit** — `git commit -am "feat: llm parse and explain nodes"`

---

### Task 8: LangGraph wiring + Langfuse

**Files:**
- Create: `src/musicagent/graph.py`
- Test: `tests/test_graph.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `class SetState(TypedDict, total=False): text: str; request: SetRequest; tracks: list[Track]; unresolved: list[TrackRef]; path: SetPath; result: SetResult`, `build_graph(cache=None, llm=None)` → compiled LangGraph, and `async run_set(text: str, cache=None, llm=None, callbacks=None) -> SetResult` (invokes graph with `config={"callbacks": callbacks or []}`); `get_langfuse_handler() -> list` returning `[CallbackHandler()]` when `LANGFUSE_SECRET_KEY` is set, else `[]` (import `from langfuse.langchain import CallbackHandler`).
- Node order per spec: parse_input → enrich_tracks → build_transition_graph+find_set_path (merged into one `find_path` node — graph build is an implementation detail of pathfinding) → explain_set → END.

- [ ] **Step 1: Write the failing test** (stub LLM, mock enrichment via cache pre-fill)

```python
# tests/test_graph.py
import pytest

from musicagent.db import TrackCache, get_engine, init_db
from musicagent.graph import run_set
from musicagent.llm import _Explanations
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
                tracks=[TrackRef(artist="a", title="t1"), TrackRef(artist="b", title="t2")],
                energy_shape="build",
            )
        return _Explanations(explanations=["works"], summary="ok")


@pytest.mark.asyncio
async def test_run_set_end_to_end_offline():
    engine = get_engine("sqlite:///:memory:"); init_db(engine)
    cache = TrackCache(engine)
    cache.put(Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A", energy=0.3))
    cache.put(Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A", energy=0.7))
    result = await run_set("a t1, b t2, build it up", cache=cache, llm=FakeLLM())
    assert len(result.transitions) == 1
    assert result.transitions[0].explanation == "works"
    assert result.unresolved == []
```

- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement**

```python
# src/musicagent/graph.py
import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from musicagent.core.pathfinder import find_path
from musicagent.enrichment import enrich_all
from musicagent.llm import explain_set, parse_input
from musicagent.models import SetPath, SetRequest, SetResult, Track, TrackRef


class SetState(TypedDict, total=False):
    text: str
    request: SetRequest
    tracks: list[Track]
    unresolved: list[TrackRef]
    path: SetPath
    result: SetResult


def get_langfuse_handler() -> list:
    if not os.environ.get("LANGFUSE_SECRET_KEY"):
        return []
    from langfuse.langchain import CallbackHandler

    return [CallbackHandler()]


def build_graph(cache=None, llm=None):
    def n_parse(state: SetState) -> SetState:
        return {"request": parse_input(state["text"], llm=llm)}

    async def n_enrich(state: SetState) -> SetState:
        tracks, unresolved = await enrich_all(state["request"].tracks, cache)
        return {"tracks": tracks, "unresolved": unresolved}

    def n_path(state: SetState) -> SetState:
        return {"path": find_path(state["tracks"], state["request"].energy_shape)}

    def n_explain(state: SetState) -> SetState:
        return {"result": explain_set(state["path"], state["unresolved"], llm=llm)}

    g = StateGraph(SetState)
    g.add_node("parse_input", n_parse)
    g.add_node("enrich_tracks", n_enrich)
    g.add_node("find_set_path", n_path)
    g.add_node("explain_set", n_explain)
    g.set_entry_point("parse_input")
    g.add_edge("parse_input", "enrich_tracks")
    g.add_edge("enrich_tracks", "find_set_path")
    g.add_edge("find_set_path", "explain_set")
    g.add_edge("explain_set", END)
    return g.compile()


async def run_set(text: str, cache=None, llm=None, callbacks=None) -> SetResult:
    graph = build_graph(cache=cache, llm=llm)
    state = await graph.ainvoke({"text": text}, config={"callbacks": callbacks or []})
    return state["result"]
```

- [ ] **Step 4: Run to verify PASS** (full suite: `uv run pytest -v`).
- [ ] **Step 5: Run graph-reviewer subagent** on `src/musicagent/graph.py`; address findings.
- [ ] **Step 6: Commit** — `git commit -am "feat: langgraph pipeline with langfuse hook"`

---

### Task 9: FastAPI app with SSE

**Files:**
- Create: `src/musicagent/api.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `build_graph`/`run_set`, `SetStore`, `get_langfuse_handler`.
- Produces: FastAPI `app` with `POST /sets` (body `{"text": "..."}`; streams SSE events `{"event": "progress", "data": "<node name>"}` after each node via `graph.astream`, final event `{"event": "result", "data": <SetResult JSON + set_id>}`), `GET /sets/{id}`, `GET /health`. CORS from `SITE_ORIGIN` env. App state built in `create_app(engine=None, llm=None)` for testability; `uvicorn musicagent.api:app` for prod. Add deps: `uv add fastapi sse-starlette uvicorn` (dev: `uv add --dev asgi-lifespan`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_api.py
import json

import httpx
import pytest

from musicagent.api import create_app
from musicagent.db import TrackCache, get_engine, init_db
from musicagent.models import Track, TrackRef
from tests.test_graph import FakeLLM


@pytest.mark.asyncio
async def test_health_and_post_sets_stream():
    engine = get_engine("sqlite:///:memory:"); init_db(engine)
    cache = TrackCache(engine)
    cache.put(Track(ref=TrackRef(artist="a", title="t1"), bpm=128, camelot="8A", energy=0.3))
    cache.put(Track(ref=TrackRef(artist="b", title="t2"), bpm=128, camelot="9A", energy=0.7))
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
    engine = get_engine("sqlite:///:memory:"); init_db(engine)
    app = create_app(engine=engine, llm=FakeLLM())
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/sets/nope")).status_code == 404
```

- [ ] **Step 2: Run to verify FAIL.**
- [ ] **Step 3: Implement**

```python
# src/musicagent/api.py
import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from musicagent.db import SetStore, TrackCache, get_engine, init_db
from musicagent.graph import build_graph, get_langfuse_handler


class SetIn(BaseModel):
    text: str


def create_app(engine=None, llm=None) -> FastAPI:
    app = FastAPI(title="Set & Release Agent")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[os.environ.get("SITE_ORIGIN", "*")],
        allow_methods=["*"], allow_headers=["*"],
    )
    eng = engine or get_engine()
    init_db(eng)
    cache, store = TrackCache(eng), SetStore(eng)
    graph = build_graph(cache=cache, llm=llm)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/sets")
    async def create_set(body: SetIn):
        async def events():
            state = {}
            async for update in graph.astream(
                {"text": body.text}, config={"callbacks": get_langfuse_handler()}
            ):
                node, out = next(iter(update.items()))
                state.update(out)
                yield {"event": "progress", "data": node}
            result = state["result"]
            set_id = store.save({"text": body.text}, result)
            yield {
                "event": "result",
                "data": json.dumps({"set_id": set_id, "result": json.loads(result.model_dump_json())}),
            }

        return EventSourceResponse(events())

    @app.get("/sets/{set_id}")
    def get_set(set_id: str):
        row = store.load(set_id)
        if not row:
            raise HTTPException(404)
        return {"set_id": set_id, "request": row["request"], "result": row["result"]}

    return app


app = create_app() if os.environ.get("DATABASE_URL") else None
```

- [ ] **Step 4: Run full suite to verify PASS** — `uv run pytest -v`.
- [ ] **Step 5: Smoke run locally** — `DATABASE_URL=sqlite:///dev.db uv run uvicorn musicagent.api:app --port 8000`, then `curl -N localhost:8000/health`.
- [ ] **Step 6: Commit** — `git commit -am "feat: fastapi sse api"`

---

### Task 10: Live smoke + docs

**Files:**
- Modify: `CLAUDE.md` (add run command), `README.md` (create: what it is, how to run, curl example)

- [ ] **Step 1:** With real `.env` (OpenAI/Luna key, Supabase URL): `uv run uvicorn musicagent.api:app` and POST a real 5-track request; verify a sensible set and a Langfuse trace appear.
- [ ] **Step 2:** Write `README.md` (project pitch, architecture diagram of the graph, quickstart, env table from `.env.example`).
- [ ] **Step 3:** Commit — `git commit -am "docs: readme and run instructions"`.
```
