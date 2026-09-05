# Set & Release Agent (music-agent)

A portfolio project: an AI agent that understands tracks the way a DJ does.
Give it a rough list of songs and it builds a harmonically mixed DJ set —
ordering tracks by key (Camelot wheel), tempo, and energy — and explains, in
plain language, why each transition works.

This is the **Set Builder** (MVP). A second mode, **Release Assistant**
(given your own track, suggest similar artists / playlists / pitch drafts),
is designed but deferred — see [Phase 2 roadmap](#whats-next) below.

Full design spec: [`spec.md`](spec.md) (mirrored at
[`docs/superpowers/specs/2026-09-05-set-and-release-agent-design.md`](docs/superpowers/specs/2026-09-05-set-and-release-agent-design.md)).
Implementation plan: [`docs/superpowers/plans/2026-09-05-set-builder-mvp.md`](docs/superpowers/plans/2026-09-05-set-builder-mvp.md).

## Design thesis

The interesting part of this project is *where the LLM is used and where it
isn't*:

- **LLM only for language** — parsing a free-text request into a structured
  track list (`parse_input`), and turning a computed set order into
  human-readable transition explanations (`explain_set`).
- **Everything about the music itself is deterministic, unit-tested Python**
  — Camelot wheel neighbor rules, the BPM compatibility window, the energy
  curve, and the beam search that finds the best playable order. None of
  that is delegated to a model: it's cheaper, faster, reproducible, and
  actually testable.

This split is enforced by the graph shape itself (see below): the two LLM
nodes sit at the edges of the pipeline, and the three nodes in between are
pure/async Python with no model calls.

## Architecture — the Set Builder graph

A five-node LangGraph pipeline, traced end-to-end with Langfuse:

```
                LLM                  code               pure Python        pure Python              LLM
        ┌────────────────┐   ┌──────────────────┐  ┌─────────────────┐  ┌───────────────┐  ┌──────────────────┐
 text ─▶│  parse_input   │──▶│  enrich_tracks    │─▶│ build_transition │─▶│ find_set_path │─▶│   explain_set    │──▶ END
        │ free text →    │   │  (parallel HTTP)  │  │    _graph        │  │ (beam search  │  │ SetPath →        │
        │ SetRequest     │   │ TrackRef → Track  │  │ Track list →     │  │  over energy  │  │ SetResult        │
        │ {tracks,       │   │ (+ unresolved)    │  │ TransitionGraph  │  │  shape)       │  │ {transitions,    │
        │  energy_shape} │   │                   │  │ {edges: (a,b,    │  │               │  │  summary}        │
        └────────────────┘   └──────────────────┘  │  score)}         │  └───────────────┘  └──────────────────┘
                                                     └─────────────────┘
```

| Node | Kind | Contract |
|---|---|---|
| `parse_input` | LLM | free text → `SetRequest{tracks, duration_min?, energy_shape}` |
| `enrich_tracks` | code, parallel | `list[TrackRef]` → `list[Track]` (+ `unresolved`) |
| `build_transition_graph` | pure Python | `list[Track]` → `TransitionGraph{edges: (a, b, score)}` |
| `find_set_path` | pure Python | `TransitionGraph` + `energy_shape` → `SetPath` (ordered tracks + per-edge scores) |
| `explain_set` | LLM | `SetPath` → `SetResult{transitions: [{from, to, explanation}], summary}` |

All contracts are Pydantic models in `src/musicagent/models.py`; the
LangGraph state schema (`SetState` in `src/musicagent/graph.py`) mirrors this
table exactly.

**Deterministic rules** (`src/musicagent/core/`, all unit-tested, no
network/LLM):
- Camelot compatibility: same key, ±1 on the wheel, or the relative
  major/minor.
- BPM window: ±6% (configurable constant).
- Edge score: weighted sum of key compatibility, BPM distance, and the
  energy delta *between the two adjacent tracks*.
- Path search: greedy beam search over the transition graph, scoring each
  candidate track by its edge score plus a separate term for how close its
  energy is to the target curve for the requested shape (`build`,
  `peak_end`, `wave`) at that position -- two different energy comparisons
  doing two different jobs (local smoothness vs. overall arc).

**Enrichment cascade** (`src/musicagent/enrichment.py`): per track, Deezer
(no key required) → GetSongBPM → audio analysis of the Deezer preview clip
(`src/musicagent/audio.py`) → MusicBrainz/AcousticBrainz for BPM/key, with
tags/genre from Last.fm. The cascade stops as soon as both are known.

Audio analysis exists because the metadata providers cover mainstream
releases well but miss underground ones almost entirely — measured on a real
15-track underground techno playlist, MusicBrainz/AcousticBrainz resolved
0/15 and Deezer's metadata gave bpm for 7 tracks and never a key, but 13/15
had a public preview clip on Deezer. Analysing that clip (via
[essentia](https://essentia.upf.edu/)'s `KeyExtractor` + `RhythmExtractor2013`,
after an `ffmpeg` decode to mono/44.1kHz wav) gives key, bpm, and an energy
proxy for anything Deezer carries at all. It's optional at runtime — `essentia`
is a separate `audio` dependency extra (not installed by default; importing
it costs ~190MB RSS) and `ffmpeg` can't come from pip — so a deployment
without either just runs on the metadata providers, no crash. Key detection
this way is roughly 70-80% accurate, same as AcousticBrainz's, which is why
`Track.key_confidence` is populated from it too. See Limitations below for
the caveat on Deezer's terms of use.

MusicBrainz/AcousticBrainz needs no API key: MusicBrainz recording search
finds candidate MBIDs, then a single batch AcousticBrainz lookup (all
candidate MBIDs in one request) supplies key, BPM, and an energy proxy for
whichever were ever analysed — one-MBID-at-a-time lookups see roughly a 1/6
hit rate, hence the batch. MusicBrainz enforces a hard 1 request/second
limit, enforced here by a module-level throttle applied only to that
provider (the others stay fully concurrent); see Limitations below for what
that costs in latency. Every external call has a timeout and retries.
Results are cached in the `tracks` table (Postgres, or SQLite locally); a
cache hit skips all external calls. Tracks that can't be resolved are
marked `unresolved`, excluded from the graph, and reported back in the
response instead of silently dropped.

## Tech stack

- **Python 3.13**, [uv](https://docs.astral.sh/uv/) for dependency management
- **LangGraph** for the graph above; [deepagents](https://github.com/langchain-ai/deepagents)
  is a dependency for the phase-2 Release Assistant subagents
- **FastAPI** + Server-Sent Events (`sse-starlette`) for streaming progress
- **LangChain** as the LLM abstraction — OpenAI (or Luna, an
  OpenAI-compatible gateway) primary, Gemini/Groq free tier as fallback
- **Langfuse** for tracing — one span per graph node, on every run
- **SQLAlchemy** over Postgres (Supabase free tier) in production, or plain
  SQLite locally/in tests — same code path, just a different
  `DATABASE_URL`
- **httpx** for external API calls; **pytest** + **respx** for tests

## Quickstart

```bash
uv sync                    # install dependencies (incl. dev group)
cp .env.example .env       # fill in keys — see table below
```

### Run the tests (no keys needed)

```bash
uv run pytest        # 158 tests, no network/LLM calls — all pure Python + stubs
uv run ruff check .   # lint
```

### Run the API locally against SQLite (no external services)

You can boot the API without any keys at all — enrichment/LLM calls will
simply fail if invoked, but `/health` and `/sets/{id}` work fully offline:

```bash
DATABASE_URL=sqlite:////tmp/musicagent.db \
  uv run uvicorn --factory musicagent.api:get_app --port 8123
```

```bash
$ curl -s http://127.0.0.1:8123/health
{"ok":true}

$ curl -s -w '\n%{http_code}\n' http://127.0.0.1:8123/sets/does-not-exist
{"detail":"Not Found"}
404
```

### Run it for real (with an LLM + database)

Once `.env` is filled in (`OPENAI_API_KEY` or a Luna endpoint, and either a
`DATABASE_URL` pointing at Supabase Postgres or the SQLite path above):

```bash
uv run uvicorn --factory musicagent.api:get_app --port 8123
```

`POST /sets` streams progress over SSE and ends with a `SetResult`:

```bash
curl -N -X POST http://127.0.0.1:8123/sets \
  -H 'Content-Type: application/json' \
  -d '{"text": "Build me a peak-time set from: Adriatique - Los Angeles, Tale Of Us - Yang, Massano - Hoshi"}'
```

```
event: progress
data: parse_input

event: progress
data: enrich_tracks

event: progress
data: build_transition_graph

event: progress
data: find_set_path

event: progress
data: explain_set

event: result
data: {"set_id": "…", "result": {"transitions": [...], "summary": "..."}}
```

Replay a saved set:

```bash
curl -s http://127.0.0.1:8123/sets/<set_id>
```

If Langfuse env vars are set, every run also produces a trace at
`LANGFUSE_HOST` (node-level spans for the whole graph).

## Environment variables

From [`.env.example`](.env.example):

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | for LLM nodes | Primary LLM provider (OpenAI or Luna) |
| `OPENAI_BASE_URL` | optional | Point at Luna or another OpenAI-compatible gateway instead of api.openai.com |
| `GOOGLE_API_KEY` | optional | Gemini free tier, fallback LLM |
| `GROQ_API_KEY` | optional | Groq free tier, fallback LLM |
| `LANGFUSE_PUBLIC_KEY` | optional | Tracing — omit to disable tracing entirely (checked at runtime) |
| `LANGFUSE_SECRET_KEY` | optional | Tracing |
| `LANGFUSE_HOST` | optional | Defaults to `https://cloud.langfuse.com` |
| `DATABASE_URL` | **required** | Postgres (Supabase) in prod, or any SQLAlchemy URL (e.g. `sqlite:////tmp/musicagent.db`) locally/in tests |
| `LASTFM_API_KEY` | for full enrichment | Tags / genre / similar artists |
| `GETSONGBPM_API_KEY` | for full enrichment | BPM/key fallback after Deezer (see limitations — currently unreliable) |
| `SITE_ORIGIN` | recommended | Allows CORS from this origin; with none set, CORS is closed (no cross-origin access at all) |

Deezer and MusicBrainz/AcousticBrainz need no key. Audio analysis of Deezer
preview clips needs no key either, but does need the `audio` extra
(`uv sync --extra audio`) and an `ffmpeg` binary on `PATH` — see below.

### Audio analysis extra (optional)

```bash
uv sync --extra audio   # installs essentia (~190MB RSS once imported)
brew install ffmpeg     # or apt-get install ffmpeg, etc — not installable via pip
```

Without this extra and/or without `ffmpeg` on `PATH`, the audio-analysis
provider silently contributes nothing (a single warning-level log line, not
per track) and the cascade falls through to MusicBrainz/AcousticBrainz as
before — the app runs fine either way. The provided `Dockerfile` installs
both by default.

### Running with Docker

```bash
docker build -t musicagent .
docker run -p 8123:8123 -e DATABASE_URL=sqlite:////tmp/musicagent.db musicagent
```

The image installs `ffmpeg` and the `audio` extra, so the audio-analysis
provider is available out of the box; supply the rest of the environment
variables below (`OPENAI_API_KEY`, `DATABASE_URL` for Postgres, etc.) with
`-e` or an env file as usual.

## Limitations (MVP)

- The enrichment cascade is **Deezer → GetSongBPM → audio analysis of the
  Deezer preview clip → MusicBrainz/AcousticBrainz**, with tags from Last.fm
  (spec §3). GetSongBPM in practice rarely contributes a key: its API
  requires a public backlink to the site that isn't in place, so that step is
  effectively skipped.
- **Coverage is uneven by release type, which is exactly why audio analysis
  exists.** Measured against the live APIs on the same underground techno
  playlist: 6/6 resolved on a mainstream electronic/rock sample via metadata
  alone, but only 0/15 via MusicBrainz/AcousticBrainz on an underground
  techno playlist and 7/15 via Deezer's own bpm field (never a key) — while
  13/15 of those same tracks had a public Deezer preview clip, which audio
  analysis can resolve key + bpm from. Catalogued, released music still
  resolves fine from metadata alone; audio analysis is what closes most of
  the gap for small-label/underground tracks that Deezer at least indexes
  (has *some* metadata and a preview for) but never analysed itself.
- **Audio analysis needs an optional dependency and a binary not installable
  via pip.** `essentia` (~190MB RSS to import) is behind the `audio` extra,
  and it needs `ffmpeg` on `PATH` to decode the mp3 preview. Both are
  optional at runtime — missing either just disables this one provider (one
  warning-level log line, not per track) and the cascade falls through to
  MusicBrainz/AcousticBrainz as before. The provided `Dockerfile` installs
  both.
- **Key detection is algorithmic, not ground truth**, whichever provider
  supplies it. Essentia's key estimate (from the preview clip) and
  AcousticBrainz's (`tonal.key_strength`) are both roughly 70-80% accurate,
  which is why `Track.key_confidence` is surfaced through to `explain_set`
  either way — the LLM is asked to hedge the key claim when confidence is low
  rather than state it flatly.
- **Deezer's terms of use don't explicitly address analysing preview clips**
  (as opposed to linking to or playing them). This MVP treats that as
  acceptable for a non-commercial portfolio demo; it isn't a reviewed legal
  position, and a production/commercial deployment should get its own read on
  this before relying on the same approach.
- **MusicBrainz's rate limit adds latency, but only for the tracks that reach
  it.** It allows only 1 request/second per client, enforced here by a
  module-level throttle on that provider alone. Audio analysis sits ahead of
  it in the cascade and resolves most tracks Deezer has any record of at all
  (~0.75s/track), so MusicBrainz/AcousticBrainz is now reached only for
  tracks Deezer never found — but a cold-cache request where every track
  falls all the way through can still take on the order of ~35s for
  enrichment alone (30 tracks × ~1.1s throttle interval, the worst case),
  even though every other provider runs fully concurrently.
- **No auth** — the API is a public portfolio demo. `POST /sets` has a
  small in-process rate limit (5 requests/minute per client host); a real
  multi-instance deployment would still want rate limiting at the platform
  level too (spec §9). The client host is the raw transport peer, so behind
  a non-loopback edge proxy every user shares one bucket unless uvicorn runs
  with `--forwarded-allow-ips`.
- Request body is capped at **4000 characters** of free text and the parsed
  track list is capped at **30 tracks** per request (excess tracks are
  truncated with a `notice` in the response, not rejected).
- No user-uploaded audio file analysis and no Spotify integration by design
  (audio features API closed to new apps since Nov 2024) — see spec §9. The
  only audio dependency in the MVP is the narrow, optional preview-clip
  provider described above; there's no path for the user to upload or submit
  their own audio.
- A longer set always beats a shorter one in the pathfinder's search (it
  prefers path length over score, see `find_path`), so `energy_shape` is a
  preference that shapes *which* tracks and *what order*, not a guarantee
  that the returned arc will closely track the target curve -- especially
  on a small or poorly-connected track pool.
- No DB migrations: `init_db` calls `metadata.create_all`, which only
  creates missing tables/columns, it doesn't alter existing ones. The first
  schema change made against a live (non-empty) database needs a manual
  migration, not just a code change.

## What's next

From spec §8 (phase 2 roadmap, not in MVP):

- **Release Assistant** on deepagents: researcher / positioning / copywriter
  subagents, given the user's own track.
- RAG on pgvector — a corpus of pitch examples and label/playlist guides
  (deliberately not used in the MVP: harmonic mixing rules are deterministic
  and need no retrieval).
- An MCP server exposing the Track Intelligence tools directly.
- A combined mode: "promo set around your release" — build a set of similar
  tracks with the user's own release at the peak.
