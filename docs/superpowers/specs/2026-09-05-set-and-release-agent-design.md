# Set & Release Agent — Design Spec

Date: 2026-09-05
Status: approved (MVP scope)
Project: music-agent (portfolio pet project, vojt)

## 1. Overview

An AI agent that understands tracks the way a DJ does. Two modes sharing one
Track Intelligence core:

- **Set Builder (MVP, this spec):** given a list of tracks, builds a
  harmonically mixed DJ set (Camelot wheel, BPM, energy curve) and explains
  every transition.
- **Release Assistant (phase 2, roadmap only):** given the user's own track,
  produces a release strategy (similar artists, playlists, pitch drafts).

Portfolio thesis: LLM only where language is needed (parsing intent,
explaining decisions); all music math is deterministic, tested Python.

## 2. Stack

- Python 3.13, uv
- LangGraph (graph), deepagents (phase 2 subagents)
- FastAPI + SSE streaming; deployed to a free tier (Render/Railway/Fly)
- LLM: OpenAI (or Luna via `OPENAI_BASE_URL`); fallback Gemini/Groq free tier
- Langfuse (cloud free tier) — tracing on every node
- Supabase Postgres (free tier); pgvector enabled in phase 2
- Music data (all free): Deezer → GetSongBPM → MusicBrainz/AcousticBrainz
  (BPM/key cascade), Last.fm (tags, similar artists)

## 3. MVP architecture — Set Builder graph

```
parse_input → enrich_tracks → build_transition_graph → find_set_path → explain_set → END
```

| Node | Kind | Contract (in → out) |
|---|---|---|
| parse_input | LLM | free text → `SetRequest{tracks, duration_min?, energy_shape}` + optional `notice` string in state (truncation msg if needed); dedupes refs on normalized (artist, title), caps at MAX_TRACKS (30) |
| enrich_tracks | code, parallel | `list[TrackRef]` → `list[Track]` (+ `unresolved: list[TrackRef]`) |
| build_transition_graph | pure Python | `list[Track]` → `TransitionGraph{edges: (a, b, score)}` |
| find_set_path | pure Python | `TransitionGraph` + `energy_shape` → `SetPath{ordered tracks, per-edge scores}`, followed by a duration trim honouring `duration_min` (keeping a floor of 2 tracks) |
| explain_set | LLM | `SetPath` → `SetResult{transitions: [{from, to, explanation}], summary, unresolved, omitted}` |

`omitted` holds tracks that enriched fine but were not placed in the final
path: those trimmed to honour `duration_min`, plus any the pathfinder itself
left out.

All contracts are Pydantic models in `src/musicagent/models.py`. State schema
in code must match this table (enforced by spec-sync skill).

### Deterministic rules (never LLM)
- Camelot compatibility: same key, ±1 on the wheel, relative major/minor.
- BPM window: ±6% (configurable constant).
- Edge score: weighted sum of key compatibility, BPM distance, and the
  energy delta between the two adjacent tracks.
- Path search: greedy beam search over the transition graph, scoring each
  candidate by its edge score plus a separate term for closeness to the
  target energy curve for the requested shape (`build`, `peak_end`,
  `wave`) at that position -- these are two distinct energy comparisons.
  Returns best-scoring Hamiltonian-ish path over resolved tracks (not
  required to include all).
- Input deduplication and cap: dedupe on normalized (artist, title) then cap
  at 30 tracks, both applied before enrichment so the cost cap is real.

### Enrichment cascade
Per track: Deezer (no key needed) → GetSongBPM → MusicBrainz/AcousticBrainz;
tags/genre from Last.fm. The cascade stops as soon as both `bpm` and
`camelot` are known, so the third provider is only consulted when the first
two didn't together supply both.

MusicBrainz/AcousticBrainz needs no API key, unlike GetSongBPM (which
additionally requires a public backlink the site doesn't have, so in
practice it never returns a key today). The lookup is two calls:
1. MusicBrainz recording search (`GET /ws/2/recording?query=...&fmt=json`)
   returns a list of candidate recording MBIDs for the artist/title — a
   track commonly has several, and only some were ever analysed by
   AcousticBrainz.
2. A single **batch** AcousticBrainz low-level lookup
   (`GET /api/v1/low-level?recording_ids=<id1>;<id2>;...`) is made for all
   candidate MBIDs at once, rather than one request per MBID (which sees
   roughly a 1/6 hit rate) — the first returned document with a `tonal`
   section is used.

From that document: `tonal.key_key` + `tonal.key_scale` give the musical
key (parsed via `parse_camelot`, which already handles sharps/flats and
major/minor); `rhythm.bpm` gives the tempo; `lowlevel.average_loudness`
(already ~0..1) is used as an energy proxy when present. `tonal.key_strength`
(0..1) is persisted as `key_confidence` on the `Track` and surfaced to
`explain_set` so the LLM can hedge low-confidence key claims — this
key-detection is algorithmic, not ground truth.

MusicBrainz enforces a hard 1 request/second limit per client (503 above
that rate); calls to it go through a module-level throttle (a lock plus a
minimum interval between requests) that applies only to MusicBrainz, not to
the other providers, which remain fully concurrent. This means a large
batch of tracks that all need the MusicBrainz fallback pays roughly
1 second of extra latency per track for that fallback alone (see README).

Results cached in `tracks` table; cache hit skips external calls.
Unresolvable tracks are marked `unresolved`, excluded from the graph, and
reported in the response.

## 4. Data model (Supabase Postgres)

- `tracks`: id, artist, title, bpm, musical_key, camelot, energy, duration_s,
  tags jsonb, source, key_confidence, fetched_at. Unique on (artist, title)
  normalized. `key_confidence` is only ever set when `camelot` came from
  AcousticBrainz (its `tonal.key_strength`); null otherwise.
- `sets`: id, request jsonb (the raw request, including the truncation
  `notice` if any), result jsonb (the full `SetResult`), created_at. Stored
  so the site demo can replay real past sets.

Phase 2 adds pgvector tables (pitch corpus embeddings).

## 5. API (FastAPI)

- `POST /sets` — body: track list + preferences; responds with SSE stream of
  `progress` events, one per graph node (bare node name as the data, e.g.
  `parse_input`, `enrich_tracks`, `build_transition_graph`, `find_set_path`,
  `explain_set`), ending with a `result` event carrying the `SetResult`.
- `GET /sets/{id}` — replay a stored set.
- `GET /health`.
- CORS restricted to the portfolio site domain.

## 6. Errors & observability

- Every external call: 10s timeout, 2 retries with exponential backoff.
- LLM nodes validate output against Pydantic; one repair retry on failure.
- Langfuse callback handler attached to the whole graph run; node-level spans.
- A set is still produced when some tracks fail enrichment; failures are
  explained in the response, never silently dropped.

## 7. Testing

- Unit (no network, no LLM): Camelot neighbor math, BPM window, edge scoring,
  beam search on fixture tracks with known best path.
- Integration: enrichment cascade against mocked HTTP APIs (respx/httpx mock),
  cache hit behavior.
- LLM nodes: contract tests with recorded/stubbed responses only.

## 8. Phase 2 roadmap (not in MVP)

- Release Assistant on deepagents: researcher / positioning / copywriter
  subagents.
- RAG on pgvector: corpus of pitch examples and label/playlist guides —
  this is where RAG is actually justified (decided against RAG in MVP:
  harmonic mixing rules are deterministic and need no retrieval).
- Own MCP server exposing Track Intelligence tools.
- "Promo set around your release" — combined mode: build a set of similar
  tracks with the user's release at the peak.

## 9. Out of scope

- Audio file analysis (no filesystem/audio dependency by design).
- Spotify API (audio-features closed for new apps since Nov 2024).
- Auth/multi-user; the API is a public portfolio demo. `POST /sets` has an
  in-process, per-client-host rate limit (5 requests/minute); the key is the
  raw transport peer address, so behind a non-loopback edge proxy every user
  shares one bucket unless the server runs with `--forwarded-allow-ips`.
