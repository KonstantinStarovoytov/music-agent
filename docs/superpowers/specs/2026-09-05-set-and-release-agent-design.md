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
| parse_input | LLM | free text → `SetRequest{tracks: list[TrackRef], duration_min?, energy_shape}` |
| enrich_tracks | code, parallel | `list[TrackRef]` → `list[Track]` (+ `unresolved: list[TrackRef]`) |
| build_transition_graph | pure Python | `list[Track]` → `TransitionGraph{edges: (a, b, score)}` |
| find_set_path | pure Python | `TransitionGraph` + `energy_shape` → `SetPath{ordered tracks, per-edge scores}` |
| explain_set | LLM | `SetPath` → `SetResult{transitions: [{from, to, explanation}], summary}` |

All contracts are Pydantic models in `src/musicagent/models.py`. State schema
in code must match this table (enforced by spec-sync skill).

### Deterministic rules (never LLM)
- Camelot compatibility: same key, ±1 on the wheel, relative major/minor.
- BPM window: ±6% (configurable constant).
- Edge score: weighted sum of key compatibility, BPM distance, energy delta
  vs. the target curve.
- Path search: greedy beam search over the transition graph targeting an
  energy shape (`build`, `peak_end`, `wave`); returns best-scoring
  Hamiltonian-ish path over resolved tracks (not required to include all).

### Enrichment cascade
Per track: Deezer (no key needed) → GetSongBPM; tags/genre from Last.fm.
Results cached in `tracks` table; cache hit skips external calls.
Unresolvable tracks are marked `unresolved`, excluded from the graph, and
reported in the response. MusicBrainz/AcousticBrainz as a further BPM/key
fallback is deferred to phase 2 (see section 8).

## 4. Data model (Supabase Postgres)

- `tracks`: id, artist, title, bpm, musical_key, camelot, energy, duration_s,
  tags jsonb, source, fetched_at. Unique on (artist, title) normalized.
- `sets`: id, request jsonb, track_order jsonb, explanations jsonb,
  created_at. Stored so the site demo can replay real past sets.

Phase 2 adds pgvector tables (pitch corpus embeddings).

## 5. API (FastAPI)

- `POST /sets` — body: track list + preferences; responds with SSE stream of
  progress events (`enriching 8/15`, `building graph`, `path found`) ending
  with the `SetResult`.
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

- MusicBrainz/AcousticBrainz as a further BPM/key fallback after
  Deezer → GetSongBPM.
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
- Auth/multi-user; the API is a public portfolio demo with rate limiting at
  the platform level.
