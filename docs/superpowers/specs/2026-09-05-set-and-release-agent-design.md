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
- Music data (all free): Deezer → GetSongBPM → audio analysis of Deezer
  preview clips → MusicBrainz/AcousticBrainz (BPM/key cascade), Last.fm
  (tags, similar artists)

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
Per track: Deezer (no key needed) → GetSongBPM → audio analysis of the
Deezer preview clip → MusicBrainz/AcousticBrainz; tags/genre from Last.fm.
The cascade stops as soon as both `bpm` and `camelot` are known, so each
later provider is only consulted when the earlier ones didn't together
supply both.

**Audio analysis (`src/musicagent/audio.py`)** exists because the metadata
providers above cover mainstream/catalogued releases well but miss
underground ones almost entirely: measured on a real 15-track underground
techno playlist, MusicBrainz/AcousticBrainz resolved 0/15 and Deezer's own
metadata supplied bpm for 7 and never a key — but 13/15 of those tracks have
a public 30-second preview clip on Deezer, and analysing that clip supplies
key, bpm, and energy for any of them. It reuses the preview URL
(`hit["preview"]`) already present on the Deezer `/search` hit `_deezer`
fetches, rather than searching Deezer a second time. Per track it costs
roughly ~0.75s (download + ffmpeg decode + essentia analysis, measured) —
well under the MusicBrainz throttle's 1.1s/track alone — and Essentia's
`KeyExtractor`/`RhythmExtractor2013` supply the key/bpm; `key_confidence` is
set from Essentia's key strength the same way it is for AcousticBrainz.

**Energy is measured from this same clip, not read from Deezer's replay
gain.** `analyze_preview` combines two Essentia readings into a heuristic
0..1 energy score: integrated loudness (`LoudnessEBUR128`, EBU R128,
normalised from an LUFS range calibrated by measurement — see
`src/musicagent/audio.py` module constants) weighted 0.6, and onset rate
(`OnsetRate`, onsets/second, capped and normalised the same way) weighted
0.4. This is a heuristic proxy for perceived intensity, not a physical
quantity — the floor/ceiling/cap/weights were chosen by measuring both raw
features across 8 real tracks and picking values that gave a sensible
spread (0.49-0.74) rather than clustering near 0 or 1. It is a direct
measurement of the actual clip, so in the enrichment merge it wins for
`energy` specifically regardless of provider order (unlike `bpm`/`camelot`,
which stay strict first-writer-wins — see below); Deezer's `gain` field
(replay gain, `(gain + 20) / 20`, clamped) is used only as a fallback when
audio analysis doesn't run or fails, and AcousticBrainz's
`lowlevel.average_loudness` remains the last-resort energy proxy below.
Deezer returns `gain = 0` when loudness is unknown (the same sentinel
convention as `bpm = 0`), so that case is treated as no energy contribution,
not as the real (maximum-energy) value 0 dB would imply.

Both `essentia` (the DSP/ML library doing the analysis) and the `ffmpeg`
binary it depends on for mp3 decoding are optional at runtime: `essentia` is
an optional dependency (the `audio` extra, not installed by default — it
costs ~190MB RSS just to import) and `ffmpeg` cannot be installed via pip at
all. If either is missing, the provider degrades to contributing nothing
(logged once, not per track) rather than crashing — a deployment without
them runs on the metadata providers alone. Essentia's own key detection is
roughly 70-80% accurate (same caveat as AcousticBrainz below), which is why
`key_confidence` is surfaced rather than treating the key as ground truth.
Note: Deezer's terms of use don't explicitly address programmatically
analysing preview clips (as opposed to just linking/playing them); this MVP
treats that as acceptable for a non-commercial portfolio demo, not as a
reviewed legal position.

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
  audio analysis (Essentia's key strength) or AcousticBrainz (its
  `tonal.key_strength`); null otherwise.
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

- User-uploaded audio file analysis. The one audio dependency the MVP does
  have is narrow and optional: analysing Deezer's own public 30-second
  preview clips as an enrichment fallback (§3, `src/musicagent/audio.py`),
  gated behind the optional `audio` extra and a runtime `ffmpeg` check so a
  deployment without them still runs. There is no upload path, no storage of
  audio, and no analysis of anything the user provides directly.
- Spotify API (audio-features closed for new apps since Nov 2024).
- Auth/multi-user; the API is a public portfolio demo. `POST /sets` has an
  in-process, per-client-host rate limit (5 requests/minute); the key is the
  raw transport peer address, so behind a non-loopback edge proxy every user
  shares one bucket unless the server runs with `--forwarded-allow-ips`.
