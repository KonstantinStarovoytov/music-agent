import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from musicagent.db import SetStore, TrackCache, get_engine, init_db
from musicagent.graph import build_graph, get_langfuse_handler
from musicagent.llm import LLMOutputError

logger = logging.getLogger(__name__)

# Fixed, generic message shown to anonymous callers when the LLM cannot produce
# a valid set from the request. Never includes the underlying exception text
# (which may contain provider/auth/rate-limit details) -- that is logged
# server-side instead. See task-9 review finding 1.
LLM_ERROR_MESSAGE = (
    "Could not build a set from this request -- try naming the tracks as "
    "'artist - title'."
)

# Fixed, generic message for any *other* unhandled exception mid-run (network
# failure in enrichment, a bug in a node, a timeout, a DB write blip, etc).
# Never includes the underlying exception text -- that is logged server-side
# instead. See task-10 review residual: only LLMOutputError was caught, so
# any other exception aborted the stream with no event at all.
GENERIC_ERROR_MESSAGE = "Something went wrong while building the set. Please try again."

# Request-body cap (finding 2a): reject oversized bodies with a 422 before any
# LLM/enrichment work starts.
MAX_TEXT_LENGTH = 4000

# Overall wall-clock budget for one /sets run (parse + enrich + path + explain).
# Enrichment no longer needs the MusicBrainz-throttle-dominated margin this
# used to be sized for: audio analysis of Deezer preview clips (see
# musicagent/audio.py) is now consulted before MusicBrainz/AcousticBrainz and
# resolves most tracks in ~1s each (download + decode + analyse), so
# MusicBrainz is only a fallback for tracks Deezer never found at all.
# Sized as enrichment's own worst case (ENRICH_DEADLINE_S = 55s, see
# enrichment.py for that arithmetic) plus ~2x for the two LLM round trips
# (parse_input, explain_set) at ~15s each in the worst case (slow provider,
# one repair retry) plus ~20s of slack for the pure-Python graph/path nodes
# and general overhead: 55 + 30 + 20 = 105s, rounded up to 110s. This is the
# outer backstop so a run can never hang the SSE connection open indefinitely.
# Note some hosting proxies cut idle streams earlier than this — progress
# events keep the stream from going idle.
OVERALL_DEADLINE_S = 110.0

# Rate limit for the public, unauthenticated POST /sets endpoint: each track
# list triggers several outbound HTTP calls plus at least one LLM call, so
# this is deliberately tight.
#
# Keyed on request.client.host, which is the raw transport peer address, not
# necessarily the end user's IP: behind a PaaS edge proxy that isn't on
# 127.0.0.1, uvicorn will not rewrite this to the real client IP unless it is
# run with --forwarded-allow-ips, so every user would share one bucket (the
# proxy's address) until that flag is set.
RATE_LIMIT_MAX_REQUESTS = 5
RATE_LIMIT_WINDOW_S = 60.0


class _RateLimiter:
    """Simple in-process sliding-window rate limiter keyed by an arbitrary
    string (here, the caller's host). Per-process only -- a multi-instance
    deployment would need a shared store -- which is fine for this single
    small portfolio deployment."""

    def __init__(self, max_requests: int, window_s: float):
        self.max_requests = max_requests
        self.window_s = window_s
        self._hits: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        # Sweep every key's window, not just this one, and drop any deque
        # that ends up empty. Without this, every distinct client host ever
        # seen keeps a (permanently empty, once its window passes) entry in
        # this dict for the rest of the process lifetime -- a slow, unbounded
        # memory leak for a long-running process. Cheap: this map only ever
        # holds as many keys as distinct hosts seen within the last window.
        for other_key, other_hits in list(self._hits.items()):
            while other_hits and now - other_hits[0] > self.window_s:
                other_hits.popleft()
            if not other_hits:
                del self._hits[other_key]

        hits = self._hits[key]
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True


class SetIn(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)


def _ref(ref) -> dict:
    return {"artist": ref.artist, "title": ref.title}


def progress_snapshot(node: str, state: dict) -> dict:
    """Client-safe projection of what `node` just produced, for the live run
    visualisation on the portfolio site (spec section 5). Only ever derived
    from the accumulated graph state -- never prompts or provider payloads.
    Indices in `edges`/`order` refer to the `enrich_tracks.tracks` list."""
    if node == "parse_input":
        req = state["request"]
        return {
            "tracks": [_ref(t) for t in req.tracks],
            "energy_shape": req.energy_shape,
            "duration_min": req.duration_min,
        }
    if node == "enrich_tracks":
        return {
            "tracks": [
                {**_ref(t.ref), "bpm": t.bpm, "camelot": t.camelot, "energy": t.energy}
                for t in state["tracks"]
            ],
            "unresolved": [
                {"artist": u.artist, "title": u.title, "reason": u.reason}
                for u in state["unresolved"]
            ],
        }
    if node == "build_transition_graph":
        return {
            "edges": [
                {"a": e.a, "b": e.b, "score": round(e.score, 3)}
                for e in state["transition_graph"].edges
            ]
        }
    if node == "find_set_path":
        index = {(t.ref.artist, t.ref.title): i for i, t in enumerate(state["tracks"])}
        return {
            "order": [index[(t.ref.artist, t.ref.title)] for t in state["path"].tracks]
        }
    return {}


def create_app(engine=None, llm=None) -> FastAPI:
    app = FastAPI(title="Set & Release Agent")

    site_origin = os.environ.get("SITE_ORIGIN")
    if not site_origin:
        logger.warning(
            "SITE_ORIGIN is not set; CORS will allow no cross-origin requests "
            "(closed by default). Set SITE_ORIGIN to the portfolio site's "
            "origin to allow it."
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[site_origin] if site_origin else [],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    eng = engine if engine is not None else get_engine()
    init_db(eng)
    cache, store = TrackCache(eng), SetStore(eng)
    graph = build_graph(cache=cache, llm=llm)
    rate_limiter = _RateLimiter(RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_S)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/sets")
    async def create_set(body: SetIn, request: Request):
        client_host = request.client.host if request.client else "unknown"
        if not rate_limiter.allow(client_host):
            raise HTTPException(
                status_code=429,
                detail="Too many requests -- please wait a minute and try again.",
            )

        async def events():
            state: dict = {}
            try:
                try:
                    async with asyncio.timeout(OVERALL_DEADLINE_S):
                        async for update in graph.astream(
                            {"text": body.text},
                            config={"callbacks": get_langfuse_handler()},
                        ):
                            node, out = next(iter(update.items()))
                            state.update(out)
                            yield {
                                "event": "progress",
                                "data": json.dumps(
                                    {
                                        "node": node,
                                        "data": progress_snapshot(node, state),
                                    }
                                ),
                            }

                    # Serializing the result and writing it to the DB is kept
                    # inside this same try so a KeyError (missing "result") or
                    # a DB blip on save() still produces a terminal SSE error
                    # event instead of silently killing the stream.
                    result = state["result"]
                    notice = state.get("notice")
                    set_id = store.save({"text": body.text, "notice": notice}, result)
                    payload = {
                        "set_id": set_id,
                        "result": json.loads(result.model_dump_json()),
                    }
                    if notice:
                        payload["notice"] = notice
                    yield {"event": "result", "data": json.dumps(payload)}
                except LLMOutputError:
                    # Log the real exception (with traceback) server-side only;
                    # the caller gets a fixed, generic message so internal
                    # provider/auth/rate-limit error text never leaks to an
                    # anonymous client. See task-9 review finding 1.
                    logger.exception("LLM output error while building set")
                    yield {"event": "error", "data": LLM_ERROR_MESSAGE}
                    return
                except TimeoutError:
                    logger.exception(
                        "Timed out building set after %.0fs", OVERALL_DEADLINE_S
                    )
                    yield {"event": "error", "data": GENERIC_ERROR_MESSAGE}
                    return
                except Exception:
                    # Any other unhandled exception mid-run (enrichment
                    # network failure, a bug in a node, a DB blip on save,
                    # etc) must not abort the stream silently. Log with
                    # traceback server-side and emit the same kind of generic
                    # error event as above, with a distinct message, so the
                    # client always gets a terminal event instead of a
                    # dropped connection. See task-10 review residual.
                    logger.exception("Unhandled exception while building set")
                    yield {"event": "error", "data": GENERIC_ERROR_MESSAGE}
                    return
            finally:
                if "result" not in state:
                    logger.info(
                        "SSE stream for /sets ended before producing a result "
                        "(client disconnect or unhandled error)"
                    )

        return EventSourceResponse(events())

    @app.get("/sets/{set_id}")
    async def get_set(set_id: str):
        row = store.load(set_id)
        if not row:
            raise HTTPException(404)
        return {"set_id": set_id, "request": row["request"], "result": row["result"]}

    return app


def get_app() -> FastAPI:
    """Lazy factory for prod: `uvicorn --factory musicagent.api:get_app`.
    Builds the app from environment on first call so the module stays
    importable (and side-effect free) with no env vars set.

    Loads `.env` here rather than at import time: tests import this module
    with a deliberately empty environment, and only the real entrypoint
    should pick up developer credentials from disk. Real environment
    variables win over the file.
    """
    from dotenv import load_dotenv

    load_dotenv(override=False)
    return create_app()
