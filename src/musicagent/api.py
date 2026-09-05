import json
import logging
import os

from fastapi import FastAPI, HTTPException
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
# failure in enrichment, a bug in a node, etc). Never includes the underlying
# exception text -- that is logged server-side instead. See task-10 review
# residual: only LLMOutputError was caught, so any other exception aborted the
# stream with no event at all.
GENERIC_ERROR_MESSAGE = "Something went wrong while building the set. Please try again."

# Request-body cap (finding 2a): reject oversized bodies with a 422 before any
# LLM/enrichment work starts.
MAX_TEXT_LENGTH = 4000


class SetIn(BaseModel):
    text: str = Field(..., max_length=MAX_TEXT_LENGTH)


def create_app(engine=None, llm=None) -> FastAPI:
    app = FastAPI(title="Set & Release Agent")

    site_origin = os.environ.get("SITE_ORIGIN")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[site_origin] if site_origin else ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    eng = engine if engine is not None else get_engine()
    init_db(eng)
    cache, store = TrackCache(eng), SetStore(eng)
    graph = build_graph(cache=cache, llm=llm)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/sets")
    async def create_set(body: SetIn):
        async def events():
            state: dict = {}
            try:
                try:
                    async for update in graph.astream(
                        {"text": body.text},
                        config={"callbacks": get_langfuse_handler()},
                    ):
                        node, out = next(iter(update.items()))
                        state.update(out)
                        yield {"event": "progress", "data": node}
                except LLMOutputError:
                    # Log the real exception (with traceback) server-side only;
                    # the caller gets a fixed, generic message so internal
                    # provider/auth/rate-limit error text never leaks to an
                    # anonymous client. See task-9 review finding 1.
                    logger.exception("LLM output error while building set")
                    yield {"event": "error", "data": LLM_ERROR_MESSAGE}
                    return
                except Exception:
                    # Any other unhandled exception mid-run (enrichment
                    # network failure, a bug in a node, etc) must not abort
                    # the stream silently. Log with traceback server-side and
                    # emit the same kind of generic error event as above, with
                    # a distinct message, so the client always gets a
                    # terminal event instead of a dropped connection. See
                    # task-10 review residual.
                    logger.exception("Unhandled exception while building set")
                    yield {"event": "error", "data": GENERIC_ERROR_MESSAGE}
                    return

                result = state["result"]
                notice = state.get("notice")
                set_id = store.save({"text": body.text}, result)
                payload = {
                    "set_id": set_id,
                    "result": json.loads(result.model_dump_json()),
                }
                if notice:
                    payload["notice"] = notice
                yield {"event": "result", "data": json.dumps(payload)}
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
    """
    return create_app()
