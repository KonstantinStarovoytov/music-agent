import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from musicagent.db import SetStore, TrackCache, get_engine, init_db
from musicagent.graph import build_graph, get_langfuse_handler
from musicagent.llm import LLMOutputError


class SetIn(BaseModel):
    text: str


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
                async for update in graph.astream(
                    {"text": body.text},
                    config={"callbacks": get_langfuse_handler()},
                ):
                    node, out = next(iter(update.items()))
                    state.update(out)
                    yield {"event": "progress", "data": node}
            except LLMOutputError as exc:
                yield {"event": "error", "data": str(exc)}
                return

            result = state["result"]
            set_id = store.save({"text": body.text}, result)
            yield {
                "event": "result",
                "data": json.dumps(
                    {"set_id": set_id, "result": json.loads(result.model_dump_json())}
                ),
            }

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
