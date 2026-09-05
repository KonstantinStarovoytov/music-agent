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
