import json
import os
import uuid

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine

from musicagent.models import SetResult, Track, TrackRef

metadata = MetaData()

tracks_table = Table(
    "tracks",
    metadata,
    Column("artist_key", String, primary_key=True),  # artist, stripped and lowercased
    Column("title_key", String, primary_key=True),  # title, stripped and lowercased
    Column("artist", String, nullable=False),
    Column("title", String, nullable=False),
    Column("bpm", Float, nullable=False),
    Column("camelot", String, nullable=False),
    Column("energy", Float, nullable=False),
    Column("duration_s", Integer),
    Column("tags", JSON),
    Column("source", String),
    Column("key_confidence", Float),
    Column("fetched_at", DateTime, server_default=func.now()),
)

sets_table = Table(
    "sets",
    metadata,
    Column("id", String, primary_key=True),
    Column("request", JSON, nullable=False),
    Column("result", JSON, nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
)


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or os.environ["DATABASE_URL"])


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


def _norm(value: str) -> str:
    return value.strip().lower()


def _upsert_insert(engine: Engine):
    """Return a dialect-aware insert() constructor that supports on_conflict_do_update."""
    if engine.dialect.name == "postgresql":
        from sqlalchemy.dialects.postgresql import insert

        return insert
    if engine.dialect.name == "sqlite":
        from sqlalchemy.dialects.sqlite import insert

        return insert
    raise NotImplementedError(
        f"upsert not implemented for dialect: {engine.dialect.name}"
    )


class TrackCache:
    def __init__(self, engine: Engine):
        self.engine = engine

    def get(self, ref: TrackRef) -> Track | None:
        with self.engine.connect() as c:
            row = (
                c.execute(
                    select(tracks_table).where(
                        tracks_table.c.artist_key == _norm(ref.artist),
                        tracks_table.c.title_key == _norm(ref.title),
                    )
                )
                .mappings()
                .first()
            )
        if not row:
            return None
        return Track(
            ref=TrackRef(artist=row["artist"], title=row["title"]),
            bpm=row["bpm"],
            camelot=row["camelot"],
            energy=row["energy"],
            duration_s=row["duration_s"],
            tags=row["tags"] or [],
            source=row["source"],
            key_confidence=row["key_confidence"],
        )

    def put(self, track: Track) -> None:
        values = {
            "artist_key": _norm(track.ref.artist),
            "title_key": _norm(track.ref.title),
            "artist": track.ref.artist,
            "title": track.ref.title,
            "bpm": track.bpm,
            "camelot": track.camelot,
            "energy": track.energy,
            "duration_s": track.duration_s,
            "tags": track.tags,
            "source": track.source,
            "key_confidence": track.key_confidence,
        }
        insert = _upsert_insert(self.engine)
        stmt = insert(tracks_table).values(**values)
        update_cols = {
            k: v for k, v in values.items() if k not in ("artist_key", "title_key")
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=[tracks_table.c.artist_key, tracks_table.c.title_key],
            set_=update_cols,
        )
        with self.engine.begin() as c:
            c.execute(stmt)


class SetStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def save(self, request_json: dict, result: SetResult) -> str:
        set_id = uuid.uuid4().hex
        with self.engine.begin() as c:
            c.execute(
                sets_table.insert().values(
                    id=set_id,
                    request=request_json,
                    result=json.loads(result.model_dump_json()),
                )
            )
        return set_id

    def load(self, set_id: str) -> dict | None:
        with self.engine.connect() as c:
            row = (
                c.execute(select(sets_table).where(sets_table.c.id == set_id))
                .mappings()
                .first()
            )
        return dict(row) if row else None
