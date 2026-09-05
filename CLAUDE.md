# music-agent — «Set & Release Agent»

AI-агент (пет-проект для портфолио vojt): по списку треков строит гармоничные DJ-сеты
(Camelot wheel, BPM, energy) и релизные стратегии (похожие артисты, плейлисты, питчи).

## Правила
- **Spec-driven**: `spec.md` — источник правды. Сначала спека, потом код (см. skill spec-sync).
- Стек: Python 3.13, uv, LangGraph + deepagents, Langfuse (трейсинг), Supabase Postgres + pgvector (данные + RAG).
- LLM: основной — OpenAI (или Luna, OpenAI-совместимый гейтвей, через `OPENAI_BASE_URL`); fallback — Gemini / Groq free tier. Все свапаются через LangChain. Ключи в `.env` (не трогать).
- Детерминированное (Camelot-соседи, BPM ±6%, energy curve) — чистый Python + unit-тесты, не LLM.
- Внешние API: MusicBrainz/AcousticBrainz, Deezer, Last.fm, GetSongBPM — все вызовы с timeout и retry.

## Команды
- `uv run pytest` — тесты
- `uv run ruff check .` — линт (формат гоняется хуком автоматически)
- `uv sync` — установка зависимостей
- `uv run uvicorn --factory musicagent.api:get_app --port 8123` — запуск API
  (нужен `DATABASE_URL`; для локального запуска без Postgres подойдёт
  `DATABASE_URL=sqlite:////tmp/musicagent.db`). PYTHONPATH не нужен —
  `pythonpath = ["src"]` уже в `pyproject.toml`.

## После изменения графа
Запускай субагента `graph-reviewer` для ревью структуры графа.
