# music-agent — «Set & Release Agent»

AI-агент (пет-проект для портфолио vojt): по списку треков строит гармоничные DJ-сеты
(Camelot wheel, BPM, energy) и релизные стратегии (похожие артисты, плейлисты, питчи).

## Правила
- **Spec-driven**: `spec.md` — источник правды. Сначала спека, потом код (см. skill spec-sync).
- Стек (MVP): Python 3.13, uv, LangGraph + deepagents, Langfuse (трейсинг), Supabase Postgres.
  pgvector (данные + RAG) — phase 2, ещё не используется.
- LLM: OpenAI (или Luna, OpenAI-совместимый гейтвей, через `OPENAI_BASE_URL`). Свапается через LangChain. Ключи в `.env` (не трогать). Fallback на Gemini / Groq free tier — phase 2 (пакеты в опциональной группе `phase2`, кода фолбэка пока нет).
- Детерминированное (Camelot-соседи, BPM ±6%, energy curve) — чистый Python + unit-тесты, не LLM.
- Внешние API (MVP): Deezer, Last.fm, GetSongBPM — все вызовы с timeout и retry.
  MusicBrainz/AcousticBrainz — phase 2 fallback, ещё не используется.

## Команды
- `uv run pytest` — тесты
- `uv run ruff check .` — линт (формат гоняется хуком автоматически)
- `uv sync` — установка зависимостей
- `uv run uvicorn --factory musicagent.api:get_app --port 8123` — запуск API
  (нужен `DATABASE_URL`; для локального запуска без Postgres подойдёт
  `DATABASE_URL=sqlite:////tmp/musicagent.db`). PYTHONPATH не нужен —
  пакет `musicagent` ставится в editable-режиме через hatchling
  (`[tool.hatch.build.targets.wheel]` в `pyproject.toml`), так что `uvicorn`
  видит его как обычный установленный модуль. (`pythonpath = ["src"]` —
  отдельная настройка, она нужна только `pytest`, не `uvicorn`.)

## После изменения графа
Запускай субагента `graph-reviewer` для ревью структуры графа.
