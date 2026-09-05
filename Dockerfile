# Minimal image for free-tier hosts (Render/Fly/Railway) that build from a
# Dockerfile. Exists specifically because ffmpeg -- needed to decode Deezer
# preview clips for audio analysis, see src/musicagent/audio.py -- can't come
# from pip; everything else here is uv installing the project normally.
#
# Pinned to linux/amd64: those hosts' free tiers run x86_64, and essentia's
# published wheels don't cover linux/arm64 at all (only x86_64 and macOS) --
# building on an arm64 machine (e.g. Apple Silicon) without this would fail
# to resolve essentia for the target platform.
FROM --platform=linux/amd64 python:3.13-slim

# ffmpeg: decodes preview mp3 clips to wav for essentia (audio analysis
# provider). Installed via apt since it has no PyPI wheel.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# uv manages the venv/lockfile for this project; installed via its official
# static binary rather than pip to keep this stage small and fast.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first so `uv sync` is cached across rebuilds that
# only change application code.
COPY pyproject.toml uv.lock ./
RUN uv sync --extra audio --no-install-project --no-dev

COPY src ./src
RUN uv sync --extra audio --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8123

# DATABASE_URL, OPENAI_API_KEY, etc. are supplied by the host's environment
# config, not baked into the image -- see README's environment variable table.
#
# Shell form (not the JSON array) so ${PORT} expands: managed hosts inject the
# port to listen on rather than letting the image choose it (Cloud Run defaults
# to 8080, Render sets PORT), and a container that ignores it is treated as
# failing its health check. 8123 is only the local fallback.
#
# --forwarded-allow-ips: behind the host's edge proxy, request.client.host is
# the proxy, not the visitor, so without this every caller would share one
# bucket in the rate limiter (see RATE_LIMIT_* in api.py). "*" is right here
# because the only route to this container is through that trusted proxy.
# `exec` so uvicorn replaces the shell as PID 1 and receives SIGTERM directly:
# without it the shell holds PID 1, swallows the host's stop signal, and the
# container is SIGKILLed after the grace period instead of shutting down.
CMD ["sh", "-c", "exec uvicorn --factory musicagent.api:get_app --host 0.0.0.0 --port ${PORT:-8123} --forwarded-allow-ips='*'"]
