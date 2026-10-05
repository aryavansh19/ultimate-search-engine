# Hugging Face Spaces (Docker SDK) image for the LinQ enrichment service.
#
# HF Spaces requires the app to listen on port 7860 and runs the container as a
# non-root user (uid 1000). The SQLite caches are pointed at /tmp because the free
# tier filesystem is ephemeral — the enrichment cache is content-addressed and
# rebuilds on demand, so losing it between restarts is harmless.

FROM python:3.11-slim

# ffmpeg is optional (used to compress video before analysis); keeping it in makes
# the "Analyse video" path cheaper. Remove it to shrink the image if you only ever
# send text/links.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces runs as uid 1000. Create a matching user so cache dirs are writable.
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first for layer caching.
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# Writable, ephemeral cache location on the HF free tier.
ENV EXTRACTOR_CACHE_PATH=/tmp/linq_cache/extractor_cache.sqlite3

# Auth is enforced by default; the Space must set LINQ_API_TOKEN (or
# SUPABASE_JWT_SECRET) and GEMINI_API_KEY as Space secrets.
ENV LINQ_REQUIRE_AUTH=1

USER appuser

EXPOSE 8000

# Most hosts (Render, Fly, Railway) inject $PORT; fall back to 8000 locally.
CMD ["sh", "-c", "uvicorn linq_server.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
