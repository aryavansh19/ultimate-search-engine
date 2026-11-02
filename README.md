# Ultimate Search Engine (Link Memory)

A personal link-memory and semantic search engine that extracts, enriches, indexes, and searches through content saved from the web and social platforms (Instagram, TikTok, YouTube, web articles, etc.).

## 🌟 Features

- **Multi-Source Extraction**: Robust extractor cascade supporting client payloads, yt-dlp, OpenGraph, and managed APIs.
- **AI-Powered Enrichment**: Enriches extracted content using multimodal LLMs (Gemini, OpenAI-compatible APIs, heuristic fallbacks) for tagging, categorization, summaries, and taxonomy mapping.
- **Hybrid Search**: Semantic vector search combined with keyword retrieval and reciprocal rank fusion (RRF) for high-accuracy recall.
- **Modern Web Interface**: Responsive web UI for searching through links and visualizing collections.
- **Wrapped Insights**: Year-in-review / collection summary visualization (`wrapped.html`).
- **RESTful API**: Fast and modular FastAPI backend.

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Virtual environment (`venv` or `conda`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aryavansh19/ultimate-search-engine.git
   cd ultimate-search-engine
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Running the Application

- **Start the API & Web Server:**
  ```bash
  python -m api
  ```
  Access the web interface at `http://127.0.0.1:8000`.

- **Run Extractor directly:**
  ```bash
  python -m extractor <url>
  ```

- **Run Search CLI:**
  ```bash
  python -m search "your query here"
  ```

### LinQ mobile enrichment API

Run `python -m linq_server` for the authenticated enrichment API. For one Supabase project,
set `SUPABASE_URL=https://<project-ref>.supabase.co`. To let one service accept isolated
Testing and Production users, set a comma-separated allowlist such as
`SUPABASE_URLS=https://<testing-ref>.supabase.co,https://<production-ref>.supabase.co`.
Signed-in user JWTs are verified locally against the selected project's public JWKS
endpoint. The unverified issuer can only select an exact configured URL; the signature,
issuer, `authenticated` audience/role, expiry, and UUID subject are then fully verified.
Signing-key rotation is picked up through `kid` discovery.

`LINQ_API_TOKEN` remains available for server-side curl and development use. Never bundle
that shared token, a Supabase secret key, or `GEMINI_API_KEY` in a mobile application. Legacy
HS256 projects can set `SUPABASE_JWT_SECRET`, though asymmetric signing keys are preferred.
A single legacy secret cannot be used with multiple projects. The `/health` response
reports the number of allowlisted issuers without exposing credentials or project URLs.

### Existing collection suggestions

`POST /v1/analyze` accepts an optional `suggest_collection: true` flag. The default is
false: older clients and manual saves do not trigger a collection lookup or an additional
model call. Send the signed-in user's Supabase access token as usual. Shared development
tokens cannot request personal collections (HTTP 403). The request must still contain
the link URL; bookmark-ID-only requests and background jobs are not implemented here.

Configure `LINQ_SUPABASE_PUBLISHABLE_KEYS` in Render's Environment settings as a JSON
object mapping each exact allowlisted project URL to that project's publishable key:

```json
{
  "https://testing-ref.supabase.co": "sb_publishable_testing",
  "https://production-ref.supabase.co": "sb_publishable_production"
}
```

For a single allowlisted project, `SUPABASE_PUBLISHABLE_KEY` (or legacy
`SUPABASE_ANON_KEY`) is also accepted. A single key is never reused across multiple
projects. No secret/service-role key is needed or accepted. The collection lookup uses
the **verified** token's issuer to select the project and its subject to filter
`public.collections.user_id`. The same user JWT is forwarded to the Data API, so RLS
still applies. It reads only `id`, `name`, and `user_id`, using ordered keyset pages.
Collection access needs an authenticated owner SELECT policy and the usual table grants.

Example request (with the user's bearer token in the Authorization header):

```json
{
  "url": "https://example.com/kyoto-guide",
  "embed": false,
  "suggest_collection": true
}
```

The existing analysis response gains this additive field:

```json
{
  "collection_suggestion": {
    "status": "selected",
    "collection_id": "00000000-0000-0000-0000-000000000010",
    "collection_name": "Travel",
    "provider": "gemini",
    "model": "<configured ENRICH_TEXT_MODEL or fallback>",
    "input_tokens": 100,
    "output_tokens": 12,
    "cost_usd": 0.0,
    "duration_ms": 500,
    "reason": null
  }
}
```

The usage values above are illustrative. Classification usage/cost is reported inside
`collection_suggestion`; existing top-level enrichment accounting is unchanged. This is
one additional bounded **text-only** classification pass over the generated summary,
description, tags, and up to 200 collection names (160 characters per name in the prompt).
It does not send media again. It uses the existing Gemini model/fallback configuration,
quota governor and pricing estimates, with one attempt per model and at most 20 seconds
per HTTP attempt. Catalog reads are capped at five 10-second pages. Collection names are sent to Gemini
only when the client explicitly opts in. The output ID must belong to the fetched list;
the returned name is copied from that list, never trusted from model output.

Other statuses are `no_match`, `no_collections`, and `unavailable`, all with null collection
ID/name. Missing configuration, denied lookups, oversized candidate lists, degraded content,
and provider failures never turn a successful enrichment into a failed save. An oversized
list is not silently truncated into an inaccurate recommendation. With the flag omitted,
`collection_suggestion` is null. No collection is created, and **no database writes or
bookmark assignments are performed by this endpoint**.

User-specific choices are never written to the content-addressed enrichment cache. Two
users saving the same URL therefore share the content enrichment where supported but
receive independent collection decisions. This also avoids invalidating expensive video
analysis when a user renames a collection.

The iOS integration must send the flag only for Auto saves, decode this field, and apply
the chosen ID only while Auto remains enabled and no later manual choice has superseded
the request. Check that the bookmark/collection still exists and belongs to the current
account. An offline-only new collection is not eligible until it has synced. Suggestions
are a snapshot, not an authorization to overwrite later edits. This change preserves the
existing synchronous HTTP response contract; durable asynchronous jobs and server-side
assignment remain a separate integration. Render's free ephemeral disk is not a durable
job queue.

Run the offline regression suite without Supabase/Gemini credentials:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Set `LINQ_LOG_PAYLOADS=1` to print sanitized request JSON, extraction/enrichment stage
summaries, and the complete response JSON with a shared trace ID. Authorization headers,
API keys, cookies, and rendered HTML are never logged. Response payloads contain bookmark
content, so disable this setting when detailed diagnostics are no longer needed.

## 📁 Project Structure

```
├── deploy/            # Deployment configurations & artifacts
├── samples/           # Sample payload fixtures
├── src/
│   ├── api/           # FastAPI backend & routes
│   ├── enrichment/    # Multimodal LLM enrichment & metadata extraction
│   ├── extractor/     # Content extraction cascade (yt-dlp, OpenGraph, etc.)
│   ├── linq_server/   # Sync and server utilities
│   └── search/        # Vector embeddings & search indexer
├── web/               # Frontend web applications
├── pyproject.toml     # Packaging & project metadata
└── requirements.txt   # Core Python dependencies
```

## 📄 License

MIT
