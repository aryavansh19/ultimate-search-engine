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

Run `python -m linq_server` for the authenticated enrichment API. In production set
`SUPABASE_URL=https://<project-ref>.supabase.co`; signed-in user JWTs are verified locally
against Supabase's public JWKS endpoint. Issuer, `authenticated` audience/role, expiry, and
UUID subject are all required. Signing-key rotation is picked up through `kid` discovery.

`LINQ_API_TOKEN` remains available for server-side curl and development use. Never bundle
that shared token, a Supabase secret key, or `GEMINI_API_KEY` in a mobile application. Legacy
HS256 projects can set `SUPABASE_JWT_SECRET`, though asymmetric signing keys are preferred.
The `/health` response reports which authentication paths are configured without exposing
credentials.

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
