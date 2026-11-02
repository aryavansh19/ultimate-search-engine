# Collection suggestion handoff — 2026-11-03

Work belongs to `deploy/render`, based on `9ef14cb`. The user explicitly selected this
Render deployment branch; do not move this change to `main` or merge branches implicitly.

The LinQ HTTP entry point is `src/linq_server/main.py`, with response contracts in
`models.py`. Optional `suggest_collection: true` requests now read the verified Supabase
user's existing collection IDs/names using their JWT and the matching project's public
key. `collections.py` performs the bounded owner-scoped lookup and a separate Gemini
text classification of the enrichment. It returns an existing ID/name or an explicit
no-match/unavailable state. It never creates collections or writes to Supabase.

Keep personal collection choices outside the shared content enrichment cache. Reuse
the current enrichment models/quota governor, but do not re-upload or reanalyze media
just to choose a collection. Shared development tokens cannot request personal data.

Configure `LINQ_SUPABASE_PUBLISHABLE_KEYS` in Render as documented in README. No key
values, browser sessions, cookies, or local `.env` files should be committed. Existing
clients remain compatible and do not invoke classification until they send the flag.

Outstanding client integration: decode `collection_suggestion`, persist/apply an Auto
assignment through normal sync, and preserve manual choices, account boundaries, and
bookmark/collection deletion checks. Durable server jobs are not included; the existing
HTTP endpoint remains synchronous. No iOS source or database schema was modified.

Verification: all 57 tests in the repository's offline unittest suite passed, including
21 collection lookup/selection/endpoint checks with mocked Supabase/Gemini transports.
Python compilation and whitespace checks passed. No live user session, paid model
invocation, live database change, or deployed
Render verification is part of these checks. The current checkout is isolated from the
older local backend checkout and its independent worktree.
