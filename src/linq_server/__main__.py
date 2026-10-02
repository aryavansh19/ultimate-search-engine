"""Entry point: `python -m linq_server`.

Binds to 127.0.0.1 by default. Exposing this on 0.0.0.0 without credentials would hand
anyone who can reach the port the ability to spend your Gemini quota, so the default is
deliberately the safe one and going public is an explicit choice.
"""

from __future__ import annotations

import argparse
import logging
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="LinQ enrichment + embedding service")
    parser.add_argument("--host", default=os.getenv("LINQ_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("LINQ_PORT", "8100")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default=os.getenv("LINQ_LOG_LEVEL", "info"))
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if args.host != "127.0.0.1" and not (
        os.getenv("SUPABASE_JWT_SECRET") or os.getenv("LINQ_API_TOKEN")
        or os.getenv("APP_ACCESS_TOKEN")
    ):
        raise SystemExit(
            f"Refusing to bind {args.host} with no credentials configured.\n"
            "Set SUPABASE_JWT_SECRET (preferred) or LINQ_API_TOKEN first."
        )

    import uvicorn

    uvicorn.run(
        "linq_server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
