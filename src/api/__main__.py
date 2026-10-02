"""Run the API and web UI.

    python -m api                 http://127.0.0.1:8000
    python -m api --port 9000

Binds to loopback by default. The server has no authentication and fetches
caller-supplied URLs, so exposing it on a network interface requires adding auth first --
see the security note in app.py.
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="api")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="restart on source changes (development). Without this, edits to the "
        "pipeline have no effect until the server is restarted by hand -- a running "
        "server keeps serving the code it started with, which silently produces stale "
        "behaviour that looks like the fix did not work.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        logging.warning(
            "binding to %s exposes an unauthenticated API that fetches arbitrary URLs "
            "on request; add authentication before doing this on an untrusted network",
            args.host,
        )

    import uvicorn

    from . import __version__

    logging.info("link-memory api %s starting on %s:%s", __version__, args.host, args.port)
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
