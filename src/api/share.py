"""Expose the locally-running API through a Cloudflare quick tunnel.

The backend stays on your machine. `cloudflared` makes an outbound connection to Cloudflare
and forwards traffic back to 127.0.0.1, so nothing needs port forwarding, a static IP, or a
firewall change -- and the server itself keeps listening only on loopback.

Why a token is mandatory here rather than optional: a tunnel URL is a public internet
address. Without auth, anyone holding it can read the entire library, delete from it, and
spend your Gemini and OpenRouter quota. The `POST /api/items` endpoint also fetches arbitrary
URLs from your home connection, which is not something to leave open. So this refuses to
start unless a token exists, and generates one if you have not set one.

Quick tunnels are throwaway by nature: the hostname changes every run and Cloudflare offers no
uptime promise. That is the right trade for handing a friend a link for an afternoon.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

WINDOWS_GUESSES = (
    r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
    r"C:\Program Files\cloudflared\cloudflared.exe",
)


def find_cloudflared(explicit: str | None = None) -> str | None:
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("cloudflared")
    if found:
        return found
    for guess in WINDOWS_GUESSES:
        if Path(guess).is_file():
            return guess
    return None


def ensure_token(env_path: Path = Path(".env")) -> tuple[str, bool]:
    """Return an access token, creating and persisting one if absent.

    Written into .env rather than only printed so the same link keeps working across server
    restarts -- a token that changes every run means re-sending the link every time.
    """
    from .security import AccessPolicy, generate_token

    existing = AccessPolicy.from_env().token
    if existing:
        return existing, False

    token = generate_token()
    line = f"\n# Shared-access token for tunnelled sessions (generated automatically).\nAPP_ACCESS_TOKEN={token}\n"
    if env_path.exists():
        env_path.write_text(env_path.read_text(encoding="utf-8") + line, encoding="utf-8")
    else:
        env_path.write_text(line.lstrip("\n"), encoding="utf-8")
    return token, True


def start_tunnel(port: int, binary: str) -> tuple[subprocess.Popen, str | None]:
    """Launch cloudflared and wait for it to report its public hostname."""
    proc = subprocess.Popen(
        [binary, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    url: list[str] = []

    def reader() -> None:
        # cloudflared prints the hostname to stderr as part of a banner, so both streams are
        # merged above and scanned rather than parsed.
        for line in proc.stdout or []:
            match = TUNNEL_URL.search(line)
            if match and not url:
                url.append(match.group(0))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()

    deadline = time.monotonic() + 45
    while time.monotonic() < deadline and not url:
        if proc.poll() is not None:
            return proc, None
        time.sleep(0.4)
    return proc, url[0] if url else None


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="share", description="Share the local server through a Cloudflare quick tunnel."
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cloudflared", help="path to the cloudflared binary")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    binary = find_cloudflared(args.cloudflared)
    if not binary:
        print("cloudflared not found. Install it, or pass --cloudflared <path>.")
        print("  winget install --id Cloudflare.cloudflared")
        return 1

    # Confirm something is actually serving, otherwise the tunnel resolves to nothing and the
    # failure looks like a tunnel problem.
    import httpx

    try:
        health = httpx.get(f"http://127.0.0.1:{args.port}/health", timeout=5)
        serving = health.status_code == 200
    except Exception:
        serving = False
    if not serving:
        print(f"Nothing is answering on http://127.0.0.1:{args.port}")
        print("  Start it first:  python -m api --port {}".format(args.port))
        return 1

    token, created = ensure_token()
    if created:
        print("Generated an access token and saved it to .env")
        print("RESTART the server so it picks the token up, then run this again.\n")
        print("  python -m api --port {}".format(args.port))
        return 2

    print(f"cloudflared: {binary}")
    print("opening tunnel...")
    proc, url = start_tunnel(args.port, binary)
    if not url:
        print("Tunnel did not report a URL. cloudflared output above may explain why.")
        proc.terminate()
        return 1

    share = f"{url}/?k={token}"
    print()
    print("=" * 74)
    print("  Send your friend this link (the ?k=... part is the access token):")
    print()
    print(f"  {share}")
    print()
    print("=" * 74)
    print("  Backend stays on this machine. Keep this window and the server running.")
    print("  The token is stripped from their address bar after first load.")
    print("  Ctrl+C here closes the tunnel; the link dies with it.")
    print()

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nclosing tunnel...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
