"""Check an Instagram cookies.txt BEFORE uploading it to Render.

    .venv\\Scripts\\python.exe scripts\\check_instagram_cookies.py <cookies.txt> [reel-url]

1. Validates the file is a Netscape cookies.txt with a live Instagram login
   (`sessionid`). Cookie VALUES are never printed -- only names and expiry.
2. Runs the server's real yt-dlp tier with the file against a reel and reports what it
   extracted. If this works here, the same file works on Render.

Exit code 0 = usable, 1 = not usable.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_URL = "https://www.instagram.com/reel/DeCxN2fS6K1/"


def inspect_file(path: Path) -> bool:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        print(f"[x] cannot read {path}: {exc}")
        return False

    if not lines or not lines[0].lstrip().startswith(("# Netscape HTTP Cookie File",
                                                       "# HTTP Cookie File")):
        print("[x] not a Netscape cookies.txt (first line must be "
              "'# Netscape HTTP Cookie File'). Export with a cookies.txt extension, "
              "not as JSON.")
        return False

    names: dict[str, int] = {}
    other_domains: set[str] = set()
    for raw in lines:
        line = raw.strip()
        if line.startswith("#HttpOnly_"):
            line = line[len("#HttpOnly_"):]
        elif not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue
        domain, expiry, name = parts[0].lower(), parts[4], parts[5]
        if "instagram.com" not in domain:
            other_domains.add(domain.lstrip("."))
            continue
        try:
            names[name] = int(float(expiry))
        except ValueError:
            names[name] = 0

    print(f"[i] instagram.com cookies: {len(names)} ({', '.join(sorted(names)) or 'none'})")
    if other_domains:
        print(f"[!] file also holds cookies for {len(other_domains)} other site(s). "
              "Export ONLY instagram.com -- everything in this file gets uploaded.")

    expiry = names.get("sessionid")
    if expiry is None:
        print("[x] no 'sessionid' cookie: the browser was not logged in to Instagram "
              "when you exported.")
        return False
    if expiry and expiry < time.time():
        print("[x] 'sessionid' has expired. Log in again and re-export.")
        return False
    if expiry:
        when = datetime.fromtimestamp(expiry, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"[ok] logged-in session found, expires {when}")
    else:
        print("[ok] logged-in session found (session cookie, no expiry)")
    return True


def try_extract(path: Path, url: str) -> bool:
    from extractor.config import ExtractorConfig
    from extractor.envelope import ExtractionTarget
    from extractor.extractors.ytdlp import YtDlpExtractor
    from extractor.urls import canonicalize, detect_platform, url_hash

    config = ExtractorConfig.from_env()
    config.cookie_file = str(path)
    config.cookies_from_browser = None
    canonical = canonicalize(url, follow_shorteners=False)
    target = ExtractionTarget(
        canonical_url=canonical,
        url_hash=url_hash(canonical),
        platform=detect_platform(canonical),
        original_url=url,
    )

    print(f"[i] running the yt-dlp tier on {url} ...")
    try:
        envelope = YtDlpExtractor(config).extract(target)
    except Exception as exc:  # noqa: BLE001 - report whatever Instagram said
        print(f"[x] extraction failed: {str(exc)[:300]}")
        return False
    if envelope is None:
        print("[x] yt-dlp returned nothing usable")
        return False

    caption = envelope.caption or ""
    print(f"[ok] author:  {envelope.author}")
    print(f"[ok] caption: {len(caption)} chars -- {caption[:100]!r}")
    print(f"[ok] video URL found: {bool(envelope.media_url)}, "
          f"duration: {envelope.duration_s}")
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    path = Path(sys.argv[1]).expanduser()
    url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_URL
    if not inspect_file(path):
        return 1
    return 0 if try_extract(path, url) else 1


if __name__ == "__main__":
    sys.exit(main())
