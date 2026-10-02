"""Shrink video before it is sent to a model.

Multimodal models bill video by sampled frames and audio seconds, not by file size, so
resolution and frame rate beyond what the model actually samples are pure waste. Gemini
samples video at roughly 1 fps, which means a 30 fps 1080p reel ships 29 frames out of every
30 that nobody looks at.

Measured on a real 17.7 MB Instagram reel:

    original            17.7 MB    16,760 input tokens    $0.00561
    20s 480p 1fps clip   0.45 MB    5,230 input tokens    $0.00056

and the compressed clip still read "ANYTIME GAMING" off the storefront signage correctly. It
also sidesteps a practical failure: a 17.7 MB video becomes ~24 MB as base64, which caused
outright connection resets against some endpoints.

Compression is best-effort by design. If ffmpeg is missing, too old, or fails on a codec, the
original bytes are returned and the pipeline continues -- a smaller upload is an optimization,
never a requirement.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("enrichment.video")


@dataclass(slots=True)
class CompressionResult:
    data: bytes
    mime: str
    original_bytes: int
    compressed: bool
    note: str | None = None

    @property
    def ratio(self) -> float:
        if not self.original_bytes:
            return 1.0
        return len(self.data) / self.original_bytes


def find_ffmpeg(configured: str | None = None) -> str | None:
    """Locate ffmpeg, preferring an explicit path from config."""
    if configured:
        candidate = Path(configured)
        if candidate.is_file():
            return str(candidate)
        found = shutil.which(configured)
        if found:
            return found
    found = shutil.which("ffmpeg")
    if found:
        return found
    # Common Windows install location that is frequently not on PATH.
    for guess in (r"C:\ffmpeg\bin\ffmpeg.exe", r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"):
        if Path(guess).is_file():
            return guess
    return None


def compress_video(
    data: bytes,
    *,
    ffmpeg: str | None = None,
    max_seconds: float = 180.0,
    fps: float = 1.0,
    height: int = 480,
    crf: int = 32,
    audio_bitrate: str = "48k",
    timeout: float = 300.0,
) -> CompressionResult:
    """Re-encode to the smallest form that preserves what a model actually samples.

    Audio is kept, at a low bitrate: speech is often the most searchable content in a reel and
    it costs far fewer tokens than video frames. Dropping it to save bytes would be a bad
    trade.
    """
    original = len(data)
    binary = find_ffmpeg(ffmpeg)
    if not binary:
        return CompressionResult(
            data, "video/mp4", original, False, "ffmpeg not found; sent original"
        )

    with tempfile.TemporaryDirectory(prefix="linkmem-") as tmp:
        source = Path(tmp) / "in.mp4"
        target = Path(tmp) / "out.mp4"
        source.write_bytes(data)

        command = [
            binary, "-y", "-loglevel", "error",
            "-i", str(source),
            "-t", str(int(max_seconds)),
            # `fps` before `scale` so scaling only runs on frames that survive.
            "-vf", f"fps={fps},scale=-2:{height}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", audio_bitrate, "-ac", "1",
            "-movflags", "+faststart",
            str(target),
        ]
        try:
            proc = subprocess.run(
                command, capture_output=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            return CompressionResult(
                data, "video/mp4", original, False, "ffmpeg timed out; sent original"
            )
        except OSError as exc:
            return CompressionResult(
                data, "video/mp4", original, False, f"ffmpeg failed ({exc}); sent original"
            )

        if proc.returncode != 0 or not target.exists() or target.stat().st_size == 0:
            detail = (proc.stderr or b"").decode("utf-8", "replace")[:160].strip()
            return CompressionResult(
                data, "video/mp4", original, False,
                f"ffmpeg error; sent original{': ' + detail if detail else ''}",
            )

        out = target.read_bytes()

    # Refuse a "compression" that made things bigger. Short, already-optimized clips can
    # grow under re-encoding, and shipping the larger file would be strictly worse.
    if len(out) >= original:
        return CompressionResult(
            data, "video/mp4", original, False, "re-encode was larger; sent original"
        )

    note = (
        f"compressed {original / 1_000_000:.1f}MB -> {len(out) / 1_000_000:.2f}MB "
        f"({height}p, {fps:g}fps, {int(max_seconds)}s cap)"
    )
    log.debug(note)
    return CompressionResult(out, "video/mp4", original, True, note)
