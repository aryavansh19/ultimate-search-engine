"""SSRF guard for the tiers that fetch caller-supplied URLs directly.

Worth having from the start rather than retrofitting. The ingest endpoint's entire
job is "take a URL from a user and fetch it", which is textbook server-side request
forgery exposure: `http://169.254.169.254/` reaches cloud instance metadata,
`http://127.0.0.1:8000/` reaches whatever else you are running, and internal
`10.x` addresses reach your own network.

Resolving the hostname and rejecting private, loopback, link-local and reserved
address space closes the obvious hole. It does not close DNS rebinding, where a name
resolves to a public address during the check and a private one during the fetch --
that needs pinning the connection to the validated IP. Local-only prototype is fine
with the check below; add pinning before exposing this to the internet.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_ALLOWED_SCHEMES = {"http", "https"}


class UnsafeUrl(Exception):
    """The URL resolves somewhere a server should not be making requests."""


def assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeUrl(f"scheme not allowed: {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise UnsafeUrl("missing host")

    try:
        infos = socket.getaddrinfo(host, parsed.port or 0, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeUrl(f"cannot resolve host {host!r}") from exc

    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            raise UnsafeUrl(f"unparseable address {address!r}") from None
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise UnsafeUrl(f"{host} resolves to non-public address {ip}")
