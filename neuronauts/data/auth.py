"""Single source of truth for the CAVE bearer token.

A token was previously hardcoded in ~11 files across the repo. It has been
removed; supply one via the environment instead (``CAVE_TOKEN``, or the
lowercase ``token`` used by some hosted environments).

This module deliberately has no default. A missing token must fail loudly at
the call site rather than silently degrading a fetch into an empty result — an
empty fetch that gets treated as data is exactly the failure mode that produced
the synthetic-data incident.
"""

from __future__ import annotations

import os

_ENV_VARS = ("CAVE_TOKEN", "CAVE_BEARER_TOKEN", "token")


class MissingCaveToken(RuntimeError):
    """Raised when no CAVE token is available in the environment."""


def cave_token(required: bool = True) -> str | None:
    """Return the CAVE bearer token from the environment.

    Args:
        required: When True (default) raise :class:`MissingCaveToken` if no
            token is set. When False return ``None`` instead, for callers that
            can legitimately operate against public endpoints.

    Raises:
        MissingCaveToken: If ``required`` and no token is configured.
    """
    for var in _ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value.strip()
    if required:
        raise MissingCaveToken(
            "No CAVE token found. Set one of: "
            + ", ".join(_ENV_VARS)
            + "\nTokens must never be committed to this repository."
        )
    return None


def redact(token: str | None) -> str:
    """Render a token safe for logs and provenance stamps."""
    if not token:
        return "<unset>"
    return f"{token[:4]}...{token[-2:]} (len={len(token)})"
