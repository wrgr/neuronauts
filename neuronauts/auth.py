"""Where the CAVE token comes from — never from a source file.

A token was committed literally in ten files of this repository, which is a
public GitHub repository, so it sat in pushed history. Rotating the credential
is the only thing that undoes that; removing the lines does not. This module
exists so there is one obvious place to get a token and no reason to paste one
again.

Resolution order:

1. ``CAVE_TOKEN`` in the environment, for continuous integration and containers.
2. ``~/.cloudvolume/secrets/cave-secret.json``, which is where CloudVolume and
   caveclient already look, so most code needs nothing at all — construct a
   ``CAVEclient`` and it resolves the token itself.

Nothing here ever prints or logs the value.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

SECRET = Path.home() / ".cloudvolume" / "secrets" / "cave-secret.json"


def cave_token(required: bool = True) -> str | None:
    """Return the CAVE token, or None when absent and ``required`` is False."""
    tok = os.environ.get("CAVE_TOKEN")
    if tok:
        return tok.strip()
    if SECRET.exists():
        try:
            tok = json.loads(SECRET.read_text()).get("token")
        except (json.JSONDecodeError, OSError):
            tok = None
        if tok:
            return str(tok).strip()
    if required:
        raise RuntimeError(
            "No CAVE token. Set CAVE_TOKEN in the environment, or place one at "
            f"{SECRET}. Never hard-code it in a source file."
        )
    return None
