"""Redact credentials and host paths in metadata and error reports."""

import os
import re
from pathlib import Path

from src.paths import PROJECT_ROOT

_KEY = re.compile(r"\bsk-(?:or-v\d+-)?[A-Za-z0-9_-]{16,}\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PRIVATE_FIELD = re.compile(r"(?:api_?key|secret|password|authorization|access_token|credential)", re.I)
_HOME = re.compile(r"/(?:workspace/users|home|Users)/[^/\s'\"]+|/root(?=/|\s|$)")


def redact_text(value: str) -> str:
    """Strip secrets and host identifiers; leave experiment data untouched."""
    text = str(value)
    for name, secret in os.environ.items():
        if _PRIVATE_FIELD.search(name) and secret and len(secret) >= 8:
            text = text.replace(secret, "<redacted>")
    text = _KEY.sub("<redacted>", text)
    text = _EMAIL.sub("<email>", text)
    text = text.replace(str(PROJECT_ROOT), "IF-MAP")
    return _HOME.sub("<home>", text)


def public_metadata(value):
    """Sanitize config snapshots, not model prompts or cache identities."""
    if isinstance(value, dict):
        return {key: "<redacted>" if _PRIVATE_FIELD.search(str(key))
                else public_metadata(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [public_metadata(item) for item in value]
    if isinstance(value, (str, Path)):
        return redact_text(str(value))
    return value
