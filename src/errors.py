"""Explicit errors and invalid-result reports."""

import sys
from src.privacy import public_metadata, redact_text


class IFIPError(Exception):
    """Experiment error with location and context."""


def _format(tag: str, location: str, reason: str, context: dict) -> str:
    msg = f"[{tag}] {location}: {redact_text(reason)}"
    if context:
        ctx = "  ".join(f"{k}={v!r}" for k, v in public_metadata(context).items())
        msg += f"  | {ctx}"
    return msg


def fail(location: str, reason: str, **context):
    """Raise an explicit experiment error."""
    raise IFIPError(_format("IFIP-ERROR", location, reason, context))


def report_invalid(location: str, reason: str, **context) -> None:
    """Report an invalid result without raising."""
    print(_format("IFIP-INVALID", location, reason, context),
          file=sys.stderr, flush=True)
