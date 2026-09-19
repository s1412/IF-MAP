"""Project-root paths; independent of the launch directory."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    """Resolve relative paths from IF-MAP, not the process cwd."""
    path = Path(os.path.expandvars(os.path.expanduser(os.fspath(value))))
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(value):
    """Show a portable path without the host's directory prefix."""
    path = project_path(value).resolve()
    try:
        return str(Path("IF-MAP") / path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(Path("<external>") / path.name)
