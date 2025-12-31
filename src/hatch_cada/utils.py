import os
from pathlib import Path

from hatch_cada.constants import UV_LOCKFILE_NAME


def find_workspace_root(start_path: Path) -> Path | None:
    """Find the workspace root by looking for uv.lock walking up from start_path.

    Can be overridden by setting WORKSPACE_ROOT environment variable.

    Args:
        start_path: Directory to start searching from.

    Returns:
        The workspace root path, or None if not found.
    """
    workspace_root_env = os.environ.get("WORKSPACE_ROOT")
    if workspace_root_env is not None:
        return Path(workspace_root_env)

    current = start_path.resolve()
    while current != current.parent:
        if (current / UV_LOCKFILE_NAME).exists():
            return current
        current = current.parent

    return None
