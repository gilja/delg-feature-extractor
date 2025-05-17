import os
import sys


def _is_running_inside_docker() -> bool:
    return os.path.exists("/.dockerenv") or "entrypoint.py" in sys.argv[0]


if not _is_running_inside_docker():
    from .docker_runner import ensure_server_running

    ensure_server_running()

from .client import extract_global_features, extract_local_features

__all__ = [
    "extract_global_features",
    "extract_local_features",
]
