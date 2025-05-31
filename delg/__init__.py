import os
import sys
import numpy as np


def _is_running_inside_docker() -> bool:
    return os.path.exists("/.dockerenv") or "entrypoint.py" in sys.argv[0]


if not _is_running_inside_docker():
    from .docker_runner import ensure_server_running

    ensure_server_running()

from .client import extract_global_features, extract_local_features
from .config import update_global_config, update_local_config, set_docker_config
from .similarity import cosine_similarity, local_feature_match

__all__ = [
    "extract_global_features",
    "extract_local_features",
    "update_global_config",
    "update_local_config",
    "set_docker_config",
    "cosine_similarity",
    "local_feature_match",
]
