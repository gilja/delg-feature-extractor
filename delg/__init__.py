"""
delg
====

This is the package entry point for the DELG feature extractor. It initializes
the Docker container when not running inside Docker and exposes the key
functions needed for global and local feature extraction.

Public functions:
-----------------

-   extract_global_features: Extracts global features from one or more images.
-   extract_local_features: Extracts local features from one or more images.
-   update_global_config: Updates the 'use_pca' setting in the global DELG config file.
-   update_local_config: Updates the 'use_pca', 'max_feature_num', and 'score_threshold'
    settings in the local DELG config file.
-   set_docker_config: Sets Docker runtime configuration variables.
-   cosine_similarity: Computes cosine similarity between two global descriptor vectors.
-   local_feature_match: Determines whether two images match based on their local features.
-   download_weights: Downloads the DELG model weights and extracts them.

For more information on the functions, refer to their docstrings.

Notes:
------

Author: Duje Giljanović (giljanovic.duje@gmail.com)
License: Apache License 2.0 (same as the official DELG implementation)

This package uses the DELG model originally developed by Google Research and published
in paper "Unifying Deep Local and Global Features for Image Search" authored by Bingyi Cao,
Andre Araujo, and Jack Sim.

If you use this Python package in your research or any other publication, please cite both this
package and the original DELG paper as follows:

@software{delg,
    title = {delg: A Python Package for Dockerized DELG Implementation},
    author = {Duje Giljanović},
    year = {2025},
    url = {https://github.com/gilja/delg-feature-extractor}
}

@article{cao2020delg,
    title = {Unifying Deep Local and Global Features for Image Search},
    author = {Bingyi Cao and Andre Araujo and Jack Sim},
    journal = {arXiv preprint arXiv:2001.05027},
    year = {2020}
}
"""

import os
import sys


def _is_running_inside_docker() -> bool:
    """
    Determines whether the current process is running inside a Docker container.

    Checks for the presence of the '/.dockerenv' file and 'entrypoint.py' in the
    process arguments to identify Docker runtime context.

    Returns:
      bool: True if running inside Docker, False otherwise.
    """

    return os.path.exists("/.dockerenv") or "entrypoint.py" in sys.argv[0]


if not _is_running_inside_docker():
    from .docker_runner import _ensure_server_running

    _ensure_server_running()

from .client import extract_global_features, extract_local_features
from .config import update_global_config, update_local_config, set_docker_config
from .similarity import cosine_similarity, local_feature_match
from .model_weights import download_weights

__all__ = [
    "extract_global_features",
    "extract_local_features",
    "update_global_config",
    "update_local_config",
    "set_docker_config",
    "cosine_similarity",
    "local_feature_match",
    "download_weights",
]
