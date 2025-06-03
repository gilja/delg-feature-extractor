"""
config
======

This module defines helper functions for configuring the Docker runtime and
adjusting DELG model parameters used during feature extraction.

Public functions:
-----------------

-   set_docker_config: Sets Docker runtime configuration variables (image name, container
    name, and port).
-   update_global_config: Updates the 'use_pca' setting in the global DELG configuration file.
-   update_local_config: Updates the 'use_pca', 'max_feature_num', and 'score_threshold'
    settings in the local DELG configuration file.

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

# Docker runtime config
docker_image = "delg-server"
docker_container = "delg-server-container"
docker_port = 8080


def set_docker_config(image=None, container=None, port=None):
    """
    Sets Docker runtime configuration variables.

    Updates the Docker image name, container name, and port for the DELG server,
    allowing dynamic configuration.

    Args:
      image: Optional string specifying the Docker image name.
      container: Optional string specifying the Docker container name.
      port: Optional integer specifying the Docker port to expose
            (must be between 1 and 65535).

    Raises:
      ValueError: If the port value is invalid (not an integer between 1 and 65535).
    """

    global docker_image, docker_container, docker_port

    if image is not None:
        docker_image = image
    if container is not None:
        docker_container = container
    if port is not None:
        if not isinstance(port, int) or not (1 <= port <= 65535):
            raise ValueError("Port must be an integer between 1 and 65535.")
        docker_port = port


def update_global_config(use_pca=False):
    """
    Updates the global DELG configuration file.

    Modifies the 'use_pca' setting in the global DELG configuration `.pbtxt` file
    based on the provided argument, allowing toggling of PCA during feature extraction.

    Args:
      use_pca: Boolean indicating whether to enable PCA for global features.

    Raises:
      TypeError: If `use_pca` is not a boolean.
    """

    if not isinstance(use_pca, bool):
        raise TypeError(f"use_pca must be a boolean, got {type(use_pca).__name__}")

    path = os.path.join(os.path.dirname(__file__), "model_configs/config_global.pbtxt")
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "use_pca" in line:
                lines.append(f"use_pca: {'true' if use_pca else 'false'}\n")
            else:
                lines.append(line)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def update_local_config(use_pca=False, max_feature_num=1000, score_threshold=54.6):
    """
    Updates the local DELG configuration file.

    Modifies the 'use_pca', 'max_feature_num', and 'score_threshold' settings in the
    local DELG configuration `.pbtxt` file based on the provided arguments, allowing
    fine-tuning of local feature extraction parameters.

    Args:
      use_pca: Boolean indicating whether to enable PCA for local features.
      max_feature_num: Positive integer specifying the maximum number of local features.
      score_threshold: Non-negative float or integer representing the threshold score for
        retaining local features.

    Raises:
      TypeError: If `use_pca` is not a boolean.
      ValueError: If `max_feature_num` is not a positive integer or if `score_threshold`
        is negative.
    """

    if not isinstance(use_pca, bool):
        raise TypeError(f"use_pca must be a boolean, got {type(use_pca).__name__}")
    if not isinstance(max_feature_num, int) or max_feature_num <= 0:
        raise ValueError("max_feature_num must be a positive integer")
    if not isinstance(score_threshold, (int, float)) or score_threshold < 0:
        raise ValueError("score_threshold must be a non-negative number")

    path = os.path.join(os.path.dirname(__file__), "model_configs/config_local.pbtxt")
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "use_pca" in line:
                lines.append(f"use_pca: {'true' if use_pca else 'false'}\n")
            elif "score_threshold" in line:
                lines.append(f"score_threshold: {score_threshold}\n")
            elif "max_feature_num" in line:
                lines.append(f"max_feature_num: {max_feature_num}\n")
            else:
                lines.append(line)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
