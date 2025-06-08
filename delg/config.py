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

import requests

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


def update_local_config(max_feature_num=1000, score_threshold=54.6):
    url = f"http://localhost:{docker_port}/config/local"
    payload = {"max_feature_num": max_feature_num, "score_threshold": score_threshold}
    response = requests.post(url, json=payload, timeout=5)
    response.raise_for_status()
    return response.json()
