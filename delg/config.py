import os

# Docker runtime config
docker_image = "delg-server"
docker_container = "delg-server-container"
docker_port = 8080


def set_docker_config(image=None, container=None, port=None):
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
    if not isinstance(use_pca, bool):
        raise TypeError(f"use_pca must be a boolean, got {type(use_pca).__name__}")

    path = os.path.join(os.path.dirname(__file__), "model_configs/config_global.pbtxt")
    lines = []
    with open(path, "r") as f:
        for line in f:
            if "use_pca" in line:
                lines.append(f"use_pca: {'true' if use_pca else 'false'}\n")
            else:
                lines.append(line)
    with open(path, "w") as f:
        f.writelines(lines)


def update_local_config(use_pca=False, max_feature_num=1000, score_threshold=54.6):
    if not isinstance(use_pca, bool):
        raise TypeError(f"use_pca must be a boolean, got {type(use_pca).__name__}")
    if not isinstance(max_feature_num, int) or max_feature_num <= 0:
        raise ValueError("max_feature_num must be a positive integer")
    if not isinstance(score_threshold, (int, float)) or score_threshold < 0:
        raise ValueError("score_threshold must be a non-negative number")

    path = os.path.join(os.path.dirname(__file__), "model_configs/config_local.pbtxt")
    lines = []
    with open(path, "r") as f:
        for line in f:
            if "use_pca" in line:
                lines.append(f"use_pca: {'true' if use_pca else 'false'}\n")
            elif "score_threshold" in line:
                lines.append(f"score_threshold: {score_threshold}\n")
            elif "max_feature_num" in line:
                lines.append(f"max_feature_num: {max_feature_num}\n")
            else:
                lines.append(line)
    with open(path, "w") as f:
        f.writelines(lines)
