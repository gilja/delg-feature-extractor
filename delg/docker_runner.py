import subprocess
import time
import socket
import os
import atexit
import requests
from . import config

_docker_process = None


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout):
            return False


def docker_image_exists() -> bool:
    result = subprocess.run(
        ["docker", "images", "-q", config.docker_image],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip() != ""


def build_docker_image():
    subprocess.run(["docker", "build", "-t", config.docker_image, "."], check=True)


def wait_for_server(timeout=60):
    """Block until the DELG server responds on /healthz."""
    url = f"http://localhost:{config.docker_port}/healthz"
    for _ in range(timeout * 2):  # 60 total wait time
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return  # ✅ Only return when server is ready
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("❌ Server did not become available in time.")


def start_docker_container():
    global _docker_process

    _docker_process = subprocess.Popen(
        [
            "docker",
            "run",
            "--rm",
            "-p",
            f"{config.docker_port}:8080",
            "--name",
            config.docker_container,
            config.docker_image,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,  # ensures separate process group
    )

    # Register automatic shutdown hook
    atexit.register(stop_docker_container)


def stop_docker_container():
    """Stop the running Docker container if it's active."""
    try:
        subprocess.run(
            ["docker", "stop", config.docker_container],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def ensure_server_running():
    if not docker_image_exists():
        build_docker_image()

    start_docker_container()
    wait_for_server()
