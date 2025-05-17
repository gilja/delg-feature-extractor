import subprocess
import time
import socket

DOCKER_IMAGE = "delg-server"
DOCKER_CONTAINER = "delg-server-container"
PORT = 8080


def is_port_open(host: str, port: int) -> bool:
    """Check if a given host:port is accepting TCP connections."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except (ConnectionRefusedError, socket.timeout):
            return False


def docker_image_exists() -> bool:
    result = subprocess.run(
        ["docker", "images", "-q", DOCKER_IMAGE],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip() != ""


def docker_container_running() -> bool:
    result = subprocess.run(
        ["docker", "ps", "--filter", f"name={DOCKER_CONTAINER}", "--format", "{{.ID}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip() != ""


def build_docker_image():
    print("🔨 Building Docker image...")
    subprocess.run(["docker", "build", "-t", DOCKER_IMAGE, "."], check=True)
    print("✅ Docker image built.")


def start_docker_container():
    print("🐳 Starting Docker container...")
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "-p",
            f"{PORT}:8080",
            "--name",
            DOCKER_CONTAINER,
            DOCKER_IMAGE,
        ],
        check=True,
    )
    print("✅ Container started.")


def wait_for_server(timeout=30):
    print("⏳ Waiting for server to become available on port", PORT)
    for _ in range(timeout * 2):
        if is_port_open("localhost", PORT):
            print("🌐 Server is up!")
            return
        time.sleep(0.5)
    raise RuntimeError("❌ Server did not become available in time.")


def ensure_server_running():
    if docker_container_running():
        return

    if not docker_image_exists():
        build_docker_image()

    start_docker_container()
    wait_for_server()
