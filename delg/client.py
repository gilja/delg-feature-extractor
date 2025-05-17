import requests
from pathlib import Path
from typing import List, Dict

SERVER_URL = "http://localhost:8080"


def _post_image(image_path: str, endpoint: str) -> Dict:
    """Helper to post an image to the FastAPI server and return the parsed JSON."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(path, "rb") as f:
        files = {"image": f}
        try:
            response = requests.post(f"{SERVER_URL}/{endpoint}", files=files)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to contact DELG server: {e}") from e


def extract_global_features(image_path: str) -> List[float]:
    """
    Sends an image to the DELG server and returns the global descriptor.

    Args:
        image_path (str): Path to image file.

    Returns:
        List[float]: Global descriptor vector.
    """
    result = _post_image(image_path, "extract/global")
    return result["global_descriptor"]


def extract_local_features(image_path: str) -> Dict:
    """
    Sends an image to the DELG server and returns local features.

    Args:
        image_path (str): Path to image file.

    Returns:
        Dict: Dictionary of local features with keys: locations, descriptors, scales, attention
    """
    result = _post_image(image_path, "extract/local")
    return result["local_features"]
