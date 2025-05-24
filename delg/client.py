import requests
from pathlib import Path
from typing import List, Dict, Union
import os
import sys

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


def extract_global_features(
    image_paths: Union[str, List[str]],
) -> Union[List[float], List[List[float]]]:
    if isinstance(image_paths, str):
        return _post_image(image_paths, "extract/global")["global_descriptor"]
    return [_post_image(p, "extract/global")["global_descriptor"] for p in image_paths]


def extract_local_features(
    image_paths: Union[str, List[str]],
) -> Union[Dict, List[Dict]]:
    if isinstance(image_paths, str):
        return _post_image(image_paths, "extract/local")["local_features"]
    return [_post_image(p, "extract/local")["local_features"] for p in image_paths]
