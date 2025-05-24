import requests
from pathlib import Path
from typing import List, Dict, Union, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_URL = "http://localhost:8080"


def _post_image(image_path: str, endpoint: str) -> Dict:
    """Helper to post an image to the FastAPI server and return the parsed JSON."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(path, "rb") as f:
        files = {"image": f}
        response = requests.post(f"{SERVER_URL}/{endpoint}", files=files)
        response.raise_for_status()
        return response.json()


def extract_global_features(
    image_paths: Union[str, List[str]],
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> Union[List[float], List[Optional[List[float]]]]:
    """
    Extract global features for one or more images.

    Args:
        image_paths: Single image path or list of image paths.
        parallel: Whether to run in parallel (only affects list input).
        max_workers: Number of threads to use (default: min(32, os.cpu_count() + 4))

    Returns:
        Single descriptor if input is a string,
        otherwise a list of descriptors or None for failed images.
    """
    if isinstance(image_paths, str):
        return _post_image(image_paths, "extract/global")["global_descriptor"]

    if not parallel:
        return [
            _post_image(p, "extract/global").get("global_descriptor")
            for p in image_paths
        ]

    results: List[Optional[List[float]]] = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_post_image, path, "extract/global"): idx
            for idx, path in enumerate(image_paths)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result.get("global_descriptor")
            except (requests.RequestException, FileNotFoundError, ValueError, KeyError):
                results[idx] = None

    return results


def extract_local_features(
    image_paths: Union[str, List[str]],
    parallel: bool = False,
    max_workers: Optional[int] = None,
) -> Union[Dict, List[Optional[Dict]]]:
    """
    Extract local features for one or more images.

    Args:
        image_paths: Single image path or list of image paths.
        parallel: Whether to run in parallel (only affects list input).
        max_workers: Number of threads to use (default: min(32, os.cpu_count() + 4))

    Returns:
        Single dict of local features if input is a string,
        otherwise a list of feature dicts or None for failed images.
    """
    if isinstance(image_paths, str):
        return _post_image(image_paths, "extract/local")["local_features"]

    if not parallel:
        return [
            _post_image(p, "extract/local").get("local_features") for p in image_paths
        ]

    results: List[Optional[Dict]] = [None] * len(image_paths)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_post_image, path, "extract/local"): idx
            for idx, path in enumerate(image_paths)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results[idx] = result.get("local_features")
            except (requests.RequestException, FileNotFoundError, ValueError, KeyError):
                results[idx] = None

    return results
