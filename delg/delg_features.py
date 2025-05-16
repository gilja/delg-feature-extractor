import os
from typing import cast, Optional
from google.protobuf import text_format
from google.protobuf.message import Message
from PIL import Image
import numpy as np

from . import delf_config_pb2
from . import extractor


def read_image_to_uint8(path: str) -> np.ndarray:
    """
    Load an image from disk and convert it to an RGB NumPy array of dtype uint8.

    This helper opens an image file using the Pillow library, converts it to RGB
    color mode (ensuring consistency across grayscale or palette images), and
    returns the result as a NumPy array with dtype `uint8`, suitable for use with
    DELG feature extractors.

    Args:
        path (str): Path to the image file to load.

    Returns:
        np.ndarray: A 3D array of shape (H, W, 3) with dtype `uint8` representing
                    the RGB image.
    """
    image = Image.open(path).convert("RGB")
    return np.array(image)


def _default_config_path(feature_type: str) -> str:
    """
    Resolve the default config file path for DELG feature extraction.

    Based on the requested feature type ('global' or 'local'), this helper
    constructs the absolute path to the corresponding `.pbtxt` configuration file
    located in the same directory as this module.

    Args:
        feature_type (str): Type of feature to extract, must be either 'global' or 'local'.

    Returns:
        str: Full path to the corresponding DELG config `.pbtxt` file.

    Raises:
        ValueError: If `feature_type` is not 'global' or 'local'.
    """
    if feature_type not in {"global", "local"}:
        raise ValueError("feature_type must be 'global' or 'local'.")
    filename = (
        "config_global.pbtxt" if feature_type == "global" else "config_local.pbtxt"
    )
    return os.path.join(os.path.dirname(__file__), filename)


def _load_config(config_path: str):
    """
    Load and parse a DELG configuration file in `.pbtxt` format.

    This helper reads a text-based DELG configuration file, parses it into a
    `DelfConfig` protobuf object, and returns the result for use in extractor setup.

    Args:
        config_path (str): Path to the `.pbtxt` file containing the DELG config.

    Returns:
        delf_config_pb2.DelfConfig: Parsed configuration object.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = delf_config_pb2.DelfConfig()
    with open(config_path, "r") as f:
        text_format.Merge(f.read(), cast(Message, config))
    return config


def extract_global_features(image_path: str, config_path: Optional[str]):
    """
    Extract global DELG features from a single RGB image.

    This function loads the given image, prepares the DELG extractor with a global
    feature config, and returns the resulting descriptor vector.

    Args:
        image_path (str): Path to the input image file.
        config_path (str, optional): Optional path to a `.pbtxt` config file.
                                     If None, a default global config is used.

    Returns:
        np.ndarray or None: 1D global descriptor vector if successful,
                            or None if extraction fails.
    """
    if config_path is None:
        config_path = _default_config_path("global")
    config = _load_config(config_path)

    image = read_image_to_uint8(image_path)

    extractor_fn = extractor.MakeExtractor(config)
    features = extractor_fn(image)

    return features.get("global_descriptor", None)


def extract_local_features(image_path: str, config_path: Optional[str]):
    """
    Extract local DELG features from a single RGB image.

    This function loads the image, applies a local-feature DELG extractor, and
    returns keypoint locations, descriptors, scales, and attention scores.

    Args:
        image_path (str): Path to the input image file.
        config_path (str, optional): Optional path to a `.pbtxt` config file.
                                     If None, a default local config is used.

    Returns:
        dict or None: Dictionary with keys 'locations', 'descriptors', 'scales',
                      and 'attention', or None if extraction fails.
    """
    if config_path is None:
        config_path = _default_config_path("local")
    config = _load_config(config_path)

    image = read_image_to_uint8(image_path)

    extractor_fn = extractor.MakeExtractor(config)
    features = extractor_fn(image)

    return features.get("local_features", None)


def batch_extract_features(
    image_paths: list[str], config_path: Optional[str], feature_type: str = "global"
) -> dict[str, np.ndarray | dict]:
    """
    Extract DELG features from a list of images using either global or local mode.

    This helper loops over a list of image file paths, loads each image,
    runs DELG feature extraction, and returns the results in a dictionary.
    It optionally falls back to a default config if `config_path` is not specified.

    Args:
        image_paths (list[str]): List of image file paths to process.
        config_path (str, optional): Path to a `.pbtxt` config file.
                                     If None, the appropriate default is used.
        feature_type (str): Either 'global' or 'local', indicating which type of
                            DELG features to extract.

    Returns:
        dict[str, np.ndarray | dict]: Dictionary mapping each image path to its
                                      extracted features (global vector or local dict).

    Raises:
        ValueError: If `feature_type` is not 'global' or 'local'.
    """
    if feature_type not in {"global", "local"}:
        raise ValueError("feature_type must be 'global' or 'local'.")

    if config_path is None:
        config_path = _default_config_path(feature_type)

    results = {}
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        try:
            if feature_type == "global":
                features = extract_global_features(path, config_path)
            else:
                features = extract_local_features(path, config_path)
            results[path] = features
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue

    return results
