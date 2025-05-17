# Copyright 2020 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helper functions for DELF."""

import os
import numpy as np
from PIL import Image
from PIL import ImageFile
import tensorflow as tf

from google.protobuf import text_format
from google.protobuf.message import Message
from typing import cast
from . import delf_config_pb2

# To avoid PIL crashing for truncated (corrupted) images.
ImageFile.LOAD_TRUNCATED_IMAGES = True


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


def RgbLoader(path):
    """Helper function to read image with PIL.

    Args:
      path: Path to image to be loaded.

    Returns:
      PIL image in RGB format.
    """
    with tf.io.gfile.GFile(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def ResizeImage(image, config, resize_factor=1.0):
    """Resizes image according to config.

    Args:
      image: Uint8 array with shape (height, width, 3).
      config: DelfConfig proto containing the model configuration.
      resize_factor: Optional float resize factor for the input image. If given,
        the maximum and minimum allowed image sizes in `config` are scaled by this
        factor. Must be non-negative.

    Returns:
      resized_image: Uint8 array with resized image.
      scale_factors: 2D float array, with factors used for resizing along height
        and width (If upscaling, larger than 1; if downscaling, smaller than 1).

    Raises:
      ValueError: If `image` has incorrect number of dimensions/channels.
    """
    if resize_factor < 0.0:
        raise ValueError("negative resize_factor is not allowed: %f" % resize_factor)
    if image.ndim != 3:
        raise ValueError("image has incorrect number of dimensions: %d" % image.ndims)
    height, width, channels = image.shape

    # Take into account resize factor.
    max_image_size = resize_factor * config.max_image_size
    min_image_size = resize_factor * config.min_image_size

    if channels != 3:
        raise ValueError("image has incorrect number of channels: %d" % channels)

    largest_side = max(width, height)

    if max_image_size >= 0 and largest_side > max_image_size:
        scale_factor = max_image_size / largest_side
    elif min_image_size >= 0 and largest_side < min_image_size:
        scale_factor = min_image_size / largest_side
    elif config.use_square_images and (height != width):
        scale_factor = 1.0
    else:
        # No resizing needed, early return.
        return image, np.ones(2, dtype=float)

    # Note that new_shape is in (width, height) format (PIL convention), while
    # scale_factors are in (height, width) convention (NumPy convention).
    if config.use_square_images:
        new_shape = (
            int(round(largest_side * scale_factor)),
            int(round(largest_side * scale_factor)),
        )
    else:
        new_shape = (
            int(round(width * scale_factor)),
            int(round(height * scale_factor)),
        )

    scale_factors = np.array([new_shape[1] / height, new_shape[0] / width], dtype=float)

    pil_image = Image.fromarray(image)
    resized_image = np.array(pil_image.resize(new_shape, resample=Image.BILINEAR))

    return resized_image, scale_factors
