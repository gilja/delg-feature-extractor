from delg.client import extract_global_features
import os
import numpy as np


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two global descriptor vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0  # avoid division by zero

    return float(np.dot(v1, v2) / (norm1 * norm2))


image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")

images_path_1 = "/Users/duje/Documents/Data_science/projects/homefinder/data/raw_images/flats_for_sale/5350f7ac6907d77b3954b5036b6c19e4cd159778"
images_1 = sorted(
    [
        os.path.join(images_path_1, f)
        for f in os.listdir(images_path_1)
        if f.lower().endswith(image_extensions)
    ]
)

images_path_2 = "/Users/duje/Documents/Data_science/projects/homefinder/data/raw_images/flats_for_sale/c6b8f22f7ba4045e5a6e92ec9c9f37646c8117b2"
images_2 = sorted(
    [
        os.path.join(images_path_2, f)
        for f in os.listdir(images_path_2)
        if f.lower().endswith(image_extensions)
    ]
)

for _, img1 in enumerate(images_1):

    phash_1 = extract_global_features(img1)

    for _, img2 in enumerate(images_2):

        phash_2 = extract_global_features(img2)

        cos_sim = cosine_similarity(phash_1, phash_2)

        print(img1)
        print(img2)
        print(f"cos_sim: {cos_sim}")
        print("------" * 20)

        # if cos_sim > 0.8:
        #     print("Duplikat:")
        #     print(img1)
        #     print(img2)
        #     print("------" * 20)
