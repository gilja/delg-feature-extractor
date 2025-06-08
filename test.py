from delg.client import extract_global_features, extract_local_features

from delg.similarity import local_feature_match, cosine_similarity
import os
import time

from delg.config import update_local_config, set_docker_config


update_local_config(max_feature_num=1000, score_threshold=454.6)
set_docker_config(image="delg-server", container="delg-server-container", port=8080)

# Configurable image suffixes
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")

# Load paths
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

# --- Start timing ---
start = time.perf_counter()

# Extract all features in parallel (preserves order)
features_1 = extract_global_features(images_1, parallel=True, max_workers=20)
features_2 = extract_global_features(images_2, parallel=True, max_workers=20)

# --- End timing ---
end = time.perf_counter()
elapsed = end - start
print(f"Feature extraction took {elapsed:.2f} seconds.")

# Compare all pairs
for idx1, (img1, phash_1) in enumerate(zip(images_1, features_1)):
    if phash_1 is None:
        continue  # Skip failed image

    for idx2, (img2, phash_2) in enumerate(zip(images_2, features_2)):
        if phash_2 is None:
            continue

        cos_sim = cosine_similarity(phash_1, phash_2)

        if cos_sim > 0.8:
            print("Duplikat:")
            print(img1)
            print(img2)
            print(f"cos_sim: {cos_sim:.4f}")
            print("------" * 20)
            continue

        if cos_sim > 0.6:

            local_f1 = extract_local_features(img1)
            local_f2 = extract_local_features(img2)

            match_score = local_feature_match(
                local_f1,
                local_f2,
                ransac_residual_threshold=15,
                min_inliers=8,
                ratio_thresh=0.9,
            )

            print("Duplikat:")
            print(img1)
            print(img2)
            print(f"cos_sim: {cos_sim:.4f}")
            print(f"match_score: {match_score}")
            print("------" * 20)
