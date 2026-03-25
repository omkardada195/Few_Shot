import os
import random
from pathlib import Path

import numpy as np
from PIL import Image


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def find_dataset_root(candidate_paths):
    for p in candidate_paths:
        if os.path.isdir(p):
            if all(os.path.isdir(os.path.join(p, split)) for split in ["train", "val", "test"]):
                return p
    raise FileNotFoundError("Could not find dataset root with train/val/test folders.")


def list_task_dirs(data_root):
    data_root = Path(data_root)
    out = {}
    for split in ["train", "val", "test"]:
        split_dir = data_root / split
        out[split] = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    return out


def list_valid_files(folder):
    folder = Path(folder)
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS])


def pair_task_files(task_dir):
    task_dir = Path(task_dir)
    img_dir = task_dir / "images"
    msk_dir = task_dir / "masks"

    img_files = list_valid_files(img_dir)
    msk_files = list_valid_files(msk_dir)

    if len(img_files) == 0 or len(msk_files) == 0:
        raise ValueError(f"No valid files found in {task_dir}")

    img_map = {p.stem: p for p in img_files}
    msk_map = {p.stem: p for p in msk_files}

    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    if len(common) == len(img_files) == len(msk_files):
        return [(img_map[k], msk_map[k]) for k in common]

    if len(img_files) != len(msk_files):
        raise ValueError(f"Image/mask count mismatch in {task_dir}")

    return list(zip(sorted(img_files), sorted(msk_files)))


def load_image(path, target_size=(192, 192)):
    img = Image.open(path).convert("L")
    img = img.resize(target_size, resample=Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=-1)


def load_mask(path, target_size=(192, 192), threshold=0.5):
    msk = Image.open(path).convert("L")
    msk = msk.resize(target_size, resample=Image.NEAREST)
    arr = np.array(msk, dtype=np.float32) / 255.0
    arr = (arr > threshold).astype(np.float32)
    return np.expand_dims(arr, axis=-1)


def mask_foreground_pixels(mask_array):
    return int(np.sum(mask_array))


def is_positive_mask(mask_array, min_pixels=10):
    return mask_foreground_pixels(mask_array) >= min_pixels


def get_task_index(task_dir, target_size=(192, 192), min_positive_pixels=10):
    """
    Lightweight index:
    Reads masks one by one and records which samples are positive/empty.
    Does NOT load all images into memory.
    """
    pairs = pair_task_files(task_dir)

    index = {
        "task_name": Path(task_dir).name,
        "pairs": [],
        "positive_indices": [],
        "empty_indices": [],
    }

    for i, (img_path, msk_path) in enumerate(pairs):
        mask = load_mask(msk_path, target_size=target_size)
        record = {
            "image_path": str(img_path),
            "mask_path": str(msk_path),
        }
        index["pairs"].append(record)

        if is_positive_mask(mask, min_pixels=min_positive_pixels):
            index["positive_indices"].append(i)
        else:
            index["empty_indices"].append(i)

    return index


def load_samples_from_indices(task_index, indices, target_size=(192, 192)):
    xs, ys = [], []
    for idx in indices:
        img_path = task_index["pairs"][idx]["image_path"]
        msk_path = task_index["pairs"][idx]["mask_path"]
        xs.append(load_image(img_path, target_size=target_size))
        ys.append(load_mask(msk_path, target_size=target_size))
    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    return X, Y


def choose_support_query_indices(
    task_index,
    k_shot,
    seed=42,
    prefer_positive=True,
    min_query=1
):
    """
    Picks support indices and query indices from one task.

    If prefer_positive=True, support images are chosen from positive masks first.
    Query indices always come from the remaining samples and are NOT filtered.
    """
    rng = random.Random(seed)

    all_indices = list(range(len(task_index["pairs"])))
    positive_indices = list(task_index["positive_indices"])
    empty_indices = list(task_index["empty_indices"])

    if prefer_positive and len(positive_indices) >= k_shot:
        support = rng.sample(positive_indices, k_shot)
    else:
        support = rng.sample(all_indices, k_shot)

    remaining = [i for i in all_indices if i not in support]

    if len(remaining) < min_query:
        raise ValueError(
            f"Not enough query samples left after selecting {k_shot} support samples."
        )

    query = remaining
    return sorted(support), sorted(query)


def dataset_summary(data_root):
    task_dirs = list_task_dirs(data_root)
    summary = []

    for split, tasks in task_dirs.items():
        for task_dir in tasks:
            pairs = pair_task_files(task_dir)
            summary.append({
                "split": split,
                "task": task_dir.name,
                "num_samples": len(pairs)
            })

    return summary