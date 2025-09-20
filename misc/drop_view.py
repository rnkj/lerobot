"""
drop_view.py

Removes specified camera views from a dataset directory by updating metadata and deleting corresponding image directories.

Usage:
    python drop_view.py --dataset_root <path_to_dataset> --camera_names <camera1> <camera2> ... [--no_copy_dataset]

Arguments:
    --dataset_root   Path to the root directory of the dataset.
    --camera_names   List of camera names to drop from the dataset.
    --no_copy_dataset   If set, does not copy the dataset before dropping views (default: False).

Notes:
    - The script expects the dataset to have a specific structure with 'meta/info.json', 'meta/episodes_stats.jsonl', and video directories.
    - Use '--no_copy_dataset' to avoid copying the dataset before dropping views.
"""

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

from utils import check_camera_names


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="path to dataset directory")
    parser.add_argument("--camera_names", type=str, nargs="+", required=True, help="camera names to drop")
    parser.add_argument(
        "--no_copy_dataset", action="store_true", help="whether to copy the dataset before dropping views"
    )
    return parser.parse_args()


def drop_view(dataset_root: str, camera_names: list[str]):
    dataset_root = Path(dataset_root)

    # Load meta/info.json
    info_path = dataset_root / "meta/info.json"
    with open(info_path, "r") as f:
        info = json.load(f)

    total_episodes = info["total_episodes"]
    features = info["features"]

    camera_views = [f for f in features if f.startswith("observation.images.")]

    # Edit meta/info.json
    num_camera_views = len(camera_views) - len(camera_names)
    info["total_videos"] = total_episodes * num_camera_views
    for camera_name in camera_names:
        features.pop(f"observation.images.{camera_name}")

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    print("[INFO] Updated 'meta/info.json'")

    # Load meta/episodes_stats.jsonl
    episode_stats_path = dataset_root / "meta/episodes_stats.jsonl"
    with open(episode_stats_path, "r") as f:
        lines = f.readlines()

    # Edit meta/episodes_stats.jsonl
    with open(episode_stats_path, "w") as f:
        for line in lines:
            info = json.loads(line)
            for camera_name in camera_names:
                info["stats"].pop(f"observation.images.{camera_name}")
            f.write(json.dumps(info) + "\n")

    print("[INFO] Updated 'meta/episodes_stats.jsonl'")

    # Find all "observation.images.{camera_name}" directory from videos
    for camera_name in camera_names:
        video_dirs = list(dataset_root.glob(f"videos/**/observation.images.{camera_name}"))
        for video_dir in video_dirs:
            if video_dir.exists() and video_dir.is_dir():
                shutil.rmtree(video_dir)
                print(f"[INFO] Removed directory: {video_dir}")


def main(args):
    check_camera_names(args.dataset_root, args.camera_names)

    if not args.no_copy_dataset:
        new_dataset_root = args.dataset_root + "_drop_view"
        if Path(new_dataset_root).exists():
            raise FileExistsError(f"{new_dataset_root} already exists.")
        shutil.copytree(args.dataset_root, new_dataset_root)
        dataset_root = new_dataset_root
        print(f"[INFO] Copied dataset to {new_dataset_root}")

    drop_view(dataset_root, args.camera_names)


if __name__ == "__main__":
    args = parse_args()
    main(args)
