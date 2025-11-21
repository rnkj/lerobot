import argparse
import json
from pathlib import Path

import h5py


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build human play dataset for LeRobot"
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        required=True,
        help="Path to the LeRobot dataset which includes the human play videos",
    )
    parser.add_argument(
        "--human_play_hdf5",
        type=str,
        required=True,
        help="Path to the hdf5 file which includes the human play data",
    )
    return parser.parse_args()

def update_info_json(info_path, camera_view1, camera_view2):
    """Update info.json based on the selected two camera views."""
    with open(info_path, "r") as f:
        info = json.load(f)

    info["total_videos"] = 2 * info["total_episodes"]

    action_ft = info["features"]["action"]
    action_ft["shape"] = [4]
    action_ft["names"] = [
        f"{camera_view1}.x",
        f"{camera_view1}.y",
        f"{camera_view2}.x",
        f"{camera_view2}.y",
    ]

    state_ft = info["features"]["observation.action"]
    state_ft["shape"] = [4]
    state_ft["names"] = [
        f"{camera_view1}.x",
        f"{camera_view1}.y",
        f"{camera_view2}.x",
        f"{camera_view2}.y",
    ]

    # Remove image observation features for cameras not matching the two selected views
    keys_to_remove = []
    for key in info["features"].keys():
        if key.startswith("observation.images."):
            camera_name = key.replace("observation.images.", "")
            if camera_name != camera_view1 and camera_name != camera_view2:
                keys_to_remove.append(key)

    for key in keys_to_remove:
        info["features"].pop(key)

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)


def main():
    args = parse_arguments()

    source_dataset_dir = Path(args.source_dataset)
    human_play_hdf5_file = Path(args.human_play_hdf5)

    # load hdf5
    hdf5 = h5py.File(human_play_hdf5_file, "r")
    camera_view1 = hdf5.attrs["camera_view1"]
    camera_view2 = hdf5.attrs["camera_view2"]

    # update info.json
    info_path = source_dataset_dir / "meta/info.json"
    update_info_json(info_path, camera_view1, camera_view2)

    # update episode_stats.jsonl
    episode_stats_path = source_dataset_dir / "meta/episode_stats.jsonl"
    with open(episode_stats_path, "r") as f:
        episode_stats = [json.loads(line) for line in f.readlines()]
        for episode_stat in episode_stats:

if __name__ == "__main__":