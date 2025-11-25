import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import h5py
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Build human play dataset for LeRobot"
    )
    parser.add_argument(
        "--source_dataset",
        "--src",
        type=str,
        required=True,
        help="Path to the LeRobot dataset which includes the human play videos",
    )
    parser.add_argument(
        "--destination_dataset",
        "--dst",
        type=str,
        default=None,
        help="Path to the output LeRobot dataset which includes human hand locations as the actions"
    )
    parser.add_argument(
        "--human_play_hdf5",
        type=str,
        required=True,
        help="Path to the hdf5 file which includes the human play data",
    )
    return parser.parse_args()

def update_info_json(info_path, camera_view1, camera_view2, total_frames):
    """Update info.json based on the selected two camera views."""
    with open(info_path, "r") as f:
        info = json.load(f)

    info["total_frames"] = total_frames
    info["total_videos"] = 2 * info["total_episodes"]

    action_ft = info["features"]["action"]
    action_ft["shape"] = [4]
    action_ft["names"] = [
        f"{camera_view1}.x",
        f"{camera_view1}.y",
        f"{camera_view2}.x",
        f"{camera_view2}.y",
    ]

    state_ft = info["features"]["observation.state"]
    state_ft["shape"] = [0]
    state_ft["names"] = []

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

    if args.destination_dataset is None:
        # append "_human_play" to the source dataset directory name
        destination_dataset_dir = Path(str(source_dataset_dir) + "_human_play")
    else:
        destination_dataset_dir = Path(args.destination_dataset)

    # Create destination dataset directory
    if destination_dataset_dir.exists():
        raise FileExistsError(
            f"Destination dataset directory {destination_dataset_dir} already exists."
        )
    destination_dataset_dir.mkdir(parents=True, exist_ok=False)

    # Load hdf5 and update parquet files
    with h5py.File(human_play_hdf5_file, "r") as hdf5:
        # get attributes
        camera_view1 = hdf5.attrs["camera_view1"]
        camera_view2 = hdf5.attrs["camera_view2"]
        # height = hdf5.attrs["camera_height"]
        # width = hdf5.attrs["camera_width"]

        # statistics
        stats_list = []
        episode_count = 0

        (destination_dataset_dir / "data").mkdir(parents=True, exist_ok=False)
        source_data_dir = source_dataset_dir / "data"
        for src_chunk_dir in sorted(source_data_dir.iterdir()):
            if not src_chunk_dir.is_dir():
                continue

            dst_chunk_dir = destination_dataset_dir / "data" / src_chunk_dir.name
            dst_chunk_dir.mkdir(parents=True, exist_ok=False)

            for demo_i, parquet_file in enumerate(sorted(src_chunk_dir.iterdir())):
                df = pd.read_parquet(parquet_file)

                # Drop the last row of all DataFrame keys
                df = df.iloc[:-1]

                # Update action and observation.state columns
                hand_loc = hdf5[f"data/demo_{demo_i}/hand_loc"][:].squeeze()
                state = np.empty((len(hand_loc), 0), dtype=np.float32)
                df["action"] = hand_loc.tolist()
                df["observation.state"] = state.tolist()

                # Save updated parquet
                destination_parquet_file = dst_chunk_dir / parquet_file.name
                df.to_parquet(destination_parquet_file)

                stats_dict = {
                    "action": {
                        "mean": hand_loc.mean(axis=0).tolist(),
                        "std": hand_loc.std(axis=0).tolist(),
                        "min": hand_loc.min(axis=0).tolist(),
                        "max": hand_loc.max(axis=0).tolist(),
                        "count": [hand_loc.shape[0]],
                    },
                    "observation.state": {
                        "mean": state.mean(axis=0).tolist(),
                        "std": state.std(axis=0).tolist(),
                        "min": state.min(axis=0).tolist(),
                        "max": state.max(axis=0).tolist(),
                        "count": [state.shape[0]],
                    },
                }
                stats_list.append(stats_dict)

                episode_count += 1

    # Copy all videos which matches the two selected camera views
    source_videos_dir = source_dataset_dir / "videos"
    destination_videos_dir = destination_dataset_dir / "videos"

    # Load hdf5 to get camera view names first
    with h5py.File(human_play_hdf5_file, "r") as temp_hdf5:
        camera_view1 = temp_hdf5.attrs["camera_view1"]
        camera_view2 = temp_hdf5.attrs["camera_view2"]

    # Iterate through chunk directories
    for chunk_dir in sorted(source_videos_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue

        # Copy only the two selected camera view directories
        for camera_view in [camera_view1, camera_view2]:
            camera_view_key = f"observation.images.{camera_view}"
            source_camera_dir = chunk_dir / camera_view_key

            if source_camera_dir.exists():
                destination_camera_dir = destination_videos_dir / chunk_dir.name / camera_view_key
                shutil.copytree(source_camera_dir, destination_camera_dir)

    # Copy meta tree
    shutil.copytree(source_dataset_dir / "meta", destination_dataset_dir / "meta")

    # update episodes.jsonl
    src_episodes_path = source_dataset_dir / "meta/episodes.jsonl"
    dst_episodes_path = destination_dataset_dir / "meta/episodes.jsonl"
    total_frames = 0
    with open(src_episodes_path, "r") as fr, open(dst_episodes_path, "w") as fw:
        for line in fr.readlines():
            ep_dict = json.loads(line)
            ep_dict["length"] -= 1  # drop the last frame
            fw.write(json.dumps(ep_dict) + "\n")
            total_frames += ep_dict["length"]

    # update info.json
    info_path = destination_dataset_dir / "meta/info.json"
    update_info_json(info_path, camera_view1, camera_view2, total_frames)

    # update episode_stats.jsonl
    src_episodes_stats_path = source_dataset_dir / "meta/episodes_stats.jsonl"
    dst_episodes_stats_path = destination_dataset_dir / "meta/episodes_stats.jsonl"
    with open(src_episodes_stats_path, "r") as fr, open(dst_episodes_stats_path, "w") as fw:
        for stats_i, line in enumerate(fr.readlines()):
            stats_dict = json.loads(line)
            stats_dict["stats"].update(stats_list[stats_i])
            fw.write(json.dumps(stats_dict) + "\n")

if __name__ == "__main__":
    main()
