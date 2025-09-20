import json
from pathlib import Path


def check_camera_names(dataset_root: str, camera_names: list[str]):
    """Check if the provided camera names exist in 'meta/info.json' of the dataset."""
    dataset_root = Path(dataset_root)
    with open(dataset_root / "meta/info.json", "r") as f:
        info = json.load(f)

    features = info["features"]
    existing_cameras = [f.split(".")[2] for f in features if f.startswith("observation.images.")]
    for camera_name in camera_names:
        if camera_name not in existing_cameras:
            raise ValueError(f"Camera '{camera_name}' not found in dataset.")
