# Miscellaneous

## Contents

- [Dataset editing](#dataset-editing)
- [Post-processing](#post-processing)

## Dataset editing
### drop_view.py
Drop specified views from a dataset.

```sh
python misc/drop_view.py \
    --dataset_root <path_to_dataset> \
    --camera_names <view1> <view2> ... \
    [--no_copy_dataset]
```

**Arguments**:
- `--dataset_root`: Path to the dataset directory.
- `--camera_names`: List of view names to be dropped from the dataset.
- `--no_copy_dataset`: If specified, the dataset will be modified in-place. Otherwise, a copy of the dataset will be created with the specified views removed.

## Post-processing
### add_hand_landmarks.py
Add hand landmarks to a dataset using MediaPipe.

```sh
python misc/add_hand_landmarks.py \
    --dataset_root <path_to_dataset> \
    --camera_names <view1> <view2> ... \
    --keypoints <keypoint1> <keypoint2> ... \
    --handedness <Left|Right> <Left|Right> \
    [--min_detection_confidence <float>] \
    [--min_presence_confidence <float>] \
    [--min_tracking_confidence <float>] \
    [--no_copy_dataset] \
    [--no_append_keypoints] \
    [--draw_landmarks] \
    [--draw_handedness]
```

**Arguments**:
- `--dataset_root`: Path to the dataset directory.
- `--camera_names`: List of view names to process.
- `--keypoints`: List of keypoint names to add.
    + Candidates of keypoints are shown in [Hand landmarks detection guide](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models).
- `--handedness`: List indicating the handedness for each keypoint ("Left" or "Right").
- `--min_detection_confidence`: Minimum confidence value for hand detection (default: 0.5).
- `--min_presence_confidence`: Minimum confidence value for hand presence (default: 0.5).
- `--min_tracking_confidence`: Minimum confidence value for hand tracking (default: 0.5).
- `--no_copy_dataset`: If specified, the dataset will be modified in-place. Otherwise, a copy of the dataset will be created with the new keypoints added.
- `--no_append_keypoints`: If specified, existing keypoints will be overwritten instead of appended.
- `--draw_landmarks`: If specified, annotated videos with hand landmarks will be generated.
- `--draw_handedness`: If specified, handedness labels will be drawn on the annotated videos.