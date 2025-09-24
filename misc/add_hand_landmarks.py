"""
add_hand_landmarks.py

Description:
    This script processes a dataset containing video observations from multiple cameras and appends hand landmark keypoints to the observation state using MediaPipe's Hand Landmarker. It can also annotate videos with detected hand landmarks and handedness labels. The script updates dataset metadata and statistics accordingly.

Usage:
    python add_hand_landmarks.py --dataset_root <path_to_dataset> --camera_names <camera1> <camera2> [options]

Arguments:
    --dataset_root (str, required): Path to the dataset root directory.
    --camera_names (str, nargs="+", required): List of camera names to track hands.
    --min_detection_confidence (float, default=0.5): Minimum detection confidence for hand detection.
    --min_presence_confidence (float, default=0.5): Minimum presence confidence for hand detection.
    --min_tracking_confidence (float, default=0.5): Minimum tracking confidence for hand detection.
    --keypoints (str, nargs="+", default=["WRIST"]): Keypoints to track. Choices are MediaPipe hand landmark names.
    --handednesses (str, nargs="+", default=["Left", "Right"]): Handednesses to track. Choices: "Left", "Right".
    --no_copy_dataset (flag): If set, do not copy the dataset before processing.
    --no_append_keypoints (flag): If set, do not append keypoints to the output.
    --draw_handedness (flag): Whether to draw handedness on the image (requires --draw_landmarks).
    --draw_landmarks (flag): Whether to draw landmarks on the image.

Notes:
    - Requires MediaPipe, OpenCV, Pandas, NumPy, imageio, tqdm, and other dependencies.
    - The script will download the MediaPipe hand landmarker model if not present.
    - The script modifies the dataset in-place unless --no_copy_dataset is specified.
    - Annotated videos will overwrite originals unless --draw_landmarks is not set.
"""

import json
import logging
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from lerobot.cameras.wrappers.mediapipe import MediapipeHandLandmarkerCamera
from lerobot.datasets.image_writer import AsyncImageWriter
from lerobot.datasets.video_utils import encode_video_frames

try:
    import mediapipe as mp
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
    HandLandmarks = solutions.hands.HandLandmark
    RunningMode = mp.tasks.vision.RunningMode
except Exception as e:
    logging.info(f"Could not import mediapipe: {e}")

from utils import check_camera_names


def parse_args():
    parser = ArgumentParser(description="A simple argument parser example.")

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        required=True,
        help="List of camera names to track hands.",
    )

    parser.add_argument(
        "--min_detection_confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence.",
    )
    parser.add_argument(
        "--min_presence_confidence",
        type=float,
        default=0.5,
        help="Minimum presence confidence.",
    )
    parser.add_argument(
        "--min_tracking_confidence",
        type=float,
        default=0.5,
        help="Minimum tracking confidence.",
    )

    parser.add_argument(
        "--keypoints",
        type=str,
        nargs="+",
        default=["WRIST"],
        choices=list(HandLandmarks.__members__.keys()),
        help="Keypoints to track.",
    )
    parser.add_argument(
        "--handednesses",
        type=str,
        nargs="+",
        default=["Left", "Right"],
        choices=["Left", "Right"],
        help="Handednesses to track.",
    )

    parser.add_argument(
        "--no_copy_dataset",
        action="store_true",
        help="If set, do not copy the dataset before processing.",
    )
    parser.add_argument(
        "--no_append_keypoints",
        action="store_true",
        help="If set, do not append keypoints to the output.",
    )
    parser.add_argument(
        "--draw_handedness",
        action="store_true",
        help="Whether to draw handedness on the image. If --draw_landmarks is False, this flag is ignored.",
    )
    parser.add_argument(
        "--draw_landmarks",
        action="store_true",
        help="Whether to draw landmarks on the image.",
    )

    parser.add_argument(
        "--num_image_writer_processes",
        type=int,
        default=0,
        help="Number of processes for AsyncImageWriter. 0 means no multiprocessing.",
    )
    parser.add_argument(
        "--num_image_writer_threads",
        type=int,
        default=4,
        help="Number of threads for AsyncImageWriter.",
    )

    return parser.parse_args()


def build_landmarker(
    handednesses: list[str],
    min_detection_confidence: float,
    min_presence_confidence: float,
    min_tracking_confidence: float,
):
    landmarker_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MediapipeHandLandmarkerCamera.MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=len(handednesses),
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return HandLandmarker.create_from_options(landmarker_options)


def draw_landmarks_on_image(rgb_image, detection_result, draw_handedness=False):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

        if draw_handedness:
            h, w, _ = annotated_image.shape
            xs = [lm.x for lm in hand_landmarks]
            ys = [lm.y for lm in hand_landmarks]
            text_x = int(min(xs) * w)
            text_y = int(min(ys) * h) - MediapipeHandLandmarkerCamera.MARGIN

            cv2.putText(
                annotated_image,
                f"{handedness[0].category_name}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_DUPLEX,
                MediapipeHandLandmarkerCamera.FONT_SIZE,
                MediapipeHandLandmarkerCamera.HANDEDNESS_TEXT_COLOR,
                MediapipeHandLandmarkerCamera.FONT_THICKNESS,
                cv2.LINE_AA,
            )
    return annotated_image


def process_hand_landmarks(args, camera_name: str):
    dataset_root = Path(args.dataset_root)

    # Load meta file
    with open(dataset_root / "meta/episodes_stats.jsonl") as f:
        episode_stats = [json.loads(line) for line in f.readlines()]

    # Find all parquets and videos
    parquet_files = sorted(dataset_root.glob("data/**/*.parquet"))
    video_files = sorted(dataset_root.glob(f"videos/**/observation.images.{camera_name}/*.mp4"))

    # Instantiate image writer
    if args.draw_landmarks:
        writer = AsyncImageWriter(
            num_processes=args.num_image_writer_processes,
            num_threads=args.num_image_writer_threads,
        )

    for ep_i, (pf, vf) in tqdm(
        enumerate(zip(parquet_files, video_files, strict=True)), total=len(parquet_files)
    ):
        # Instantiate landmarker
        landmarker = build_landmarker(
            args.handednesses,
            args.min_detection_confidence,
            args.min_presence_confidence,
            args.min_tracking_confidence,
        )

        # Load existing observation.state
        if not args.no_append_keypoints:
            parquet = pd.read_parquet(pf)
            states = parquet["observation.state"]

            # Initialize landmarks
            landmark_array = np.zeros(
                len(args.handednesses) * len(args.keypoints) * 2,
                dtype=np.float32,
            )
            # Mask array for calculation of stats
            is_detected = np.zeros((len(states), len(args.handednesses)), dtype=bool)

        # Load video
        decoder = VideoDecoder(vf)
        metadata = decoder.metadata
        average_fps = metadata.average_fps
        ms_per_frame = 1000.0 / average_fps

        if args.draw_landmarks:
            # Make temporal image folder
            img_dir = Path(str(vf).replace("videos", "images").rsplit(".", 1)[0])
            os.makedirs(img_dir, exist_ok=True)

        for frame_i in range(len(decoder)):
            # Read frame
            frame_batch = decoder.get_frame_at(frame_i)
            frame = frame_batch.data.numpy().transpose(1, 2, 0)
            ts = round(frame_i * ms_per_frame)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # Detect hand landmarks
            result = landmarker.detect_for_video(mp_image, ts)

            if not args.no_append_keypoints:
                # Append landmarks to observation.state
                landmark_array = landmark_array.copy()

                for handedness, hand_landmarks in zip(result.handedness, result.hand_landmarks, strict=True):
                    category = handedness[0].category_name
                    if category in args.handednesses:
                        hand_i = args.handednesses.index(category)
                        # Append keypoints
                        start = hand_i * len(args.keypoints) * 2
                        for key_i, keypoint in enumerate(args.keypoints):
                            idx = start + 2 * key_i
                            landmark = hand_landmarks[HandLandmarks[keypoint]]
                            landmark_array[idx] = landmark.x
                            landmark_array[idx + 1] = landmark.y
                        # Keypoints detected
                        is_detected[frame_i, hand_i] = True

                states.at[frame_i] = np.concatenate((states.at[frame_i], landmark_array), dtype=np.float32)

            if args.draw_landmarks:
                # Draw landmarks
                annotated_rgb = draw_landmarks_on_image(frame, result, args.draw_handedness)
                writer.save_image(annotated_rgb, img_dir / f"frame_{frame_i:06d}.png")

        landmarker.close()

        if not args.no_append_keypoints:
            # Synchronize image writing
            writer.wait_until_done()

            # Save updated parquet
            parquet["observation.state"] = states
            parquet.to_parquet(pf)
            logging.info(f"Updated '{pf.name}' with hand keypoints")

            # Update episode stats
            stats = episode_stats[ep_i]["stats"]
            state_stats = stats["observation.state"]

            state_len = len(state_stats["max"])
            landmark_seq = np.stack(states.to_numpy())[:, state_len:]

            # Calculate only for detected landmarks
            for hand_i in range(len(args.handednesses)):
                start = hand_i * len(args.keypoints) * 2
                stop = (hand_i + 1) * len(args.keypoints) * 2

                detected_landmarks = landmark_seq[is_detected[:, hand_i], start:stop]
                state_stats["max"].extend(detected_landmarks.max(axis=0).tolist())
                state_stats["min"].extend(detected_landmarks.min(axis=0).tolist())
                state_stats["mean"].extend(detected_landmarks.mean(axis=0).tolist())
                state_stats["std"].extend(detected_landmarks.std(axis=0).tolist())

        if args.draw_landmarks:
            # Save annotated video
            encode_video_frames(img_dir, vf, round(average_fps), overwrite=True)
            shutil.rmtree(img_dir)
            logging.info(f"Saved annotated video to {vf}")

    # Save updated episode stats
    if not args.no_append_keypoints:
        with open(dataset_root / "meta/episodes_stats.jsonl", "w") as f:
            for stats in episode_stats:
                f.write(json.dumps(stats) + "\n")
        logging.info("Updated 'meta/episodes_stats.jsonl'")

    # Remove temporal image folder
    if args.draw_landmarks:
        shutil.rmtree(dataset_root / "images", ignore_errors=True)


def main(args):
    check_camera_names(args.dataset_root, args.camera_names)

    # Backup dataset
    if not args.no_copy_dataset:
        new_dataset_root = args.dataset_root + "_with_hands"
        if Path(new_dataset_root).exists():
            raise FileExistsError(f"{new_dataset_root} already exists.")
        shutil.copytree(args.dataset_root, new_dataset_root)
        args.dataset_root = new_dataset_root
        logging.info(f"Copied dataset to {new_dataset_root}")

    # Download model if not exists
    model_path = MediapipeHandLandmarkerCamera.MODEL_PATH
    if not model_path.exists():
        import urllib.request

        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, model_path)

    # Load meta files
    dataset_root = Path(args.dataset_root)

    with open(dataset_root / "meta/info.json") as f:
        info = json.load(f)

    # Append keypoints to observation.states
    if not args.no_append_keypoints:
        features = info["features"]
        ft_obs_st = features["observation.state"]
        obs_state_names = ft_obs_st["names"]

        for camera_name in args.camera_names:
            for handedness in args.handednesses:
                for keypoint in args.keypoints:
                    feature_name = f"{camera_name}_{handedness}_{keypoint}".lower()
                    if feature_name not in obs_state_names:
                        obs_state_names.extend([f"{feature_name}.x", f"{feature_name}.y"])
        ft_obs_st["names"] = obs_state_names
        ft_obs_st["shape"] = [len(obs_state_names)]

        with open(dataset_root / "meta/info.json", "w") as f:
            json.dump(info, f, indent=4)
        logging.info("Updated 'meta/info.json'")

    # Process hand landmarks for each camera
    for camera_name in args.camera_names:
        logging.info(f"Processing camera '{camera_name}'")
        process_hand_landmarks(args, camera_name)


if __name__ == "__main__":
    args = parse_args()
    main(args)
