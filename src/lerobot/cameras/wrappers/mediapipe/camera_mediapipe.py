import logging
import time

import cv2
import numpy as np

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

from lerobot.cameras import ColorMode
from lerobot.constants import HF_LEROBOT_HOME

from ..wrapper import WrapperCamera
from .configuration_mediapipe import MediapipeHandLandmarkerCameraConfig

logger = logging.getLogger(__name__)


class MediapipeHandLandmarkerCamera(WrapperCamera):
    MODEL_PATH = HF_LEROBOT_HOME / "mediapipe/hand_landmarker.task"
    MARGIN = 10
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54)

    STREAM_BUFFER_SIZE = 16

    def __init__(self, config: MediapipeHandLandmarkerCameraConfig):
        super().__init__(config)

        self.keypoints = config.keypoints
        self.handednesses = config.handednesses
        self.draw_handedness = config.draw_handedness
        self.draw_landmarks = config.draw_landmarks
        self.feature_keys = list(config.features.keys())

        # Download model if not exists
        if not MediapipeHandLandmarkerCamera.MODEL_PATH.exists():
            import urllib.request

            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            MediapipeHandLandmarkerCamera.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, MediapipeHandLandmarkerCamera.MODEL_PATH)

        self.landmarker_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MediapipeHandLandmarkerCamera.MODEL_PATH),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=len(config.handednesses),
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_presence_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
            result_callback=self._landmarker_callback,
        )
        self.landmarker = HandLandmarker.create_from_options(self.landmarker_options)

        self.annotated_image = None
        self.detected_landmarks = self._init_landmarks()

        self._stream_buffer: dict[str, np.ndarray] = {}
        self._last_timestamp = 0

    def __str__(self) -> str:
        return f"{self.__class__.__name__})"

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        color_mode = color_mode or self.color_mode
        start_time = time.perf_counter()
        color_image = self.camera.read(color_mode=color_mode)
        self._detect_landmarks(color_image, color_mode)
        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")
        return self.annotated_image

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        color_image = self.camera.async_read(timeout_ms=timeout_ms)
        self._detect_landmarks(color_image, self.color_mode)
        return self.annotated_image

    def get_features(self):
        features = super().get_features()
        features.update(self.detected_landmarks)
        return features

    def _put_frame(self, timestamp, rgb_image) -> None:
        self._stream_buffer[timestamp] = rgb_image
        if len(self._stream_buffer) > MediapipeHandLandmarkerCamera.STREAM_BUFFER_SIZE:
            self._stream_buffer.popitem(last=False)

    def _pop_frame(self, timestamp) -> np.ndarray | None:
        return self._stream_buffer.pop(timestamp, None)

    def _timestamp_for_landmarker(self) -> int:
        timestamp = int(time.perf_counter() * 1e3)
        if timestamp <= self._last_timestamp:
            timestamp = self._last_timestamp + 1
        self._last_timestamp = timestamp
        return timestamp

    def _init_landmarks(self) -> dict[str, float]:
        return dict.fromkeys(self.feature_keys, 0.0)

    def _landmarker_callback(self, result: HandLandmarkerResult, image: mp.Image, timestamp: int):
        self.annotated_image = self._pop_frame(timestamp)
        self.detected_landmarks = self._init_landmarks()

        if self.annotated_image is None:
            return

        hand_landmarks_list = result.hand_landmarks
        handedness_list = result.handedness
        for idx in range(len(hand_landmarks_list)):
            hand_landmark = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            category = handedness[0].category_name
            for keypoint in self.keypoints:
                ft_key = f"{category}_{keypoint}".lower()
                self.detected_landmarks[f"{ft_key}.x"] = hand_landmark[HandLandmarks[keypoint]].x
                self.detected_landmarks[f"{ft_key}.y"] = hand_landmark[HandLandmarks[keypoint]].y

            if self.draw_landmarks:
                landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                landmarks_proto.landmark.extend(
                    [landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmark]
                )
                solutions.drawing_utils.draw_landmarks(
                    self.annotated_image,
                    landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style(),
                )

                if self.draw_handedness:
                    h, w, _ = self.annotated_image.shape
                    xs = [lm.x for lm in hand_landmark]
                    ys = [lm.y for lm in hand_landmark]
                    text_x = int(min(xs) * w)
                    text_y = int(min(ys) * h) - self.MARGIN

                    cv2.putText(
                        self.annotated_image,
                        f"{handedness[0].category_name}",
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_DUPLEX,
                        self.FONT_SIZE,
                        self.HANDEDNESS_TEXT_COLOR,
                        self.FONT_THICKNESS,
                        cv2.LINE_AA,
                    )

    def _detect_landmarks(self, color_image: np.ndarray, color_mode: ColorMode) -> None:
        timestamp = self._timestamp_for_landmarker()
        rgb_image = color_image
        if color_mode == ColorMode.BGR:
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        self._put_frame(timestamp, rgb_image)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        self.landmarker.detect_async(mp_image, timestamp)

        if self.draw_landmarks:
            if self.annotated_image is None:
                self.annotated_image = color_image
            elif color_mode == ColorMode.BGR:
                self.annotated_image = cv2.cvtColor(self.annotated_image, cv2.COLOR_RGB2BGR)
        else:
            self.annotated_image = color_image

