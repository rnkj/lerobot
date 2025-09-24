import logging
from dataclasses import dataclass
from typing import Any

try:
    import mediapipe
    HandLandmarks = mediapipe.solutions.hands.HandLandmark
except Exception as e:
    logging.info(f"Could not import mediapipe: {e}")

from lerobot.cameras import CameraConfig

from ..configs import WrapperCameraConfig


@CameraConfig.register_subclass("mediapipe_hand")
@dataclass
class MediapipeHandLandmarkerCameraConfig(WrapperCameraConfig):
    """Configuration class for cameras using MediaPipe for hand landmark detection.

    This class provides specialized configuration options for cameras that utilize
    MediaPipe's hand landmark detection capabilities.
    """

    min_detection_confidence: float = 0.5
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5

    keypoints: tuple[str, ...] = ("WRIST",)
    handednesses: tuple[str, ...] = ("Left", "Right")

    no_append_keypoints: bool = False
    draw_handedness: bool = False
    draw_landmarks: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.no_append_keypoints and (not self.draw_landmarks):
            raise ValueError(
                "Appending keypoints or drawing landmarks must be available"
            )
        for keypoint in self.keypoints:
            if keypoint not in HandLandmarks.__members__:
                raise ValueError(
                    f"Invalid keypoint: {keypoint}. "
                    f"Available keypoints are [{' | '.join(HandLandmarks.__members__.keys())}]"
                )
        for hand in self.handednesses:
            if hand not in ("Left", "Right"):
                raise ValueError(
                    f"Invalid hand: {hand}. Available hands are [Left | Right]"
                )

    @property
    def features(self) -> dict[str, Any]:
        ft = super().features
        if not self.no_append_keypoints:
            for handedness in self.handednesses:
                for keypoint in self.keypoints:
                    pair = f"{handedness}_{keypoint}".lower()
                    d = {f"{pair}.x": float, f"{pair}.y": float}
                    ft.update(d)
        return ft

