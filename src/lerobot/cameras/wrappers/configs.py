import abc
from dataclasses import dataclass
from typing import Any

from lerobot.cameras import CameraConfig


@dataclass
class WrapperCameraConfig(CameraConfig, abc.ABC):
    camera_config: CameraConfig

    def __post_init__(self):
        self.width = self.camera_config.width
        self.height = self.camera_config.height
        self.fps = self.camera_config.fps

    @property
    def features(self) -> dict[str, Any]:
        return {}

