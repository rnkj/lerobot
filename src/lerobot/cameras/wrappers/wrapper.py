from .. import Camera, ColorMode
from ..utils import make_cameras_from_configs
from .configs import WrapperCameraConfig


class WrapperCamera(Camera):
    def __init__(self, config: WrapperCameraConfig):
        super().__init__(config)

        camera_dict = make_cameras_from_configs({"unwrapped": config.camera_config})
        self.camera = camera_dict["unwrapped"]

        self.width = self.camera.width
        self.height = self.camera.height
        self.fps = self.camera.fps
        self.color_mode = getattr(self.camera, "color_mode", ColorMode.RGB)

    @property
    def is_connected(self) -> bool:
        return self.camera.is_connected

    @staticmethod
    def find_cameras() -> list[Camera]:
        raise NotImplementedError("Camera wrappers do not support find_cameras()")

    def connect(self, warmup: bool = True) -> None:
        self.camera.connect(warmup=warmup)

    def disconnect(self):
        self.camera.disconnect()

    def get_features(self) -> dict[str, float]:
        return {}

