#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import DiffusionConfig


@PreTrainedConfig.register_subclass("humanplay")
@dataclass
class HumanPlayConfig(DiffusionConfig):
    """Configuration class for HumanPlayConfig.

    The configuration is based on the diffusion policy.
    """

    # Extra dataset path which constrains the gap between the human and robot domains.
    robot_dataset_path: str | None = None

    # Weighting factor for the robot dataset loss.
    kl_loss_weight: float = 0.1


@PreTrainedConfig.register_subclass("mimicplay-diffusion")
@dataclass
class MimicPlayDiffusionConfig(DiffusionConfig):
    """Configuration class for MimicPlay Diffusion Policy.

    The configuration is based on the diffusion policy.
    """
    # Architecture / modeling.
    # Pretrained high-level planner
    image_for_latent_planner: str | None = None
    latent_planner_path: str | None = None

    def __post_init__(self):
        super().__post_init__()

        # Forcely use separate RGB encoders per camera for MimicPlay Diffusion.
        self.use_separate_rgb_encoder_per_camera = True

        # Check that the pretrained latent planner
        if self.latent_planner_path is not None:
            if self.image_for_latent_planner is None:
                raise ValueError(
                    "If `latent_planner_path` is provided, "
                    "`image_for_latent_planner` must also be provided."
                )

            # if self.image_for_latent_planner not in self.image_features:
            #     raise ValueError(
            #         f"`image_for_latent_planner` ({self.image_for_latent_planner}) "
            #         f"must be one of the `image_features` ({list(self.image_features.keys())}) provided."
            #     )
