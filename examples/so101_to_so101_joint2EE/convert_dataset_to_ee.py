#!/usr/bin/env python
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.datasets.compute_stats import aggregate_feature_stats, get_feature_stats
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    robot_action_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    ForwardKinematicsJointsToEE,
    InverseKinematicsEEToJoints,
)
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.teleoperators.so101_leader.config_so101_leader import SO101LeaderConfig
from lerobot.teleoperators.so101_leader.so101_leader import SO101Leader


SO_ARM100_ROOT = "./sandbox/SO-ARM100"
HF_REPO_ID = "hf_username/repo_id"
DATASET_ROOT = "dataset/HRC/sort_block/robot/large_block_robot_1212_260ep"

def main():
    # Load existing dataset
    dataset = LeRobotDataset(
        repo_id=HF_REPO_ID,
        root=DATASET_ROOT,
    )
    
    # Create the teleoperator configurations
    follower_config = SO101FollowerConfig(
        port="/dev/tty.dummyfollower",
        id="follower_arm",
        use_degrees=True,
    )
    leader_config = SO101LeaderConfig(
        port="/dev/tty.dummyleader", id="leader_arm"
    )

    # Initialize the teleoperator
    follower = SO101Follower(follower_config)
    leader = SO101Leader(leader_config)

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    urdf_path = Path(SO_ARM100_ROOT) / "Simulation/SO101/so101_new_calib.urdf"
    follower_kinematics_solver = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=list(follower.bus.motors.keys()),
    )

    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    leader_kinematics_solver = RobotKinematics(
        urdf_path=str(urdf_path),
        target_frame_name="gripper_frame_link",
        joint_names=list(leader.bus.motors.keys()),
    )

    # Build pipeline to convert follower joints to EE observation
    follower_joints_to_ee = RobotProcessorPipeline[RobotObservation, RobotObservation](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=follower_kinematics_solver, motor_names=list(follower.bus.motors.keys())
            ),
        ],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    # Build pipeline to convert leader joints to EE action
    leader_joints_to_ee = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            ForwardKinematicsJointsToEE(
                kinematics=leader_kinematics_solver, motor_names=list(leader.bus.motors.keys())
            ),
        ],
        to_transition=robot_action_to_transition,
        to_output=transition_to_robot_action,
    )

    # Build pipeline to convert EE action to follower joints
    ee_to_follower_joints = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        [
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,
            ),
            InverseKinematicsEEToJoints(
                kinematics=follower_kinematics_solver,
                motor_names=list(follower.bus.motors.keys()),
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    ## Build pipeline to convert leader joints to EE action
    #leader_joints_to_ee = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
    #    steps=[
    #        ForwardKinematicsJointsToEE(
    #            kinematics=leader_kinematics_solver, motor_names=list(leader.bus.motors.keys())
    #        ),
    #    ],
    #    to_transition=robot_action_to_transition,
    #    to_output=transition_to_robot_action,
    #)

    # Copy existing dataset to new location
    old_root = Path(DATASET_ROOT)
    new_root = Path(DATASET_ROOT + "_ee")

    if new_root.is_dir():
        raise IOError(f"Dataset {new_root} already exists.")

    shutil.copytree(old_root, new_root)

    # Update metadata
    info_path = new_root / "meta/info.json"

    with open(info_path, "r") as f:
        info = json.load(f)

    info["features"]["action"] = {
        "dtype": "float32",
        "names": [
            "ee.x",
            "ee.y",
            "ee.z",
            "ee.wx",
            "ee.wy",
            "ee.wz",
            "ee.gripper_pos",
        ],
        "shape": [7],
    }
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "names": [
            "ee.x",
            "ee.y",
            "ee.z",
            "ee.wx",
            "ee.wy",
            "ee.wz",
            "ee.gripper_pos",
        ],
        "shape": [7],
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    # Convert action from leader joints to EE 
    data_paths = sorted(new_root.glob("data/**/file-*.parquet"))
    for data_path in data_paths:
        data = pd.read_parquet(data_path)

        # replace "data" with "meta/episodes"
        rel_path = data_path.relative_to(new_root)
        meta_path = new_root / str(rel_path).replace("data", "meta/episodes")
        meta = pd.read_parquet(meta_path)

        dataset_from_index = meta["dataset_from_index"].to_numpy()

        # Convert robot states and actions from joint to EE positions
        ee_states_all = [None] * len(dataset_from_index)
        ee_actions_all = [None] * len(dataset_from_index)
        ee_state_stats_all = [None] * len(dataset_from_index)
        ee_action_stats_all = [None] * len(dataset_from_index)
        for episode_i in range(len(dataset_from_index)):
            start = dataset_from_index[episode_i]
            stop = (
                dataset_from_index[episode_i + 1]
                if episode_i + 1 < len(dataset_from_index)
                else len(data["action"])
            )

            # Convert joint actions to EE actions
            ee_state_list = [None] * (stop - start)
            ee_action_list = [None] * (stop - start)

            for i, ts in enumerate(range(start, stop)):
                state = data["observation.state"][ts]
                state_dict = {
                    f"{key}.pos": float(state[i])
                    for i, key in enumerate(follower.bus.motors.keys())
                }
                ee_state = follower_joints_to_ee(state_dict)
                ee_state_list[i] = [value for value in ee_state.values()]

                action = data["action"][ts]
                action_dict = {
                    f"{key}.pos": float(action[i])
                    for i, key in enumerate(leader.bus.motors.keys())
                }
                ee_action = leader_joints_to_ee(action_dict)
                ee_action_list[i] = [value for value in ee_action.values()]

            ee_states = np.array(ee_state_list, dtype=np.float32)
            ee_states_all[episode_i] = ee_states

            ee_actions = np.array(ee_action_list, dtype=np.float32)
            ee_actions_all[episode_i] = ee_actions

            # Compute statistics of the episode
            ee_state_stats = get_feature_stats(ee_states, axis=0, keepdims=False)
            ee_state_stats_all[episode_i] = ee_state_stats
            ee_action_stats = get_feature_stats(ee_actions, axis=0, keepdims=False)
            ee_action_stats_all[episode_i] = ee_action_stats

        # Update aggregated stats in stats.json
        with open(new_root / "meta/stats.json", "r") as f:
            stats = json.load(f)

        aggregated_ee_state_stats = aggregate_feature_stats(ee_state_stats_all)
        stats["observation.state"] = {
            key: value.tolist() for key, value in aggregated_ee_state_stats.items()
        }

        aggregated_ee_action_stats = aggregate_feature_stats(ee_action_stats_all)
        stats["action"] = {
            key: value.tolist() for key, value in aggregated_ee_action_stats.items()
        }

        with open(new_root / "meta/stats.json", "w") as f:
            json.dump(stats, f, indent=4)

        # Save converted EE actions
        data["action"] = np.concatenate(ee_actions_all, axis=0)
        data["observation.state"] = np.concatenate(ee_states_all, axis=0)
        data.to_parquet(data_path)

        # Save updated episode metadata
        for key in ("mean", "std", "min", "max", "count"):
            meta[f"stats/observation.state/{key}"] = np.array(
                [ee_stats[key] for ee_stats in ee_state_stats_all]
            )
            meta[f"stats/action/{key}"] = np.array(
                [ee_stats[key] for ee_stats in ee_action_stats_all]
            )
        meta.to_parquet(meta_path)

        print("[INFO] Done")

if __name__ == "__main__":
    main()
