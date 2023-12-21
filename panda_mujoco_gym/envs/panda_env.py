import mujoco
import numpy as np
from gymnasium.core import ObsType
from gymnasium_robotics.envs.robot_env import MujocoRobotEnv
from gymnasium_robotics.utils import rotations
from typing import Optional, Any, SupportsFloat

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.5,
    "azimuth": 135.0,
    "elevation": -20.0,
    "lookat": np.array([0.0, 0.5, 0.0]),
}


class FrankaEnv(MujocoRobotEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        model_path: str = None,
        n_substeps: int = 50,
        reward_type: str = "sparse",
        block_gripper: bool = False,
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        obj_xy_range: float = 0.3,
        goal_x_offset: float = 0.4,
        goal_z_range: float = 0.2,
        **kwargs,
    ):
        self.block_gripper = block_gripper
        self.model_path = model_path

        action_size = 3
        action_size += 0 if self.block_gripper else 1

        self.reward_type = reward_type

        self.neutral_joint_values = np.array([0.00, 0.41, 0.00, -1.85, 0.00, 2.26, 0.79, 0.00, 0.00])

        super().__init__(
            n_actions=action_size,
            n_substeps=n_substeps,
            model_path=self.model_path,
            initial_qpos=self.neutral_joint_values,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self.distance_threshold = distance_threshold

        # sample areas for the object and goal target
        self.obj_xy_range = obj_xy_range
        self.goal_xy_range = goal_xy_range
        self.goal_x_offset = goal_x_offset
        self.goal_z_range = goal_z_range

        self.goal_range_low = np.array([-self.goal_xy_range / 2 + goal_x_offset, -self.goal_xy_range / 2, 0])
        self.goal_range_high = np.array([self.goal_xy_range / 2 + goal_x_offset, self.goal_xy_range / 2, self.goal_z_range])
        self.obj_range_low = np.array([-self.obj_xy_range / 2, -self.obj_xy_range / 2, 0])
        self.obj_range_high = np.array([self.obj_xy_range / 2, self.obj_xy_range / 2, 0])

        self.goal_range_low[0] += 0.6
        self.goal_range_high[0] += 0.6
        self.obj_range_low[0] += 0.6
        self.obj_range_high[0] += 0.6

        # Three auxiliary variables to understand the component of the xml document but will not be used
        # number of actuators/controls: 7 arm joints and 2 gripper joints
        self.nu = self.model.nu
        # 16 generalized coordinates: 9 (arm + gripper) + 7 (object free joint: 3 position and 4 quaternion coordinates)
        self.nq = self.model.nq
        # 9 arm joints and 6 free joints
        self.nv = self.model.nv

        # control range
        self.ctrl_range = self.model.actuator_ctrlrange

    # override the methods in MujocoRobotEnv
    # -----------------------------
    def _initialize_simulation(self) -> None:
        self.model = self._mujoco.MjModel.from_xml_path(self.fullpath)
        self.data = self._mujoco.MjData(self.model)
        self._model_names = self._utils.MujocoModelNames(self.model)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height

        # index used to distinguish arm and gripper joints
        free_joint_index = self._model_names.joint_names.index("obj_joint")
        self.arm_joint_names = self._model_names.joint_names[:free_joint_index][0:7]
        self.gripper_joint_names = self._model_names.joint_names[:free_joint_index][7:9]

        self._env_setup(self.neutral_joint_values)
        self.initial_time = self.data.time
        self.initial_qvel = np.copy(self.data.qvel)

    def _env_setup(self, neutral_joint_values) -> None:
        self.set_joint_neutral()
        self.data.ctrl[0:7] = neutral_joint_values[0:7]
        self.reset_mocap_welds(self.model, self.data)

        self._mujoco.mj_forward(self.model, self.data)

        self.initial_mocap_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()
        self.grasp_site_pose = self.get_ee_orientation().copy()

        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._mujoco_step()

        self.initial_object_height = self._utils.get_joint_qpos(self.model, self.data, "obj_joint")[2].copy()

    def step(self, action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if np.array(action).shape != self.action_space.shape:
            raise ValueError("Action dimension mismatch")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)

        self._mujoco_step(action)

        self._step_callback()

        if self.render_mode == "human":
            self.render()

        obs = self._get_obs().copy()

        info = {"is_success": self._is_success(obs["achieved_goal"], self.goal)}

        terminated = info["is_success"]
        truncated = self.compute_truncated(obs["achieved_goal"], self.goal, info)
        reward = self.compute_reward(obs["achieved_goal"], self.goal, info)

        return obs, reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info) -> SupportsFloat:
        d = self.goal_distance(achieved_goal, desired_goal)
        if self.reward_type == "sparse":
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _set_action(self, action) -> None:
        action = action.copy()
        # for the pick and place task
        if not self.block_gripper:
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            fingers_ctrl = gripper_ctrl * 0.2
            fingers_width = self.get_fingers_width().copy() + fingers_ctrl
            fingers_half_width = np.clip(fingers_width / 2, self.ctrl_range[-1, 0], self.ctrl_range[-1, 1])

        elif self.block_gripper:
            pos_ctrl = action
            fingers_half_width = 0

        # control the gripper
        self.data.ctrl[-2:] = fingers_half_width

        # control the end-effector with mocap body
        pos_ctrl *= 0.05
        pos_ctrl += self.get_ee_position().copy()
        pos_ctrl[2] = np.max((0, pos_ctrl[2]))

        self.set_mocap_pose(pos_ctrl, self.grasp_site_pose)

    def _get_obs(self) -> dict:
        # robot
        ee_position = self._utils.get_site_xpos(self.model, self.data, "ee_center_site").copy()

        ee_velocity = self._utils.get_site_xvelp(self.model, self.data, "ee_center_site").copy() * self.dt

        if not self.block_gripper:
            fingers_width = self.get_fingers_width().copy()

        # object
        # object cartesian position: 3
        object_position = self._utils.get_site_xpos(self.model, self.data, "obj_site").copy()

        # object rotations: 3
        object_rotation = rotations.mat2euler(self._utils.get_site_xmat(self.model, self.data, "obj_site")).copy()

        # object linear velocities
        object_velp = self._utils.get_site_xvelp(self.model, self.data, "obj_site").copy() * self.dt

        # object angular velocities
        object_velr = self._utils.get_site_xvelr(self.model, self.data, "obj_site").copy() * self.dt

        if not self.block_gripper:
            obs = {
                "observation": np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
                        fingers_width,
                        object_position,
                        object_rotation,
                        object_velp,
                        object_velr,
                    ]
                ).copy(),
                "achieved_goal": object_position.copy(),
                "desired_goal": self.goal.copy(),
            }
        else:
            obs = {
                "observation": np.concatenate(
                    [
                        ee_position,
                        ee_velocity,
                        object_position,
                        object_rotation,
                        object_velp,
                        object_velr,
                    ]
                ).copy(),
                "achieved_goal": object_position.copy(),
                "desired_goal": self.goal.copy(),
            }

        return obs

    def _is_success(self, achieved_goal, desired_goal) -> np.float32:
        d = self.goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self) -> None:
        # visualize goal site
        sites_offset = (self.data.site_xpos - self.model.site_pos).copy()
        site_id = self._model_names.site_name2id["target"]
        self.model.site_pos[site_id] = self.goal - sites_offset[site_id]
        self._mujoco.mj_forward(self.model, self.data)

    def _reset_sim(self) -> bool:
        self.data.time = self.initial_time
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None

        self.set_joint_neutral()
        self.set_mocap_pose(self.initial_mocap_position, self.grasp_site_pose)

        self._sample_object()

        self._mujoco.mj_forward(self.model, self.data)
        return True

    def _mujoco_step(self, action: Optional[np.ndarray] = None) -> None:
        for _ in range(10):
            self._mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)

    # custom methods
    # -----------------------------
    def reset_mocap_welds(self, model, data) -> None:
        if model.nmocap > 0 and model.eq_data is not None:
            for i in range(model.eq_data.shape[0]):
                if model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    # relative pose
                    model.eq_data[i, 3:10] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self._mujoco.mj_forward(model, data)

    def goal_distance(self, goal_a, goal_b) -> SupportsFloat:
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def set_mocap_pose(self, position, orientation) -> None:
        self._utils.set_mocap_pos(self.model, self.data, "panda_mocap", position)
        self._utils.set_mocap_quat(self.model, self.data, "panda_mocap", orientation)

    def set_joint_neutral(self) -> None:
        # assign value to arm joints
        for name, value in zip(self.arm_joint_names, self.neutral_joint_values[0:7]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)

        # assign value to finger joints
        for name, value in zip(self.gripper_joint_names, self.neutral_joint_values[7:9]):
            self._utils.set_joint_qpos(self.model, self.data, name, value)

    def _sample_goal(self) -> np.ndarray:
        goal = np.array([0.0, 0.0, self.initial_object_height])
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        # for the pick and place task
        if not self.block_gripper and self.goal_z_range > 0.0:
            if self.np_random.random() < 0.3:
                noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> None:
        object_position = np.array([0.0, 0.0, self.initial_object_height])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        object_xpos = np.concatenate([object_position, np.array([1, 0, 0, 0])])
        self._utils.set_joint_qpos(self.model, self.data, "obj_joint", object_xpos)

    def get_ee_orientation(self) -> np.ndarray:
        site_mat = self._utils.get_site_xmat(self.model, self.data, "ee_center_site").reshape(9, 1)
        current_quat = np.empty(4)
        self._mujoco.mju_mat2Quat(current_quat, site_mat)
        return current_quat

    def get_ee_position(self) -> np.ndarray:
        return self._utils.get_site_xpos(self.model, self.data, "ee_center_site")

    def get_body_state(self, name) -> np.ndarray:
        body_id = self._model_names.body_name2id[name]
        body_xpos = self.data.xpos[body_id]
        body_xquat = self.data.xquat[body_id]
        body_state = np.concatenate([body_xpos, body_xquat])
        return body_state

    def get_fingers_width(self) -> np.ndarray:
        finger1 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint1")
        finger2 = self._utils.get_joint_qpos(self.model, self.data, "finger_joint2")
        return finger1 + finger2
