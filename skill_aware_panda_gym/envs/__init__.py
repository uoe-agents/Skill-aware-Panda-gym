#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   PandaBase.py
@Author  :   lixin 
@Version :   1.0
@Desc    :   None
'''

from typing import Any, Dict, Optional, Tuple

import numpy as np

from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet
from panda_gym.envs.tasks.pick_and_place import PickAndPlace

class PandaPushWrapper(PickAndPlace):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.3,
        goal_z_range: float = 0.11,
        obj_xy_range: float = 0.3,
        object_height: float= 1.0,
    ) -> None:
        self.object_height = object_height
        super().__init__(sim,reward_type,distance_threshold,goal_xy_range,goal_z_range,obj_xy_range)

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        object_size = np.ones(3) * self.object_size / 2
        object_size[2] = object_size[2] * self.object_height
        object_position = np.array([0.0, 0.0, object_size[2]])
        self.sim.create_box(
            body_name="object",
            half_extents=object_size,
            mass=1.0,
            position=object_position,
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=object_size,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.0, 0.05]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )
        
    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.0, self.object_size / 2 * self.object_height])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, 0.0, self.object_size / 2 * self.object_height])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position
        
class PandaPushEnv(RobotTaskEnv):
    """Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Rool of the camera. Defaults to 0.
    """
    
    def __init__(
        self,
        render_mode: str = "rgb_array",
        reward_type: str = "sparse",
        control_type: str = "ee",
        renderer: str = "Tiny",
        render_width: int = 720,
        render_height: int = 480,
        render_target_position: Optional[np.ndarray] = None,
        render_distance: float = 1.4,
        render_yaw: float = 45,
        render_pitch: float = -30,
        render_roll: float = 0,
        object_height: float = 1,
        friction:float = 1.0,
        mass:float=1.0,
    ) -> None:
        sim = PyBullet(render_mode=render_mode, renderer=renderer)
        robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
        task = PandaPushWrapper(sim, reward_type=reward_type,object_height=object_height,goal_z_range=0.0)
        self.friction = friction
        self.mass = mass
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=render_target_position,
            render_distance=render_distance,
            render_yaw=render_yaw,
            render_pitch=render_pitch,
            render_roll=render_roll,
        )
        self.set_obs_friction_mass(self.friction, self.mass)
        
    def set_obs_friction_mass(self,friction,mass):
        self.friction = friction
        self.mass = mass
        block_uid = self.sim._bodies_idx['object']
        self.sim.physics_client.changeDynamics(bodyUniqueId=block_uid, linkIndex=-1, mass=mass)
        self.sim.set_lateral_friction('table', -1, lateral_friction=friction)

    def get_state(self):
        object = self.sim._bodies_idx['object']
        table = self.sim._bodies_idx['table']
        robot = self.sim._bodies_idx['panda']
        contact_points = self.sim.physics_client.getContactPoints(bodyA=table, bodyB=object, linkIndexA=-1,linkIndexB = -1)
        contact_points1 = self.sim.physics_client.getContactPoints(bodyA = robot,bodyB = object, linkIndexA = 9, linkIndexB = -1)
        contact_points2 = self.sim.physics_client.getContactPoints(bodyA = robot,bodyB = object, linkIndexA = 10, linkIndexB = -1)
        object_info = self.sim.physics_client.getBasePositionAndOrientation(bodyUniqueId=object)

        at_high = object_info[0][2] > 0.021 * 1
        # clamp_finger = fingers_width > 0.03 and fingers_width < 0.0405
        zero_table_contact = len(contact_points) == 0
        contact_with_two_fingers = (len(contact_points1) > 0 and len(contact_points2)) > 0
        
        if at_high and zero_table_contact and contact_with_two_fingers:
            return 'pick'
        elif at_high:
            return 'roll'
        elif object_info[0][2] > 0.019 and object_info[0][2]<0.21:
            return 'push'
        else:
            return 'down'
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        # An episode is terminated iff the agent has reached the target
        terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_goal()))
        truncated = False
        info = {"is_success": terminated}
        reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_goal(), info))
        info['state'] = self.get_state()
        
        return observation, reward, terminated, truncated, info

__all__ = ['PandaPushEnv']