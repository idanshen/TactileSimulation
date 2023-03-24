import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

from TactileSimulation.envs.redmax_torch_env import RedMaxTorchEnv
from TactileSimulation.utils.common import *
from scipy.spatial.transform import Rotation
from TactileSimulation.utils import math
from gym import spaces
import torch
import cv2
import time


class DClawReorientationEnv(RedMaxTorchEnv):
    def __init__(self, use_torch=False, observation_type="no_tactile",
                 render_tactile=False, verbose=False, seed=0,
                 torque_control=False, relative_control=True):
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        self.is_torque_control = torque_control
        self.relative_control = relative_control
        self.relative_q_scale = 0.06
        self.rot_coef = 1.0
        self.power_coef = 0.005
        self.has_been_reset = False
        if torque_control:
            model_path = os.path.join(asset_folder, 'dclaw_reorientation/dclaw_torque_control.xml')
        else:
            model_path = os.path.join(asset_folder, 'dclaw_reorientation/dclaw_position_control.xml')

        self.observation_type = observation_type
        self.use_torch = use_torch
        self.verbose = verbose
        self.render_tactile = render_tactile

        self.tactile_rows = 20
        self.tactile_cols = 20

        if self.observation_type == "tactile_flatten":
            raise NotImplementedError
            # self.tactile_obs_shape = (self.tactile_rows * self.tactile_cols * 3 * 3,)
        elif self.observation_type == "tactile":
            raise NotImplementedError
            # self.tactile_obs_shape = (3 * 3, self.tactile_rows, self.tactile_cols)
        elif self.observation_type == "no_tactile":
            self.tactile_obs_shape = (42,)
        else:
            raise NotImplementedError

        self.goal_quat = np.array([0., 0., np.sqrt(2.), np.sqrt(2.)])
        self.random_initialization = True
        self.rotate_all_axis = True

        self.tactile_force_buf = torch.zeros(self.tactile_obs_shape)
        self.tactile_obs_buf = torch.zeros(self.tactile_obs_shape)
        self.reward_buf = 0.
        self.done_buf = False
        self.tactile_force_his = []
        self.tactile_obs_his = []
        self.info_buf = {}

        self.ndof_u = 9
        self.actions = np.zeros(self.ndof_u)

        super(DClawReorientationEnv, self).__init__(model_path, seed=seed)

        self.frame_skip = 20
        self.dt = self.sim.options.h * self.frame_skip
        print(f'Dt:{self.dt}')  # Currently 10 Hz, Should be 12 Hz control
        self.dof_vel_pol_limit = 3.5

        self.action_space = spaces.Box(low=np.full(self.ndof_u, -1.), high=np.full(self.ndof_u, 1.), dtype=np.float32)

        self.sim.viewer_options.camera_lookat = np.array([0., 0., 1])
        self.sim.viewer_options.camera_pos = (np.array(
            [4., -5., 4]) - self.sim.viewer_options.camera_lookat) / 1.8 + self.sim.viewer_options.camera_lookat

        image_poses = self.sim.get_tactile_image_pos('one3_link_fingertip')
        self.tactile_mask = np.full((20, 20), 0, dtype=bool)
        for i in range(len(image_poses)):
            self.tactile_mask[image_poses[i][0]][image_poses[i][1]] = True

        self.q_init = self.sim.get_q_init().copy()
        print(f'Q_init:{self.q_init}')
        self.q_init[1], self.q_init[4], self.q_init[7] = -0.8, -0.8, -0.8
        self.q_init[2], self.q_init[5], self.q_init[8] = 1.1, 1.1, 1.1
        self.dof_limit = np.array([
            [-0.45, 1.35],
            [-2, 2],
            [-2, 2],
            [-0.45, 1.35],
            [-2, 2],
            [-2, 2],
            [-0.45, 1.35],
            [-2, 2],
            [-2, 2],
        ])
        self.sim.set_q_init(self.q_init)
        # self.cap_top_surface_z = 0.05
        # self.cap_center = np.array([0, 0, 0.035])

    def _get_obs(self):
        q, qdot = self.sim.get_q().copy(), self.sim.get_qdot().copy()
        hand_dofs = q[:9].copy()  # joint angles of the hand
        # scaled_dof_pos = math.unscale(q[:9], self.joint_limit_lower, self.joint_limit_upper) # TODO: Necessary?

        variables = self.sim.get_variables().copy()  # the variables contains the positions of three finger tips [0:3], [3:6], [6:9]
        fingertip_pos_world = variables[3:12]
        cube_pos_world = variables[:3]
        cube_rotvec = q[-3:].copy()
        cube_quat = np.array(Rotation.from_rotvec(cube_rotvec).as_quat())
        cube_pose = np.concatenate([cube_pos_world, cube_quat])

        quat_diff = math.quat_mul(cube_quat, math.quat_conjugate(self.goal_quat))

        if self.observation_type == "no_tactile":
            # state = np.concatenate((hand_dofs, fingertip_pos_world))
            state = np.concatenate([hand_dofs, cube_pose, self.goal_quat, quat_diff, fingertip_pos_world, self.actions])
        else:
            raise NotImplementedError
            # tactiles = np.array(self.sim.get_tactile_flow_images())
            #
            # self.tactile_force_buf = torch.tensor(tactiles)  # (3, 20, 20, 3)
            #
            # self.tactile_obs_buf = self.tactile_force_buf.clone()
            #
            # if self.observation_type == "tactile_flatten":
            #     self.tactile_obs_buf = self.tactile_obs_buf.reshape(-1)
            # elif self.observation_type == "tactile":
            #     self.tactile_obs_buf = self.tactile_obs_buf \
            #         .permute(0, 3, 1, 2) \
            #         .reshape(-1, self.tactile_rows, self.tactile_cols)
            #
            # obs = self.tactile_obs_buf
            #
            # state = np.concatenate((q[:9], fingertip_pos_world, obs.reshape(-1)))

        return state

    def _get_reward(self, action):
        return 0, False, False
        # reward, dones, resets, goal_resets, progress_buf, successes, \
        # cons_successes, abs_rot_dist, reward_terms, time_due_envs = self.compute_reward(
        #     reset_buf, reset_goal_buf, progress_buf,
        #     successes, consecutive_successes, max_episode_length,
        #     object_pos, object_rot, target_pos, target_rot,
        #     actions, max_consecutive_successes,
        #     av_factor, ignore_z_rot,
        #     fingertip_pos,
        #     object_linvel, object_angvel,
        #     raw_actions_from_policy, dof_vel, quat_diff,
        #     dof_torque, add_dof_vel_penalty,
        #     relative_control, dist_reward_scale,
        #     rot_reward_scale, rot_eps,
        #     reach_goal_bonus, fall_dist,
        #     fall_penalty, success_tolerance,
        #     no_pos_reward, time_due_penalty,
        #     ftip_reward_scale, ftipRewardMaxOnly,
        #     timeout_not_done, energy_scale,
        #     max_dof_vel, dof_vel_rew_scale, dof_vel_thresh,
        #     obj_lin_vel_thresh, obj_ang_vel_thresh, action_norm_thresh,
        #     penalize_tb_contact, table_cf, tb_cf_scale,
        #     clip_energy_reward, energy_upper_bound,
        #     tip_height_penalty, tip_high_reset, )

        # q, qdot = self.sim.get_q().copy(), self.sim.get_qdot().copy()
        # hand_dofs = q[:9].copy()  # joint angles of the hand
        #
        # variables = self.sim.get_variables().copy()  # the variables contains the positions of three finger tips [0:3], [3:6], [6:9]
        # fingertip_pos_world = np.array(variables[:9])
        #
        # cap_angle = q[-1]
        #
        # tactile_force = self.tactile_force_buf.cpu().numpy()
        # tactile_force_grid = np.linalg.norm(tactile_force, axis=-1)
        # finger_tactile_force = tactile_force_grid.sum(-1).sum(-1)
        # not_in_contact = finger_tactile_force < 1.0
        #
        # reward = 0.
        #
        # reward -= not_in_contact.sum() * 0.5
        #
        # max_rotation_angle = np.pi / 4
        # rotation_reward = -self.rot_coef * (min(cap_angle - max_rotation_angle, 0)) ** 2
        # reward += rotation_reward
        #
        # action_penalty = -self.power_coef * np.sum(action ** 2)
        # reward += action_penalty
        #
        # done = False
        # success = False
        #
        # if np.any(fingertip_pos_world[2::3] > self.cap_top_surface_z):
        #     done = True
        #     reward += -50
        #     if self.verbose:
        #         print(f'fingertip pos:{fingertip_pos_world[2::3]}')
        #
        # if cap_angle > max_rotation_angle:
        #     reward += 50
        #     success = True
        #     done = True
        # if self.verbose:
        #     print(f'cap angle:{cap_angle}, target:{max_rotation_angle}   done:{done}')
        # return reward, done, success

    def compute_reward(self, reset_goal_buf,
                       progress_buf,
                       successes,
                       max_episode_length: float,
                       object_pos,
                       target_pos,
                       actions,
                       max_consecutive_successes: int,
                       fingertip_pos,
                       object_linvel,
                       object_angvel,
                       dof_vel,
                       quat_diff,
                       dof_torque,
                       relative_control: bool,
                       rot_reward_scale: float = 1.0,
                       rot_eps: float = 0.1,
                       reach_goal_bonus: float = 800,
                       fall_dist: float = 0.24,
                       fall_penalty: float = -100,
                       success_tolerance: float = 0.4,
                       time_due_penalty: bool = True,
                       ftip_reward_scale: float = -1.0,
                       ftipRewardMaxOnly: bool = False,
                       timeout_not_done: bool = False,
                       energy_scale: float = 20,
                       dof_vel_thresh: float = 0.25,
                       obj_lin_vel_thresh: float = 0.04,
                       obj_ang_vel_thresh: float = 0.5,
                       action_norm_thresh: float = 1.0,
                       penalize_tb_contact: bool = False,
                       table_cf = 1.0,
                       tb_cf_scale: float = 0.5,
                       clip_energy_reward: bool = True,
                       energy_upper_bound: float = 10,
                       ):
        # Distance from the hand to the object
        goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
        num_envs = object_pos.shape[0]

        reward_terms = dict()
        if ftip_reward_scale is not None and ftip_reward_scale < 0:
            ftip_diff = (fingertip_pos.view(num_envs, -1, 3) - object_pos[:, None, :])
            ftip_dist = torch.linalg.norm(ftip_diff, dim=-1).view(num_envs, -1)
            if ftipRewardMaxOnly:
                ftip_dist_max = ftip_dist.max(dim=-1)[0]
                ftip_reward = ftip_dist_max * ftip_reward_scale
            else:
                ftip_dist_mean = ftip_dist.mean(dim=-1)
                ftip_reward = ftip_dist_mean * ftip_reward_scale
            reward_terms['ftip_reward'] = ftip_reward

        object_linvel_norm = torch.linalg.norm(object_linvel, dim=-1)
        object_angvel_norm = torch.linalg.norm(object_angvel, dim=-1)

        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
        abs_rot_dist = torch.abs(rot_dist)

        rot_rew = 1.0 / (abs_rot_dist + rot_eps) * rot_reward_scale
        reward_terms['rot_reward'] = rot_rew
        action_norm = torch.linalg.norm(actions, dim=-1)
        energy_cost = torch.abs(dof_vel * dof_torque).sum(dim=-1)
        if clip_energy_reward:
            energy_cost = torch.clamp(energy_cost, max=energy_upper_bound)
        reward_terms['energy_reward'] = -energy_cost * energy_scale

        if penalize_tb_contact:
            in_contact = torch.abs(table_cf).sum(-1) > 0.2
            reward_terms['tb_contact_reward'] = -in_contact.float() * tb_cf_scale

        dof_vel_norm = torch.linalg.norm(dof_vel, dim=-1)

        goal_reach = (abs_rot_dist <= success_tolerance) & (dof_vel_norm <= dof_vel_thresh) \
                     & (object_linvel_norm <= obj_lin_vel_thresh) & (object_angvel_norm <= obj_ang_vel_thresh)
        if penalize_tb_contact:
            goal_reach = goal_reach & (torch.abs(table_cf).sum(-1) < 0.2)

        if relative_control:
            goal_reach = goal_reach & (action_norm <= action_norm_thresh)
        goal_resets = torch.where(goal_reach, torch.ones_like(reset_goal_buf), reset_goal_buf)

        fall_envs = goal_dist >= fall_dist
        dones = torch.logical_or(goal_reach, fall_envs)
        successes = successes + goal_resets

        reward = torch.sum(torch.stack(list(reward_terms.values())), dim=0)

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        reward = torch.where(goal_reach, reward + reach_goal_bonus, reward)

        # Fall penalty: distance to the goal is larger than a threashold
        reward = torch.where(fall_envs, reward + fall_penalty, reward)

        truncated = progress_buf >= max_episode_length - 1
        if not timeout_not_done:
            dones = torch.logical_or(dones, truncated)
        # Time out penalty: distance to the goal is larger than a threashold
        if max_consecutive_successes > 0 and time_due_penalty:
            reward = torch.where(truncated, reward + 0.5 * fall_penalty, reward)

        return reward, dones.int(), truncated, successes

    def reset(self):
        if self.random_initialization:  # random initial cube initial orientation
            q_init = self.sim.get_q_init().copy()
            angle_axis = self.generate_random_rotation().as_rotvec()
            cube_init_pos = np.array([0., 0., 0.02])
            q_init[-6:-3] = cube_init_pos
            q_init[-3:] = angle_axis
            self.sim.set_q_init(q_init)

        if not self.rotate_all_axis:  # random y axis
            self.goal_q = self.np_random.uniform(low=-np.pi, high=np.pi)
            self.goal_quat = Rotation.from_rotvec(self.goal_q * np.array([0., 1., 0.])).as_quat()
        else:  # random rotation around all axis
            self.goal_quat = self.generate_random_rotation().as_quat()

        # goal_pos = np.array([0.15, 0.05, 0.08])
        goal_pos = np.array([0.0, 0.0, 0.08])
        goal_sim_quat = np.concatenate([self.goal_quat[3:], self.goal_quat[0:3]])
        if self.verbose:
            self.sim.update_virtual_object("goal", np.concatenate([goal_pos, goal_sim_quat]))

        self.sim.reset(backward_flag=False)
        self.energy_usage = 0
        self.has_been_reset = True

        self.tactile_force_his = []
        self.tactile_obs_his = []

        return self._get_obs()

    def step(self, u):
        if self.use_torch:
            u = u.detach().cpu().numpy()

        action = np.clip(u, -1., 1.)

        self.actions = action
        policy_out = u.copy()

        if not self.is_torque_control:
            if self.relative_control:
                cur_q = self.sim.get_q().copy()[:9]
                action = cur_q + action * self.dof_vel_pol_limit * self.dt
                action = np.clip(action, self.dof_limit[:, 0], self.dof_limit[:, 1])
            else:
                action = scale(action, self.dof_limit[:, 0], self.dof_limit[:, 1])
        self.sim.set_u(action)

        self.sim.forward(self.frame_skip, verbose=False, test_derivatives=False)

        reward, done, success = self._get_reward(policy_out)

        # append tactile his
        if self.render_tactile:
            raise NotImplementedError
            # self.tactile_force_his.append(self.get_tactile_forces_array())
            # self.tactile_obs_his.append(self.get_tactile_obs_array())

        info = {'success': success}
        if done:
            info['reward_energy'] = self.energy_usage

        return self._get_obs(), reward, done, False, info

    def visualize_tactile(self, tactile_array):
        resolution = 40
        horizontal_space = 10
        vertical_space = 20
        T = len(tactile_array)
        N = tactile_array.shape[1]
        nrows = tactile_array.shape[2]
        ncols = tactile_array.shape[3]

        imgs_tactile = np.zeros(
            (ncols * resolution * N + vertical_space * (N + 1), nrows * resolution * T + horizontal_space * (T + 1), 3),
            dtype=float)

        for timestep in range(T):
            for finger_idx in range(N):
                for row in range(nrows):
                    for col in range(ncols):
                        if self.tactile_mask[row][col]:
                            loc0_x = row * resolution + resolution // 2 + timestep * nrows * resolution + timestep * horizontal_space + horizontal_space
                            loc0_y = col * resolution + resolution // 2 + finger_idx * ncols * resolution + finger_idx * vertical_space + vertical_space
                            loc1_x = loc0_x + tactile_array[timestep][finger_idx][row, col][0] / 10. * resolution
                            loc1_y = loc0_y + tactile_array[timestep][finger_idx][row, col][1] / 10. * resolution
                            depth_ratio = min(1., np.abs(tactile_array[timestep][finger_idx][row, col][2]) / 0.2)
                            color = (0.0, 1.0 - depth_ratio, depth_ratio)
                            cv2.arrowedLine(imgs_tactile, (int(loc0_x), int(loc0_y)), (int(loc1_x), int(loc1_y)), color,
                                            12, tipLength=0.4)

        return imgs_tactile

    def render(self, mode='once'):
        if self.render_tactile and self.observation_type != "no_tactile":
            for t in range(len(self.tactile_force_his)):
                img_tactile_1 = self.visualize_tactile(self.tactile_force_his[t][:, 0:1, ...])
                img_tactile_2 = self.visualize_tactile(self.tactile_force_his[t][:, 1:2, ...])
                img_tactile_3 = self.visualize_tactile(self.tactile_force_his[t][:, 2:3, ...])
                img_tactile_1 = img_tactile_1.transpose([1, 0, 2])
                img_tactile_2 = img_tactile_2.transpose([1, 0, 2])
                img_tactile_3 = img_tactile_3.transpose([1, 0, 2])
                cv2.imshow("tactile_1", img_tactile_1)
                cv2.imshow("tactile_2", img_tactile_2)
                cv2.imshow("tactile_3", img_tactile_3)

                cv2.waitKey(1)

                time.sleep(0.05)

        super().render(mode)

    def get_tactile_forces_array(self):
        tactile_force_array = self.tactile_force_buf.clone().detach().cpu().numpy()
        tactile_force_array[..., 0:2] = tactile_force_array[..., 0:2] / 0.03
        tactile_force_array[..., 2:3] = tactile_force_array[..., 2:3] / 3.
        return np.expand_dims(tactile_force_array, axis=0)

    # return tactile obs array: shape (T, 2, nrows, ncols, 2)
    def get_tactile_obs_array(self):
        if self.observation_type == 'tactile_flatten' or self.observation_type == 'privilege' or self.observation_type == "tactile_flatten_his" or self.observation_type == "no_tactile":
            tactile_obs = self.tactile_obs_buf.reshape(1, 3, self.tactile_rows, self.tactile_cols, 3)
        elif self.observation_type == 'tactile':
            tactile_obs = self.tactile_obs_buf.reshape(1, 3, 3, self.tactile_rows, self.tactile_cols) \
                .permute(0, 1, 3, 4, 2)

        return tactile_obs.detach().cpu().numpy()

    def generate_random_rotation(self):
        '''
        follow the formula in http://planning.cs.uiuc.edu/node198.html
        '''
        ran = np.random.rand(3)
        r1, r2, r3 = ran[0], ran[1], ran[2]
        pi2 = 2 * np.pi
        r1_1 = np.sqrt(1.0 - r1)
        r1_2 = np.sqrt(r1)
        t1 = pi2 * r2
        t2 = pi2 * r3

        quat = np.zeros(4)
        quat[3] = r1_1 * (np.sin(t1)) # w
        quat[0] = r1_1 * (np.cos(t1)) # x
        quat[1] = r1_2 * (np.sin(t2)) # y
        quat[2] = r1_2 * (np.cos(t2)) # z

        return Rotation.from_quat(quat)

def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)
