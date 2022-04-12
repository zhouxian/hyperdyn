import numpy as np
from modules.utils.serializable import Serializable
from modules.envs.mujoco_env import MujocoEnv
from modules.logger import logger
import os


class HalfCheetahHFieldEnv(MujocoEnv, Serializable):
    def __init__(self, reset_every_episode=False, reward=True, *args, **kwargs):
        Serializable.quick_init(self, locals())

        self.cripple_mask = None
        self.reset_every_episode = reset_every_episode
        self.first = True
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                              "assets", "half_cheetah_hfield.xml"))

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.dt = self.model.opt.timestep

        self.x_walls = np.array([250, 260, 261, 270, 280, 285])
        self.height_walls = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.height = 0.8
        self.width = 15

    def get_env_info(self):
        info = {'info': np.array([self.height])}
        return info

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        done = False
        info = self.get_env_info()
        return next_obs, reward, done, info

    def reward(self, obs, action, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action), axis=1)
        forward_reward = (next_obs[:, -3] - obs[:, -3])/self.dt
        reward = forward_reward - ctrl_cost
        return reward

    def reset_mujoco(self, init_state=None):
        super(HalfCheetahHFieldEnv, self).reset_mujoco(init_state=init_state)
        if self.reset_every_episode and not self.first:
            self.reset_task()

        if self.first:
            self.first = False

    def reset_task(self, value=None):
        self.height_walls = np.array([1, 1, 1, 1, 1, 1])
        self.height = np.random.uniform(0, 4)
        height = self.height
        width = self.width
        self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
        self.model.hfield_size = np.array([50, 5, height, 0.1])
        row = np.zeros((500,))
        for i, x in enumerate(self.x_walls):
            terrain = np.cumsum([self.height_walls[i]] * width)
            row[x:x + width] += terrain
            row[x + width:] = row[x + width - 1]
        row = (row - np.min(row)) / (np.max(row) - np.min(row))
        hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
        self.model.hfield_data = hfield
        self.model.forward()

    def log_diagnostics(self, paths, prefix):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
            ]
        logger.logkv(prefix + 'AverageForwardProgress', np.mean(progs))
        logger.logkv(prefix + 'MaxForwardProgress', np.max(progs))
        logger.logkv(prefix + 'MinForwardProgress', np.min(progs))
        logger.logkv(prefix + 'StdForwardProgress', np.std(progs))