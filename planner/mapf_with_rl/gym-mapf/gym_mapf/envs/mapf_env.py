import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from planner.mapf_with_rl.mapf_with_rl import Scenario, make_useful_scenarios
from torch_geometric.data import Data


class MapfEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.scenario = None
        self.max_n_nodes = 50  # TODO: fix within reason based on n-hop deep obeservations
        self.max_n_edges = 40  # TODO: "
        self.node_samples = 9
        self.observation_space = spaces.MultiDiscrete(
            # data_x
            # 0: 0, 1: 1, 2: unused
            [3, ] * (self.max_n_nodes * self.node_samples) +
            # data_edge_index
            # (0,0): unused
            [self.max_n_nodes] * (2 * self.max_n_edges)
        )
        self.action_space = spaces.Discrete(2)

    def _to_gym_state(self, state: Data):
        if state is None:
            return [0, ] * self.observation_space.nvec.shape[0]
        x = state.x
        edge_index = state.edge_index
        n_nodes = x.shape[0]
        n_edges = edge_index.shape[0]
        assert self.max_n_nodes > n_nodes
        assert self.max_n_edges > n_edges
        assert x.shape[1] == self.node_samples
        data_x = [2] * (self.max_n_nodes * self.node_samples)
        for i_n in range(n_nodes):
            for i_f in range(self.node_samples):
                i_x = i_n * self.node_samples + i_f
                data_x[i_x] = x[i_n, i_f]
        data_edge_index = [0] * (2 * self.max_n_edges)
        for i_e in range(n_edges):
            i_ei0 = i_e * 2
            data_edge_index[i_ei0] = edge_index[i_e, 0]
            data_edge_index[i_ei0+1] = edge_index[i_e, 1]
        return data_x + data_edge_index

    def step(self, action):
        state, reward = self.scenario.step(action)
        done = state is None
        info = {}
        gs = self._to_gym_state(state)
        return gs, reward, done, info

    def reset(self):
        seed = random.random()
        [self.scenario] = make_useful_scenarios(1, seed, True, 6, 4)
        state = self.scenario.start()
        gs = self._to_gym_state(state)
        return gs

    def render(self, mode='human'):
        for a in self.scenario.agents:
            print(a)

    def close(self):
        del self.scenario
