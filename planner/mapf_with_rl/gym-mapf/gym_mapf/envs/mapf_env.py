import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from planner.mapf_with_rl.mapf_with_rl import Scenario, make_useful_scenarios


class MapfEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.scenario = None

    def step(self, action):
        state, reward = self.scenario.step(action)
        done = state is None
        info = {}
        return state, reward, done, info

    def reset(self):
        seed = random.random()
        self.scenario = make_useful_scenarios(1, seed)[0]
        state = self.scenario.start()
        return state

    def render(self, mode='human'):
        pass

    def close(self):
        del self.scenario
