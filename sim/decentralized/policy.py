
import random
from enum import Enum, auto

import numpy as np
from importtf import keras
from planner.policylearn.generate_fovs import (add_padding_to_gridmap,
                                               extract_all_fovs)


class PolicyType(Enum):
    RANDOM = auto()
    CLOSEST = auto()
    FILL = auto()
    LEARNED = auto()


class Policy(object):
    @classmethod
    def construct_by_type(cls, type: PolicyType, agent):
        if type == PolicyType.RANDOM:
            return RandomPolicy(agent)
        elif type == PolicyType.CLOSEST:
            return ClosestPolicy(agent)
        elif type == PolicyType.FILL:
            return FillPolicy(agent)
        elif type == PolicyType.LEARNED:
            return LearnedPolicy(agent)

    def __init__(self, agent) -> None:
        super().__init__()
        self.a = agent

    def register_observation(self, id, path, pos) -> None:
        """we have seen another agent"""
        pass

    def get_priority(self, id) -> float:
        raise NotImplementedError()


class RandomPolicy(Policy):
    # simply returning a random number on every call
    def __init__(self, agent) -> None:
        super().__init__(agent)

    def get_priority(self, _) -> float:
        return random.random()


class ClosestPolicy(Policy):
    # a policy that prefers the agent that is currently closest to its
    # goal.
    def __init__(self, agent) -> None:
        super().__init__(agent)

    def get_priority(self, _) -> float:
        return 1. / np.linalg.norm(self.a.goal - self.a.pos)


class FillPolicy(Policy):
    # an agent will get a higher prio if the map around it is fuller of
    # obstacles.
    def __init__(self, agent) -> None:
        super().__init__(agent)

    def get_priority(self, _) -> float:
        FILL_RADIUS = 2
        n_total = ((FILL_RADIUS * 2 + 1)**2)
        n_free = 0
        for x in range(max(0, self.a.pos[0] - FILL_RADIUS),
                       min(self.a.env.shape[0],
                           self.a.pos[0] + FILL_RADIUS + 1)):
            for y in range(max(0, self.a.pos[1] - FILL_RADIUS),
                           min(self.a.env.shape[1],
                               self.a.pos[1] + FILL_RADIUS + 1)):
                if self.a.env[x, y] == 0:
                    n_free += 1
        return float(n_total - n_free) / n_total


class LearnedPolicy(Policy):
    # using machine learning for a greater tomorrow
    def __init__(self, agent) -> None:
        super().__init__(agent)
        self.radius = 3  # how far to look in each direction
        self.ts = 3  # how long to collect data for
        self.padded_gridmap = add_padding_to_gridmap(self.a.env, self.radius)
        self.paths = {}
        self.poss = {}
        self.t = 0
        self.model: keras.Model = keras.models.load_model(
            "planner/policylearn/my_model.h5")

    def _path_until_pos(self, path, until_pos, n_t):
        path_until_pos = []
        for i_t, pos in enumerate(path):
            if tuple(pos) != tuple(until_pos):
                path_until_pos.append(pos)
            else:
                path_until_pos.append(pos)
                while len(path_until_pos) < n_t:
                    path_until_pos.insert(0, pos)
                return path_until_pos[-n_t:]
        assert False, "you should not be here"

    def register_observation(self, id, path, pos) -> None:
        if path is None:
            self.paths[id] = [pos] * self.ts
        else:
            self.paths[id] = path
        self.poss[id] = pos

    def get_priority(self, id) -> float:
        """[summary]

        :param id: which agent are we meeting
        :return: priority
        """
        N_T = 3
        i_oa = None
        paths_full = [self.a.path]  # self always first
        paths_until_col = [self._path_until_pos(
            self.a.path, self.a.pos, N_T)]  # self always first
        for i_a, i_id in enumerate(self.paths.keys()):
            if id == i_id:
                i_oa = i_a
            paths_full.append(self.paths[i_id])
            paths_until_col.append(self._path_until_pos(
                self.paths[i_id], self.poss[i_id], N_T))
        assert i_oa is not None
        x = extract_all_fovs(
            t=N_T-1,
            paths_until_col=np.array(paths_until_col),
            paths_full=paths_full,
            padded_gridmap=self.padded_gridmap,
            i_a=0,  # self always first
            i_oa=i_oa,
            radius=self.radius
        )
        y = self.model.predict(np.array([x]))[0]
        return y
