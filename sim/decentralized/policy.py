import random
from enum import Enum, auto

import numpy as np


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

    def get_priority(self) -> float:
        raise NotImplementedError()


class RandomPolicy(Policy):
    # simply returning a random number on every call
    def __init__(self, agent) -> None:
        super().__init__(agent)

    def get_priority(self) -> float:
        return random.random()


class ClosestPolicy(Policy):
    # a policy that prefers the agent that is currently closest to its
    # goal.
    def __init__(self, agent) -> None:
        super().__init__(agent)

    def get_priority(self) -> float:
        return 1. / np.linalg.norm(self.a.goal - self.a.pos)


class FillPolicy(Policy):
    # an agent will get a higher prio if the map around it is fuller of
    # obstacles.
    def __init__(self, agent) -> None:
        super().__init__(agent)

    def get_priority(self) -> float:
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
        print("init")

    def get_priority(self) -> float:
        return .5
