import numpy as np
from enum import Enum
import random


class Policy(Enum):
    RANDOM = 0
    CLOSEST = 1


class Agent():
    def __init__(pos: np.ndarray, policy: Policy):
        self.pos = pos
        self.goal = None
        self.policy = policy

    def give_a_goal(goal: np.ndarray):
        self.goal = goal

    def get_priority():
        if self.policy == Policy.RANDOM:
            return random.random()
        elif self.policy == Policy.CLOSEST:
            raise NotImplementedError
