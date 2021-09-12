import logging
import random
from collections import OrderedDict
from enum import Enum, auto
from typing import Optional

import numpy as np
from importtf import keras, tf
from numpy.core.fromnumeric import shape
from planner.policylearn.generate_fovs import (add_padding_to_gridmap,
                                               extract_all_fovs)
from planner.policylearn.train_model import (CLASSIFICATION_STR, CONVRNN_STR,
                                             fix_data_convrnn)
from tensorflow.keras.models import load_model
from tensorflow.python.ops.variables import model_variables

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

logger = logging.getLogger(__name__)


class PolicyType(Enum):
    RANDOM = auto()
    CLOSEST = auto()
    FILL = auto()
    LEARNED = auto()
    INV_LEARNED = auto()
    ONE_THEN_RANDOM = auto()
    ZERO_THEN_RANDOM = auto()


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
        elif type == PolicyType.INV_LEARNED:
            return InverseLearnedPolicy(agent)
        elif type == PolicyType.ONE_THEN_RANDOM:
            return FirstThenRandomPolicy(agent, 1.)
        elif type == PolicyType.ZERO_THEN_RANDOM:
            return FirstThenRandomPolicy(agent, 0.)

    def __init__(self, agent) -> None:
        super().__init__()
        self.a = agent

    def __str__(self):
        return type(self).__name__

    def register_observation(self, id, path, pos, path_i) -> None:
        """we have seen another agent"""
        pass

    def get_priority(self, id: int) -> float:
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
        return 1. / np.linalg.norm(np.array(self.a.goal) - np.array(self.a.pos))


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
        self.paths: OrderedDict = OrderedDict()
        self.poss: OrderedDict = OrderedDict()
        self.path_is: OrderedDict = OrderedDict()
        self.t = 0
        self.model: tf.keras.Model = load_model(
            "planner/policylearn/my_model.h5")
        model_input_shape = list(self.model.layers[0].input_spec[0].shape)
        self.model_type: Optional[str] = None
        if model_input_shape[-3:] == [7, 7, 5]:
            self.model_type = CONVRNN_STR
        else:
            logging.error("can not determine model type")
            raise NotImplementedError("see how this data looks")
        logging.info(f"model_type: {self.model_type}")

    def _path_until_coll(self, path, path_i, n_t):
        path_until_pos = []
        for t in range(n_t):
            i_t = min(max(0, path_i - 1 + t), len(path) - 1)
            path_until_pos.append(path[i_t, :])
        return path_until_pos

    def step(self):
        super().step()
        self.paths = OrderedDict()
        self.poss = OrderedDict()
        self.path_is = OrderedDict()

    def register_observation(self, id, path, pos, path_i) -> None:
        if path is None:
            self.paths[id] = np.array([pos] * self.ts)
            self.path_is[id] = 0
        else:
            path = np.array(path)
            self.paths[id] = path[:, :-1]
            self.path_is[id] = path_i
        self.poss[id] = pos

    def get_priority(self, id_coll: int) -> float:
        """[summary]

        :param id: which agent are we meeting
        :return: priority
        """
        logger.debug(
            f"get_priority, self_agent: {self.a}, other_id: {id_coll}")
        N_T = 3
        i_oa = None
        path = np.array(self.a.path)
        paths_full = [path[:, :-1]]  # self always first
        paths_until_col = [self._path_until_coll(
            path[:, :-1], self.a.path_i, N_T)]  # self always first
        ids = sorted(self.paths.keys())
        if id_coll not in ids:
            logger.warn(f"{id_coll} not in {ids}")
            return .5
        i_oa = ids.index(id_coll) + 1
        for i_id in ids:
            paths_full.append(self.paths[i_id])
            paths_until_col.append(self._path_until_coll(
                self.paths[i_id], self.path_is[i_id], N_T))
        assert i_oa is not None
        x = np.array(
            [extract_all_fovs(
                t=N_T-1,
                paths_until_col=np.array(paths_until_col),
                paths_full=paths_full,
                padded_gridmap=self.padded_gridmap,
                i_a=0,  # self always first
                i_oa=i_oa,
                radius=self.radius
            )]
        )
        if self.model_type == CONVRNN_STR:
            x = fix_data_convrnn(x)
        x_tensor = tf.constant(x)
        y = self.model.predict(x_tensor)[0][0]
        logger.debug(f"y: {y}")
        # from planner.policylearn.generate_data_demo import plot_fovs
        # plot_fovs(x, y)
        return y


class InverseLearnedPolicy(Policy):
    # For demonstration purposes: What if we do the exact opposite?
    def __init__(self, agent) -> None:
        self.po = LearnedPolicy(agent)

    def step(self):
        self.po.step()

    def register_observation(self, id, path, pos, path_i) -> None:
        self.po.register_observation(id, path, pos, path_i)

    def get_priority(self, id_coll: int) -> float:
        prio = 1 - self.po.get_priority(id_coll)
        assert prio >= 0
        assert prio <= 1
        return prio


class FirstThenRandomPolicy(Policy):
    def __init__(self, agent, first_return_value) -> None:
        super().__init__(agent)
        self.first_call = True
        self.first_return_value = first_return_value

    def get_priority(self, _) -> float:
        if self.first_call:
            self.first_call = False
            return self.first_return_value
        return random.random()
