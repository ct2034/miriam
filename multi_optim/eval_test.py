import unittest
from itertools import product
from random import Random

import networkx as nx
import pytest
import torch
from planner.policylearn.edge_policy import EdgePolicyModel
from scenarios.test_helper import make_cache_folder_and_set_envvar
from sim.decentralized.iterators import IteratorType

from multi_optim.eval import Eval


class EvalTest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:  # type: ignore
        super().__init__(methodName)
        torch.manual_seed(0)
        make_cache_folder_and_set_envvar()
        self.rng = Random(0)

        # square graph
        self.sq_graph = nx.Graph()
        res = 5

        for x, y in product(range(res), range(res)):
            self.sq_graph.add_node(x + res*y, pos=(float(x)/res, float(y)/res))
            if x > 0:
                self.sq_graph.add_edge(x + res*y, x - 1 + res*y)
            if y > 0:
                self.sq_graph.add_edge(x + res*y, x + res*(y - 1))
        self.radius = 0.08

    @pytest.mark.skip(reason="hard to make deterministic")
    def test_eval(self):
        n_agents = 3
        eval = Eval(
            self.sq_graph,
            map_img=((255,),),
            n_agents_s=n_agents,
            n_eval=10,
            iterator_type=IteratorType.LOOKAHEAD2,
            radius=self.radius,
            rng=self.rng)

        model_a = EdgePolicyModel(gpu=torch.device("cpu"))
        model_b = EdgePolicyModel(gpu=torch.device("cpu"))

        regret_a, success_a, accuracy_a = eval.evaluate_policy(model_a)
        regret_b, success_b, accuracy_b = eval.evaluate_policy(model_b)

        # self.assertNotEqual(regret_a, regret_b)
        # self.assertNotEqual(success_a, success_b)
        # self.assertNotEqual(accuracy_a, accuracy_b)


if __name__ == "__main__":
    unittest.main()
