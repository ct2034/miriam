import abc
from random import Random
from typing import Dict, List, Tuple

from roadmaps.var_odrm_torch.var_odrm_torch import read_map


class RoadmapToTest:
    @abc.abstractmethod
    def __init__(self, map_fname: str, rng: Random):
        pass

    @abc.abstractmethod
    def evaluate(self) -> Dict[str, float]:
        pass


class Spars:
    def __init__(self, map_fname: str, rng: Random):
        from roadmaps.SPARS.build.libsparspy import Spars
        s = Spars()
        s.run("x.png", "", "")
        # self.map_img = map_img
        self.rng = rng

    def evaluate(self) -> Dict[str, float]:
        return {"time": 1, "path": 2}


if __name__ == "__main__":
    rm = Spars("", Random(0))
    rm.evaluate()
