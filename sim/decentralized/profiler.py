#!/usr/bin/env python3
import cProfile
import logging
import pstats
from pstats import SortKey

from sim.decentralized.runner import evaluate_policies

logging.getLogger("sim.decentralized.agent").setLevel(logging.ERROR)
logging.getLogger("sim.decentralized.policy").setLevel(logging.ERROR)
logging.getLogger("__main__").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)
cProfile.run('evaluate_policies(8, 8, 2)', 'restats')

p = pstats.Stats('restats')
p.strip_dirs().sort_stats(1).print_stats()
