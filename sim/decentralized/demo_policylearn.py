#!/usr/bin/env python3
import random

from matplotlib import pyplot as plt
from scenarios import visualization
from scenarios.generators import tracing_pathes_in_the_dark
from sim.decentralized.agent import Agent
from sim.decentralized.policy import PolicyType
from sim.decentralized.runner import run_a_scenario


def make_scenario():
    """this makes a scenario as in generate_data.py 
    (what the policy was trained on)"""
    width = 8
    n_agents = 8
    fill = .4
    seed = 2035
    gridmap, starts, goals = tracing_pathes_in_the_dark(
        width, fill, n_agents, seed
    )
    return (gridmap, starts, goals)


if __name__ == "__main__":
    (gridmap, starts, goals) = make_scenario()
    visualization.plot_with_arrows(gridmap, starts, goals)
    # plt.show()

    for p in PolicyType:
        agents = []
        for i_a in range(len(starts)):
            a = Agent(gridmap, starts[i_a], p)
            a.give_a_goal(goals[i_a])
            agents.append(a)
        (average_time, max_time, average_length,
         max_length, successful) = run_a_scenario(gridmap, agents, False)
        print(p.name)
        print((average_time, max_time, average_length,
               max_length, successful))
