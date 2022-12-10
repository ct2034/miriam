
import pickle as pkl
from pprint import pprint

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt

from definitions import POS
from multi_optim.eval import Eval
from roadmaps.var_odrm_torch.var_odrm_torch import (draw_graph,
                                                    make_graph_and_flann)
from scenarios.generators import movingai_read_mapfile
from sim.decentralized.agent import gridmap_to_graph
from sim.decentralized.iterators import IteratorType

if __name__ == '__main__':
    eval_results_folder: str = "multi_optim/results/eval_reaction_diffusion"

    # load map
    map_name = "random-32-32-10"
    map_np = movingai_read_mapfile(
        f"roadmaps/odrm/odrm_eval/maps/{map_name}.map").astype(np.bool8)
    map_img = tuple((np.logical_not(map_np) * 255).astype(np.uint8))

    # DHC gridmap for comparison
    grid_size = map_np.shape[0]
    g_dhc, coord_to_node = gridmap_to_graph(map_np)
    pos_grid = nx.get_node_attributes(g_dhc, POS)
    nx.set_node_attributes(
        g_dhc,
        {i: (
            (pos_grid[i][0]+.5) / grid_size,
            (pos_grid[i][1]+.5) / grid_size
        ) for i in coord_to_node.values()
            if i in pos_grid},
        POS)
    draw_graph(g_dhc, map_img, title="DHC")
    plt.savefig(f"{eval_results_folder}/{map_name}_dhc.png")
    print('start eval init')
    eval = Eval(roadmap=g_dhc,
                map_img=map_img,
                n_agents_s=[1],
                n_eval_per_n_agents=2,
                iterator_type=IteratorType.LOOKAHEAD3,
                radius=.01,
                seed=0)
    print('done eval init')

    # evaluate
    results = {}

    # load reaction diffusion poses
    width_rd = 256
    poses_rd = pkl.load(
        open("learn/reaction_diffusion/rd4mapf-rms-poses.pkl", "rb"))
    rep = width_rd // map_np.shape[0]
    mask = np.repeat(map_np, rep, axis=0).repeat(rep, axis=1)
    # evaluate reaction diffusion graphs
    for name, point_poses in poses_rd.items():
        print(name)
        g_rd, flann_rd = make_graph_and_flann(
            pos=torch.Tensor(point_poses),
            map_img=map_img,
            desired_n_nodes=len(point_poses))
        draw_graph(g_rd, map_img, title=f"RD {name}")
        plt.savefig(f"{eval_results_folder}/{map_name}_rd_{name}.png")
        results[name] = eval.evaluate_roadmap(g_rd, flann_rd)

    # evaluate multi optim graph
    g_mo = nx.read_gpickle(
        f"multi_optim/results/mapf_benchm_{map_name}_graph.gpickle")
    _, flann_mo = make_graph_and_flann(pos=torch.Tensor(
        [g_mo.nodes[n][POS] for n in g_mo.nodes]),
        map_img=map_img,
        desired_n_nodes=len(g_mo.nodes))
    draw_graph(g_mo, map_img, title=f"MultiOptim")
    plt.savefig(f"{eval_results_folder}/{map_name}_mo.png")
    results["multi_optim"] = eval.evaluate_roadmap(g_mo, flann_mo)

    # print results
    pprint(results)
