
import pickle as pkl

import numpy as np
import torch

from multi_optim.eval import Eval
from roadmaps.var_odrm_torch.var_odrm_torch import make_graph_and_flann
from scenarios.generators import movingai_read_mapfile
from sim.decentralized.iterators import IteratorType

if __name__ == '__main__':
    N = 256
    poses = pkl.load(
        open("learn/reaction_diffusion/rd4mapf-rms-poses.pkl", "rb"))
    map_np = movingai_read_mapfile(
        "roadmaps/odrm/odrm_eval/maps/random-32-32-10.map").astype(np.bool8)
    rep = N // map_np.shape[0]
    mask = np.repeat(map_np, rep, axis=0).repeat(rep, axis=1)
    map_img = tuple((np.logical_not(mask) * 255).astype(np.uint8))
    for name, point_poses in poses.items():
        print(name)
        g, _ = make_graph_and_flann(pos=torch.Tensor(point_poses),
                                    map_img=map_img,
                                    desired_n_nodes=len(point_poses))
        eval = Eval(roadmap=g,
                    map_img=map_img,
                    n_agents_s=[2, 3],
                    n_eval_per_n_agents=2,
                    iterator_type=IteratorType.LOOKAHEAD3,
                    radius=.01,
                    seed=0)
