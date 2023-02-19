from itertools import product
import logging
from math import ceil, sqrt
import os
import re
import timeit
from random import Random
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bresenham import bresenham
from matplotlib import pyplot as plt
from pyflann import FLANN
from tqdm import tqdm

from definitions import DISTANCE, POS
from planner.astar_boost.converter import initialize_from_graph
from roadmaps.var_odrm_torch.var_odrm_torch import (
    get_path_len, is_coord_free, is_pixel_free, make_paths,
    plan_path_between_coordinates, read_map, PATH_W_COORDS)

from planner.astar_boost.build.libastar_graph import AstarSolver


CSV_PATH = "roadmaps/benchmark.csv"
PLOT_FOLDER = "roadmaps/benchmark_plots"
DPI = 500

logger = logging.getLogger(__name__)

# this list is roughly sorted by complexity
MAP_NAMES = [
    'plain',
    'c',
    # 'x',
    'b',
    # 'o',
    # 'dual_w',
    # 'dual2',
    'dual',
    'z',
    'dense',
    # 'simple'
]
N_SEEDS = 10


class RoadmapToTest:
    def __init__(self, map_fname: str, rng: Random,
                 roadmap_specific_kwargs: Dict[str, Any] = {}):
        self.map_fname = map_fname
        self.map_img = read_map(map_fname)
        # swap rows and columns
        # self.map_img = np.swapaxes(np.array(self.map_img), 0, 1)
        self.n_eval = 100
        self.rng = rng
        self.roadmap_specific_kwargs = roadmap_specific_kwargs
        self.g: Optional[nx.Graph] = None
        self.runtime_ms: Optional[float] = None
        self.astar_solver: Optional[AstarSolver] = None
        self.flann: Optional[FLANN] = None

    def _initialize_eval_rng(self):
        self.rng = Random(0)

    def _plan_path_indices(self, a: int, b: int):
        assert self.astar_solver is not None, 'Roadmap must be built.'
        if isinstance(a, np.int32):
            a = a.item()
        if isinstance(b, np.int32):
            b = b.item()
        return self.astar_solver.plan_w_number_nodes_visited(a, b)

    def _plan_path_coords(
            self,
            a: Tuple[float, float],
            b: Tuple[float, float]) -> Tuple[PATH_W_COORDS, int]:
        assert self.g is not None, 'Roadmap must be built.'
        pos_np = np.zeros((max(self.g.nodes) + 1, 2))
        for n, (x, y) in nx.get_node_attributes(self.g, POS).items():
            pos_np[n] = torch.tensor([x, y])
        assert self.flann is not None, 'Roadmap must be built.'
        nn = 1
        result, _ = self.flann.nn_index(
            np.array([a, b], dtype='float32'), nn,
            random_seed=0)
        start_v, goal_v = result
        path, visited_nodes = self._plan_path_indices(start_v, goal_v)
        return (a, b, path), visited_nodes

    def _make_graph_indices_strictly_increasing(self):
        assert self.g is not None, 'Roadmap must be built.'
        n_nodes = self.g.number_of_nodes()
        max_node = max(self.g.nodes)
        if max_node == n_nodes - 1:
            return
        mapping = {n: i for i, n in enumerate(self.g.nodes)}
        self.g = nx.relabel_nodes(self.g, mapping)

    def _set_graph(self, g: nx.Graph):
        self.g = g
        if self.g.number_of_nodes() == 0:
            return
        self._make_graph_indices_strictly_increasing()
        self.astar_solver = initialize_from_graph(self.g)
        self.flann = FLANN(random_seed=0)
        self.pos_np = np.zeros((len(self.g.nodes), 2))
        for n, (x, y) in nx.get_node_attributes(self.g, POS).items():
            self.pos_np[n, :] = [x, y]
        self.flann.build_index(
            np.array(self.pos_np, dtype=np.float32), random_seed=0)

    def evaluate_path_length(self) -> Dict[str, float]:
        assert self.g is not None, 'Roadmap must be built.'
        if self.g.number_of_nodes() == 0:
            logger.error('Error while making paths, no nodes in graph.')
            return {'success_rate': 0.0}
        map_img_inv = np.swapaxes(np.array(self.map_img), 0, 1)
        lens = []
        visited_nodes_s = []
        for _ in range(self.n_eval):
            a = self.rng.random(), self.rng.random()
            b = self.rng.random(), self.rng.random()
            if not is_coord_free(map_img_inv, a) or \
                    not is_coord_free(map_img_inv, b):
                continue
            path_w_coords, visited_nodes = self._plan_path_coords(a, b)
            visited_nodes_s.append(visited_nodes)
            if path_w_coords[-1] is None:
                continue
            if len(path_w_coords[-1]) == 0:
                continue
            pos_t = torch.tensor(self.pos_np)
            lens.append(
                get_path_len(pos_t, path_w_coords, False).item()
            )
        data = {'success_rate': len(lens) / self.n_eval}
        for i, d in enumerate(lens):
            data[f'path_length_{i:02d}'] = d
        data['path_length_mean'] = np.mean(lens).item()
        data['visited_nodes_mean'] = np.mean(visited_nodes_s).item()
        return data

    def _check_line(self, a: Tuple[float, float], b: Tuple[float, float]):
        line = bresenham(
            round(a[0] * len(self.map_img)),
            round(a[1] * len(self.map_img)),
            round(b[0] * len(self.map_img)),
            round(b[1] * len(self.map_img)),
        )
        # print(list(line))
        return all([is_pixel_free(
            self.map_img, (x[0], x[1])) for x in line])

    def evaluate_straight_path_length(self) -> Dict[str, float]:
        assert self.g is not None, 'Roadmap must be built.'
        if self.g.number_of_nodes() == 0:
            logger.error('Error while making paths, no nodes in graph.')
            return {'success_rate': 0.0}
        pos_np = np.zeros((max(self.g.nodes) + 1, 2))
        for n, (x, y) in nx.get_node_attributes(self.g, POS).items():
            pos_np[n] = torch.tensor([x, y])
        pos_t = torch.tensor(pos_np)
        flann = FLANN(random_seed=0)
        flann.build_index(np.array(pos_np, dtype=np.float32), random_seed=0)
        rel_straight_lengths: List[float] = []
        for _ in range(self.n_eval):
            found = False
            while not found:
                start = (self.rng.random(), self.rng.random())
                if not is_coord_free(self.map_img, start):
                    continue
                end = (self.rng.random(), self.rng.random())
                if not is_coord_free(self.map_img, end):
                    continue
                if not self._check_line(start, end):
                    continue
                length = np.linalg.norm(np.array(start) - np.array(end)).item()
                if length < 0.5:
                    continue
                try:
                    path = plan_path_between_coordinates(
                        self.g, flann, start, end)
                except Exception:
                    logger.error("Error while planning path.: ", exc_info=True)
                    continue
                if path is None:
                    continue
                found = True
            assert path is not None
            path_len = get_path_len(pos_t, path, False).item()
            rel_straight_lengths.append(path_len / length)
        data = {}
        for i, d in enumerate(rel_straight_lengths):
            data[f"rel_straight_length_{i:02d}"] = d
        data["rel_straight_length_mean"] = np.mean(rel_straight_lengths).item()
        return data

    def evaluate_n_nodes(self) -> Dict[str, float]:
        assert self.g is not None, "Roadmap must be built."
        return {"n_nodes": self.g.number_of_nodes()}

    def evaluate_runtime(self) -> Dict[str, float]:
        if self.runtime_ms is None:
            return {}
        return {"runtime_ms": self.runtime_ms}

    def evaluate(self):
        results = {}
        for fun in [
            self.evaluate_path_length,
            self.evaluate_straight_path_length,
            self.evaluate_n_nodes,
            self.evaluate_runtime,
        ]:
            self._initialize_eval_rng()
            results.update(fun())
        return results

    def plot_example(self, folder: str, i: int):
        assert self.g is not None, 'Roadmap must be built.'
        fig, ax = plt.subplots(dpi=DPI)
        ax.imshow(self.map_img, cmap='gray')

        # plot graph
        pos = nx.get_node_attributes(self.g, POS)
        pos = {k: (v[0] * len(self.map_img), v[1] * len(self.map_img))
               for k, v in pos.items()}
        nx.draw_networkx_nodes(self.g, pos, ax=ax, node_size=1)
        # exclude self edges
        edges = [(u, v) for u, v in self.g.edges if u != v]
        nx.draw_networkx_edges(self.g, pos, ax=ax, edgelist=edges, width=0.5)

        # what will be the file name
        n_nodes = self.g.number_of_nodes()
        if n_nodes == 0:
            return
        name = f"{i:03d}_{self.__class__.__name__}_" +\
            f"{n_nodes:04d}n_" +\
            f"{os.path.basename(self.map_fname)}"

        # make an example path
        # invert x and y of map
        map_img = np.swapaxes(np.array(self.map_img), 0, 1)
        map_img_t = tuple(map_img.tolist())
        seed = 0
        path = None
        while path is None:
            try:
                path = make_paths(self.g, 1, map_img_t, Random(seed))[0]
            except (ValueError, IndexError):
                seed += 1
        start, end, node_path = path
        coord_path = [pos[n] for n in node_path]
        full_path = (
            [(start[0] * len(self.map_img),
              start[1] * len(self.map_img))] +
            coord_path +
            [(end[0] * len(self.map_img),
              end[1] * len(self.map_img))])
        x, y = zip(*full_path)
        ax.plot(x, y, color="red", linewidth=0.8, alpha=0.7)

        fig.savefig(os.path.join(folder, name))
        plt.close(fig)


class GSRM(RoadmapToTest):
    """Its gsorm without the o.
    So no optimization, only the generation of the points by the
    Grey Scott model."""

    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)
        kwargs = roadmap_specific_kwargs.copy()
        if 'plot' not in kwargs:
            kwargs['plot'] = False
        if 'target_n' not in kwargs:
            print(kwargs)
        target_n = kwargs.pop('target_n')
        actual_n = 200
        resolution = 300
        from roadmaps.gsorm.build.libgsorm import Gsorm
        from roadmaps.var_odrm_torch.var_odrm_torch import make_graph_and_flann

        while target_n > actual_n:
            print(f"Trying resolution {resolution}...")
            kwargs['resolution'] = resolution
            gs = Gsorm()
            nodes, runtime_points = gs.run(
                mapFile=map_fname,
                **kwargs,
            )
            pos = torch.Tensor(nodes) / resolution
            actual_n = pos.shape[0]
            print(f"Got {actual_n} nodes. Target was {target_n}.")
            factor = 1 + (target_n / actual_n - 1) * 0.7
            resolution *= sqrt(factor)
            resolution = int(resolution)

        # swap x and y
        pos = pos[:, [1, 0]]
        n = pos.shape[0]
        start_t = timeit.default_timer()
        g, _ = make_graph_and_flann(pos, self.map_img, n, rng)
        runtime_delaunay = (timeit.default_timer() - start_t) * 1000
        self.runtime_ms = runtime_points + runtime_delaunay

        # swap x and y
        nx.set_node_attributes(g,
                               {i: (p[1],
                                    p[0]) for i, p in nx.get_node_attributes(
                                   g, POS).items()}, POS)
        self._set_graph(g)


# class GSORM(RoadmapToTest):
#     def __init__(self,
#                  map_fname: str,
#                  rng: Random,
#                  roadmap_specific_kwargs):
#         super().__init__(map_fname, rng, roadmap_specific_kwargs)
#         from roadmaps.gsorm.build.libgsorm import Gsorm
#         from roadmaps.var_odrm_torch.var_odrm_torch import (
#             make_graph_and_flann, optimize_poses)

#         # prepare args
#         gsorm_kwargs = self.roadmap_specific_kwargs.copy()
#         epochs_optim = gsorm_kwargs.pop("epochs_optim", None)
#         lr_optim = gsorm_kwargs.pop("lr_optim", None)

#         # run gsorm
#         gs = Gsorm()
#         nodes, runtime_points = gs.run(
#             mapFile=map_fname,
#             **gsorm_kwargs,
#             plot=False,
#         )
#         pos = torch.Tensor(nodes) / roadmap_specific_kwargs["resolution"]
#         # swap x and y
#         pos = pos[:, [1, 0]]
#         pos.requires_grad = True
#         n = pos.shape[0]

#         # optimize
#         optimizer = torch.optim.Adam([pos], lr=lr_optim)
#         start_t = timeit.default_timer()
#         g, _ = make_graph_and_flann(pos, self.map_img, n, rng)
#         for i_e in tqdm(range(epochs_optim)):
#             g, pos, test_length, training_length = optimize_poses(
#                 g, pos, self.map_img, optimizer, n, rng)
#         runtime_optim = (timeit.default_timer() - start_t) * 1000
#         self.runtime_ms = runtime_points + runtime_optim

#         # swap x and y
#         nx.set_node_attributes(g,
#                                {i: (p[1],
#                                     p[0]) for i, p in nx.get_node_attributes(
#                                    g, POS).items()}, POS)
#         self._set_graph(g)


class SPARS(RoadmapToTest):
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)

        from roadmaps.SPARS.build.libsparspy import Spars
        s = Spars()
        edges, self.runtime_ms = s.run(
            mapFile=map_fname,
            seed=rng.randint(0, 2**16),
            **self.roadmap_specific_kwargs,
        )

        # to networkx
        g = nx.Graph()
        for (a, ax, ay, b, bx, by) in edges:
            # positions to unit square
            ax /= len(self.map_img)
            ay /= len(self.map_img)
            bx /= len(self.map_img)
            by /= len(self.map_img)
            if self._check_line((ay, ax), (by, bx)):
                g.add_edge(
                    a, b, **{DISTANCE: np.sqrt((ax - bx)**2 + (ay - by)**2)})
                g.add_node(a, **{POS: (ax, ay)})
                g.add_node(b, **{POS: (bx, by)})
        print("nodes", g.number_of_nodes())
        print("edges", g.number_of_edges())

        # plot
        # fig, ax = plt.subplots(figsize=(10, 10), dpi=DPI)
        # nx.draw(
        #     g,
        #     ax=ax,
        #     node_size=4,
        #     width=0.2,
        #     pos=nx.get_node_attributes(g, POS),
        # )
        # ax.set_aspect("equal")
        # fig.tight_layout()
        # fig.savefig("a.png")

        self._set_graph(g)


class ORM(RoadmapToTest):
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)
        from roadmaps.var_odrm_torch.var_odrm_torch import (
            make_graph_and_flann, optimize_poses, sample_points)
        n = roadmap_specific_kwargs["n"]
        epochs = roadmap_specific_kwargs["epochs"]
        lr = roadmap_specific_kwargs["lr"]

        start_t = timeit.default_timer()
        pos = sample_points(n, self.map_img, self.rng)
        g, _ = make_graph_and_flann(pos, self.map_img, n, rng)

        optimizer = torch.optim.Adam([pos], lr=lr)

        for i_e in tqdm(range(epochs)):
            g, pos, test_length, training_length = optimize_poses(
                g, pos, self.map_img, optimizer, n, rng)
        end_t = timeit.default_timer()
        self.runtime_ms = (end_t - start_t) * 1000

        # swap x and y
        nx.set_node_attributes(g,
                               {i: (p[1],
                                    p[0]) for i, p in nx.get_node_attributes(
                                   g, POS).items()}, POS)

        self._set_graph(g)


class PRM(RoadmapToTest):
    # Hint: PRM is a special case of ORM without optimization
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)
        from roadmaps.var_odrm_torch.var_odrm_torch import (sample_points)
        n = roadmap_specific_kwargs['n']

        g = nx.Graph()
        start_t = timeit.default_timer()
        pos = sample_points(n, self.map_img, self.rng)
        pos_np = pos.detach().numpy()
        for i in range(n):
            g.add_node(i, **{POS: (pos_np[i, 0], pos_np[i, 1])})
            for j in range(i):
                length = np.linalg.norm(pos_np[i] - pos_np[j])
                if length > roadmap_specific_kwargs['radius']:
                    continue
                if self._check_line(
                    (pos_np[i, 0].item(), pos_np[i, 1].item()),
                        (pos_np[j, 0].item(), pos_np[j, 1].item())):
                    g.add_edge(i, j, **{DISTANCE: length})

        end_t = timeit.default_timer()
        self.runtime_ms = (end_t - start_t) * 1000

        # swap x and y
        nx.set_node_attributes(g,
                               {i: (p[1],
                                    p[0]) for i, p in nx.get_node_attributes(
                                   g, POS).items()}, POS)

        self._set_graph(g)


class GridMap(RoadmapToTest):
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)
        n_side = ceil(sqrt(roadmap_specific_kwargs["n"]))
        start_t = timeit.default_timer()
        g = self._make_gridmap(n_side)
        while g.number_of_nodes() < roadmap_specific_kwargs["n"]:
            n_side += 1
            g = self._make_gridmap(n_side)
        end_t = timeit.default_timer()
        self.runtime_ms = (end_t - start_t) * 1000
        self._set_graph(g)

    def _make_gridmap(self, n_side):
        edge_length = 1 / (n_side + 1)
        g = nx.Graph()
        grid = np.full((n_side, n_side), -1)
        for x, y in product(range(n_side), range(n_side)):
            i_to_add = len(g)
            coords = (
                x * edge_length + edge_length / 2,
                y * edge_length + edge_length / 2,
            )
            if is_coord_free(self.map_img, coords):
                g.add_node(i_to_add, **{POS: coords})
                grid[x, y] = i_to_add
                if x > 0 and grid[x - 1, y] != -1:
                    if self._check_line(
                            coords,
                            g.nodes[grid[x - 1, y]][POS]):
                        g.add_edge(i_to_add, grid[x - 1, y], **{
                            DISTANCE: edge_length
                        })
                if y > 0 and grid[x, y - 1] != -1:
                    if self._check_line(
                            coords,
                            g.nodes[grid[x, y - 1]][POS]):
                        g.add_edge(i_to_add, grid[x, y - 1], **{
                            DISTANCE: edge_length
                        })
        # swap x and y
        nx.set_node_attributes(
            g,
            {i: (p[1],
                 p[0]) for i, p in nx.get_node_attributes(
                g, POS).items()}, POS)
        return g


# class VisibilityGraph(RoadmapToTest):
#     def __init__(self,
#                  map_fname: str,
#                  rng: Random,
#                  roadmap_specific_kwargs):
#         super().__init__(map_fname, rng, roadmap_specific_kwargs)
#         n = roadmap_specific_kwargs["n"]
#         start_t = timeit.default_timer()
#         g = self._make_visibility_graph(n)
#         end_t = timeit.default_timer()
#         self.runtime_ms = (end_t - start_t) * 1000
#         self._set_graph(g)

#     def _make_visibility_graph(self, n):


def run():
    df = pd.DataFrame()

    trials = [
        (GSRM, {
            'DA': 0.14,
            'DB': 0.06,
            'f': 0.035,
            'k': 0.065,
            'delta_t': 1.0,
            'iterations': 10000,
            'target_n': 500,
            'plot': True,
        }),
        (GSRM, {
            'DA': 0.14,
            'DB': 0.06,
            'f': 0.035,
            'k': 0.065,
            'delta_t': 1.0,
            'iterations': 10000,
            'target_n': 900,
        }),
        (GSRM, {
            'DA': 0.14,
            'DB': 0.06,
            'f': 0.035,
            'k': 0.065,
            'delta_t': 1.0,
            'iterations': 10000,
            'target_n': 1500,
        }),
        # (GSORM, {
        #     'DA': 0.14,
        #     'DB': 0.06,
        #     'f': 0.035,
        #     'k': 0.065,
        #     'delta_t': 1.0,
        #     'iterations': 5000,  # of grey scott model
        #     'resolution': 300,
        #     'epochs_optim': 25,  # of optimization
        #     'lr_optim': 1e-3,
        # }),
        # (GSORM, {
        #     'DA': 0.14,
        #     'DB': 0.06,
        #     'f': 0.035,
        #     'k': 0.065,
        #     'delta_t': 1.0,
        #     'iterations': 5000,
        #     'resolution': 400,
        #     'epochs_optim': 25,  # of optimization
        #     'lr_optim': 1e-3,
        # }),
        # (GSORM, {
        #     'DA': 0.14,
        #     'DB': 0.06,
        #     'f': 0.035,
        #     'k': 0.065,
        #     'delta_t': 1.0,
        #     'iterations': 5000,
        #     'resolution': 500,
        #     'epochs_optim': 25,  # of optimization
        #     'lr_optim': 1e-3,
        # }),
        (SPARS, {
            'denseDelta':    80.,
            'sparseDelta':    800,
            'stretchFactor': 1.01,
            'maxFailures': 500,
            'maxTime': 8.,
        }),
        (SPARS, {
            'denseDelta':    50.,
            'sparseDelta':    500.,
            'stretchFactor': 1.01,
            'maxFailures': 500,
            'maxTime': 8.,
        }),
        (SPARS, {
            'denseDelta':    20.,
            'sparseDelta':    200.,
            'stretchFactor': 1.01,
            'maxFailures': 500,
            'maxTime': 8.,
        }),
        (ORM, {
            'n': 400,
            'lr': 1e-3,
            'epochs': 50,
        }),
        (ORM, {
            'n': 900,
            'lr': 1e-3,
            'epochs': 50,
        }),
        (ORM, {
            'n': 1400,
            'lr': 1e-3,
            'epochs': 50,
        }),
        (PRM, {
            'n': 500,
            'radius': 0.08,
        }),
        (PRM, {
            'n': 900,
            'radius': 0.05,
        }),
        (PRM, {
            'n': 1500,
            'radius': 0.04,
        }),
        (GridMap, {
            'n': 400,
        }),
        (GridMap, {
            'n': 900,
        }),
        (GridMap, {
            'n': 1400,
        })
        # TODO: visibility graphs, voronoi diagrams
    ]
    if not os.path.exists(PLOT_FOLDER):
        os.makedirs(PLOT_FOLDER)
    if not len(os.listdir(PLOT_FOLDER)) == 0:
        # delete all files in folder
        for f in os.listdir(PLOT_FOLDER):
            os.remove(os.path.join(PLOT_FOLDER, f))
    params_to_run = []
    df['i'] = np.nan
    df.set_index('i', inplace=True)
    for cls, args in trials:
        for map_name in MAP_NAMES:
            if 'plot' in args:
                if map_name == 'c' and args['plot']:
                    print(f'Plotting {len(df) + 1}')
                else:
                    args['plot'] = False

            for seed in range(N_SEEDS):
                i = len(df) + 1  # new experiment in new row
                df.at[i, 'i'] = i
                df.at[i, 'map'] = map_name
                df.at[i, 'roadmap'] = cls.__name__
                df.at[i, 'seed'] = seed
                params_to_run.append((cls, args, map_name, seed, i))
    Random(0).shuffle(params_to_run)
    df = df.copy()

    for ptr in tqdm(params_to_run):
        cls, args, map_name, seed, i = ptr
        _, data = _run_proxy(ptr)
        for k, v in data.items():
            df.at[i, k] = v
        for k, v in args.items():
            df.at[i, cls.__name__ + "_" + k] = v
    # with Pool(2) as p:
    #     for i, data in p.imap_unordered(_run_proxy, params_to_run):
    #         for k, v in data.items():
    #             df.at[i, k] = v
    #         for k, v in args.items():
    #             df.at[i, cls.__name__ + "_" + k] = v

    df.head()
    df.to_csv(CSV_PATH)


def _run_proxy(args):
    cls, args, map_name, seed, i = args
    map_fname = f"roadmaps/odrm/odrm_eval/maps/{map_name}.png"
    t = cls(map_fname, Random(seed), args)
    t.plot_example(PLOT_FOLDER, i)
    data = t.evaluate()
    return (i, data)


def _get_i_from_column_name(col_name):
    return int(col_name.split("_")[-1])


def _get_cols_by_prefix(df, prefix):
    return filter(
        # must math {prefix}XX
        lambda x: re.match(f"^({prefix})[0-9]+$", x),
        df.columns)


def plot():
    interesting_vars = [
        'path_length_mean',
        'rel_straight_length_mean',
        'n_nodes',
        'visited_nodes_mean',
        'success_rate',
        'runtime_ms'
    ]

    df = pd.read_csv(CSV_PATH)
    sns.set_theme(style='whitegrid')
    sns.set(rc={'figure.dpi': DPI})
    sns.pairplot(
        df,
        hue='roadmap',
        vars=interesting_vars,
        markers='.',
    )
    plt.savefig(CSV_PATH.replace('.csv', '.png'))
    plt.close('all')

    for roadmap in df.roadmap.unique():
        df_roadmap = df[df.roadmap == roadmap]
        sns.set_theme(style='whitegrid')
        sns.pairplot(
            df_roadmap,
            hue='map',
            x_vars=[
                col for col in df_roadmap.columns if col.startswith(roadmap)],
            y_vars=interesting_vars,
        )
        plt.savefig(CSV_PATH.replace('.csv', f'_{roadmap}.png'))
        plt.close('all')

    COMPARE_TO = 'GSRM'
    assert COMPARE_TO in df.roadmap.unique(), \
        f'{COMPARE_TO=} must be in {df.roadmap.unique()=}'
    for roadmap in df.roadmap.unique():
        if roadmap == COMPARE_TO:
            continue
        df_roadmap = df[df.roadmap == roadmap]
        df_compare = df[df.roadmap == COMPARE_TO]
        data_len = []
        data_rel = []
        map_names_len = []
        map_names_rel = []
        # this is not ideal, but it works
        n_rows = min(len(df_roadmap), len(df_compare))
        for data, map_names, prefix in [
            (data_len, map_names_len, 'path_length_'),
            (data_rel, map_names_rel, 'rel_straight_length_')
        ]:
            cols = _get_cols_by_prefix(df_roadmap, prefix)
            i_s = [_get_i_from_column_name(col) for col in cols]
            for row in range(n_rows):
                map_name = df_roadmap.iloc[row]['map']
                for i in i_s:
                    data.append((
                        df_compare.iloc[row][f'{prefix}{i:02d}'],
                        df_roadmap.iloc[row][f'{prefix}{i:02d}'],
                    ))
                    map_names.append(map_name)
        assert len(data_len) == len(map_names_len), \
            f'{len(data_len)=} != {len(map_names_len)=}'
        assert len(data_rel) == len(map_names_rel), \
            f'{len(data_rel)=} != {len(map_names_rel)=}'
        data_len_np = np.array(data_len)
        data_rel_np = np.array(data_rel)
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=DPI)
        # sort by map names list
        sorted_map_names = []
        all_map_names = set()
        all_map_names.update(map_names_len)
        all_map_names.update(map_names_rel)
        for map_name in MAP_NAMES:
            if map_name in all_map_names:
                sorted_map_names.append(map_name)
        for i, (d, mns) in enumerate([
            (data_len_np, map_names_len),
            (data_rel_np, map_names_rel),
        ]):
            colors_per_data = []
            for map_name in mns:
                colors_per_data.append(
                    plt.cm.get_cmap('hsv')(sorted_map_names.index(map_name) /
                                           len(sorted_map_names)))
            axs[i].scatter(d[:, 0],
                           d[:, 1],
                           marker='.',
                           alpha=.4,
                           c=colors_per_data,
                           edgecolors='none',
                           s=2)
            axs[i].set_xlabel(COMPARE_TO)
            axs[i].set_ylabel(roadmap)
            axs[i].plot([0, 1000], [0, 1000], color='black', linewidth=1)
            axs[i].set_xlim(min(d[:, 0]) - .1, max(d[:, 0]) + .1)
            axs[i].set_ylim(min(d[:, 1]) - .1, max(d[:, 1]) + .1)
        axs[0].set_title('Path Length')
        axs[1].set_title('Relative Path Length')

        # make legend
        legend_elements = [
            plt.Line2D([-1], [-1],
                       marker='.',
                       color='w',
                       label=map_name,
                       markerfacecolor=plt.cm.get_cmap('hsv')(
                       sorted_map_names.index(map_name) /
                       len(sorted_map_names)),
                       markersize=10)
            for map_name in sorted_map_names
        ]
        axs[0].legend(handles=legend_elements, loc='upper left')

        fig.savefig(CSV_PATH.replace('.csv', f'_{roadmap}_scatter.png'))
        plt.close('all')

        # KDE Plot similar to scatter plot
        fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=DPI)
        for i, (d, mns) in enumerate([
            (data_len_np, map_names_len),
            (data_rel_np, map_names_rel),
        ]):
            colors_per_data = []
            for map_name in mns:
                colors_per_data.append(
                    plt.cm.get_cmap('hsv')(sorted_map_names.index(map_name) /
                                           len(sorted_map_names)))
            sns.kdeplot(
                x=d[:, 0],
                y=d[:, 1],
                ax=axs[i],
                shade=True,
                shade_lowest=False,
                cmap='hsv',
                alpha=.4,
                levels=100,
                cbar=True,
                cbar_ax=axs[i].cax,
                cbar_kws={
                    'ticks': [0, .5, 1],
                    'label': 'Density'
                },
            )
            axs[i].set_xlabel(COMPARE_TO)
            axs[i].set_ylabel(roadmap)
            axs[i].plot([0, 1000], [0, 1000], color='black', linewidth=1)
            axs[i].set_xlim(min(d[:, 0]) - .1, max(d[:, 0]) + .1)
            axs[i].set_ylim(min(d[:, 1]) - .1, max(d[:, 1]) + .1)


if __name__ == '__main__':
    run()
    plot()
