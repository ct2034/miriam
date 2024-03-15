import logging
import os
import re
import timeit
from itertools import product
from math import ceil, sqrt
from random import Random
from typing import Any, Dict, List, Optional, Tuple

import cv2 as cv
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bresenham import bresenham
import matplotlib
from matplotlib import pyplot as plt
from pyflann import FLANN
from tqdm import tqdm

from definitions import DISTANCE, POS, MAP_IMG
from planner.astar_boost.build.libastar_graph import AstarSolver
from planner.astar_boost.converter import initialize_from_graph
from roadmaps.var_odrm_torch.var_odrm_torch import (
    PATH_W_COORDS, get_path_len, is_coord_free, is_pixel_free, make_paths,
    plan_path_between_coordinates, read_map)

# fix for papercept --This document has a Type 3 font (on page 6)
# https://tex.stackexchange.com/questions/77968/how-do-i-avoid-type3-fonts-when-submitting-to-manuscriptcentral
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

logger = logging.getLogger(__name__)


CSV_PATH = 'roadmaps/benchmark.csv'
CSV_MEAN_PATH = 'roadmaps/benchmark_w_mean.csv'
PLOT_FOLDER = 'roadmaps/benchmark_plots'
PLOT_FOLDER_PAPER = 'roadmaps/benchmark_plots_paper'
EXAMPLE_FOLDER = 'roadmaps/benchmark_examples'
GSRM_EXAMPLE_FOLDER = 'roadmaps/gsorm/examples'
DPI = 500

# this list is sorted roughly by complexity
MAP_NAMES = [
    'den520',
    'plain',
    # 'c',
    # 'x',
    # 'b',
    # 'o',
    # 'dual_w',
    # 'dual2',
    # 'dual',
    # 'z',
    # 'dense34',
    # 'dense',
    # 'rooms_20',
    # 'rooms2_30',
    'rooms_30',
    # 'rooms_40',
    # 'simple',
    'slam',
    # 'brc201',
]
PLOT_GSRM_ON_MAP = 'slam'
assert PLOT_GSRM_ON_MAP in MAP_NAMES
N_SEEDS = 10

edge_radius_stats = {}


class RoadmapToTest:
    def __init__(self, map_fname: str, rng: Random,
                 roadmap_specific_kwargs: Dict[str, Any] = {}):
        if self.__class__ == GSRM:
            self.map_fname = map_fname
        else:
            map_name = os.path.splitext(os.path.basename(map_fname))[0]
            n: Optional[int] = None
            target_n = roadmap_specific_kwargs['target_n']
            ideal_n = roadmap_specific_kwargs['ideal_n']
            self.map_fname = os.path.join(
                EXAMPLE_FOLDER, f"{map_name}_inflated_{int(ideal_n)}.png")
        self.map_img = read_map(self.map_fname)
        # swap rows and columns
        # self.map_img = np.swapaxes(np.array(self.map_img), 0, 1)
        self.n_eval = 100
        self.rng = rng
        self.roadmap_specific_kwargs = roadmap_specific_kwargs
        self.g: Optional[nx.Graph] = None
        self.runtime_ms: Optional[float] = None
        self.astar_solver: Optional[AstarSolver] = None
        self.flann: Optional[FLANN] = None

        print('\n' + '+' * 80)
        print(f'{self.__class__.__name__=}')
        print(f'{map_fname=}')
        print(f'{roadmap_specific_kwargs=}')

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

    def _check_line(
            self,
            a: Tuple[float, float],
            b: Tuple[float, float],
            map_img: Optional[MAP_IMG] = None
    ) -> bool:
        if map_img is None:
            map_img = self.map_img
        line = bresenham(
            round(a[0] * len(map_img)),
            round(a[1] * len(map_img)),
            round(b[0] * len(map_img)),
            round(b[1] * len(map_img)),
        )
        # print(list(line))
        return all([is_pixel_free(
            map_img, (x[0], x[1])) for x in line])

    def evaluate_path_length(self) -> Dict[str, float]:
        assert self.g is not None, 'Roadmap must be built.'
        if self.g.number_of_nodes() == 0:
            logger.error('Error while making paths, no nodes in graph.')
            return {'success_rate': 0.0}
        map_img_inv = np.swapaxes(np.array(self.map_img), 0, 1)
        data = {}
        pos_t = torch.tensor(self.pos_np)
        for i in range(self.n_eval):
            a = self.rng.random(), self.rng.random()
            b = self.rng.random(), self.rng.random()
            while not is_coord_free(map_img_inv, a) or \
                    not is_coord_free(map_img_inv, b):
                a = self.rng.random(), self.rng.random()
                b = self.rng.random(), self.rng.random()
            path_w_coords, visited_nodes = self._plan_path_coords(a, b)
            success = True
            if path_w_coords[-1] is None:
                success = False
            elif len(path_w_coords[-1]) == 0:
                success = False
            elif not self._check_line(
                    a,
                    pos_t[path_w_coords[-1][0]].tolist(),
                    map_img_inv):
                success = False
            elif not self._check_line(
                    b,
                    pos_t[path_w_coords[-1][-1]].tolist(),
                    map_img_inv):
                success = False
            data[f'success_{i:03d}'] = success
            if success:
                data[f'path_length_{i:03d}'] = get_path_len(
                    pos_t, path_w_coords, False).item()
                data[f'visited_nodes_{i:03d}'] = visited_nodes
            else:
                data[f'path_length_{i:03d}'] = np.nan
                data[f'visited_nodes_{i:03d}'] = np.nan
        return data

    def evaluate_straight_path_length(self) -> Dict[str, float]:
        assert self.g is not None, 'Roadmap must be built.'
        if self.g.number_of_nodes() == 0:
            logger.error('Error while making paths, no nodes in graph.')
        pos_np = np.zeros((max(self.g.nodes) + 1, 2))
        for n, (x, y) in nx.get_node_attributes(self.g, POS).items():
            pos_np[n] = torch.tensor([x, y])
        pos_t = torch.tensor(pos_np)
        flann = FLANN(random_seed=0)
        flann.build_index(np.array(pos_np, dtype=np.float32), random_seed=0)
        data = {}
        for i in range(self.n_eval):
            found = False
            while not found:
                start = (self.rng.random(), self.rng.random())
                if not is_coord_free(self.map_img, start):
                    continue
                end = (self.rng.random(), self.rng.random())
                if not is_coord_free(self.map_img, end):
                    continue
                if not np.linalg.norm(np.array(start) - np.array(end)) > 0.5:
                    # should have some length
                    continue
                if not self._check_line(start, end):
                    continue
                straight_len = np.linalg.norm(
                    np.array(start) - np.array(end)).item()
                if straight_len < 0.5:
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
            data[f"rel_straight_length_{i:03d}"] = path_len / straight_len
        return data

    def evaluate_n_nodes(self) -> Dict[str, float]:
        assert self.g is not None, "Roadmap must be built."
        return {
            'n_nodes': self.g.number_of_nodes(),
            'n_edges': self.g.number_of_edges()}

    def evaluate_runtime(self) -> Dict[str, float]:
        if self.runtime_ms is None:
            return {}
        return {"runtime_ms": self.runtime_ms}

    def evaluate(self):
        results = {}
        for fun in [
            self.evaluate_path_length,
            # self.evaluate_straight_path_length,
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
        pos_torch = torch.tensor(list(pos.values()))
        pos = {k: (v[0] * len(self.map_img), v[1] * len(self.map_img))
               for k, v in pos.items()}
        nx.draw_networkx_nodes(self.g, pos, ax=ax, node_size=1)
        # exclude self edges
        edges = [(u, v) for u, v in self.g.edges if u != v]
        nx.draw_networkx_edges(self.g, pos, ax=ax, edgelist=edges, width=0.5)

        n_nodes = self.g.number_of_nodes()
        if n_nodes == 0:
            return

        # make an example path
        # invert x and y of map
        map_img = np.swapaxes(np.array(self.map_img), 0, 1)
        map_img_t = tuple(map_img.tolist())
        seed = 0
        path = None
        while path is None:
            try:
                path = make_paths(self.g, 1, map_img_t, Random(seed))[0]
                path_len = get_path_len(
                    pos_torch, path, False).item()
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

        # what will be the file name
        map_no_ext = os.path.splitext(
            os.path.basename(self.map_fname))[0]

        # save
        for extenstion in ['pdf', 'png']:
            name = f"{i:03d}_{self.__class__.__name__}_" +\
                f"n_nodes{n_nodes:04d}_" +\
                f"n_edges{self.g.number_of_edges():04d}_" +\
                f"path_len{path_len:.3f}_" +\
                f"{map_no_ext}" +\
                f".{extenstion}"
            fig.savefig(os.path.join(folder, name))
        plt.close(fig)


class GSRM(RoadmapToTest):
    """Its gsorm without the o.
    So no optimization, only the generation of the points by the
    Grey Scott model."""

    resolution_stats = {}

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

        # initialize this with the number of nodes we expect to get at the
        # (initial) resolution set below
        actual_n = 100  # 100 works right now
        assert target_n >= actual_n, f'{target_n=} must be >= {actual_n=}'

        def _goal_check(target_n, actual_n):
            return abs(target_n - actual_n) < 0.1 * target_n

        pos = None
        from roadmaps.gsorm.build.libgsorm import Gsorm
        from roadmaps.var_odrm_torch.var_odrm_torch import make_graph_and_flann

        first_run = True

        if map_fname not in GSRM.resolution_stats:
            GSRM.resolution_stats[map_fname] = ([], [])
        resolutions, ns = GSRM.resolution_stats[map_fname]
        
        resolution = 250

        while not _goal_check(target_n, actual_n) or first_run:
            first_run = False
            if len(resolutions) < 1:
                resolution = 250
            elif len(resolutions) < 2:
                resolution = int(resolution * 2)
            else:
                # fit linear fn through resolutions and ns
                m, b = np.polyfit(np.sqrt(ns), resolutions, 1)
                # find resolution that gives target_n
                resolution = int(m * sqrt(target_n) + b)
                if resolution == resolutions[-1]:
                    if target_n > actual_n:
                        resolution += 1
                    else:  # target_n < actual_n
                        resolution -= 1
            kwargs['resolution'] = resolution
            gs = Gsorm()
            nodes, runtime_points = gs.run(
                mapFile=map_fname,
                **kwargs,
                seed=rng.randint(0, 2**8),
            )
            pos = torch.Tensor(nodes) / resolution
            # swap x and y
            pos = pos[:, [1, 0]]
            n = pos.shape[0]
            if n < 10:
                # not enough points, no need to make a graph
                actual_n = n
                continue
            start_t = timeit.default_timer()
            g, _ = make_graph_and_flann(pos, self.map_img, n, rng)
            runtime_delaunay = (timeit.default_timer() - start_t) * 1000
            actual_n = g.number_of_nodes()
            resolutions.append(resolution)
            ns.append(actual_n)
            if _goal_check(target_n, actual_n):
                print(f'✅ {actual_n=}, {target_n=}, {resolution=}')
            else:
                print(f'❌ {actual_n=}, {target_n=}, {resolution=}')
        assert pos is not None

        self.runtime_ms = runtime_points + runtime_delaunay

        # statistics on edge lengths
        edge_lengths = []
        for (a, b) in g.edges:
            if a == b:
                continue
            edge_lengths.append(
                np.linalg.norm(
                    np.array(nx.get_node_attributes(g, POS)[a]) -
                    np.array(nx.get_node_attributes(g, POS)[b])))
        edge_lengths = np.array(edge_lengths)
        edge_len_median = np.median(edge_lengths)
        edge_len_std = np.std(edge_lengths)
        EDGE_LEN_TO_RADIUS_SCALE = 0.7
        agent_radius = edge_len_median * EDGE_LEN_TO_RADIUS_SCALE
        print(f"{edge_len_median=}, {edge_len_std=}, {agent_radius=}")
        if map_fname not in edge_radius_stats:
            edge_radius_stats[map_fname] = {}
        edge_radius_stats[map_fname][target_n] = agent_radius

        # if the right map is already saved, load it
        map_name = os.path.splitext(os.path.basename(map_fname))[0]
        inflated_map_fname = os.path.join(
            EXAMPLE_FOLDER, f"{map_name}_inflated_{int(target_n)}.png")
        if os.path.exists(inflated_map_fname):
            self.map_img = read_map(inflated_map_fname)
        else:
            # inflate map
            agent_radius_map = int(agent_radius * len(self.map_img))
            map_img_np = np.array([list(row) for row in self.map_img])
            elm = cv.getStructuringElement(
                cv.MORPH_ELLIPSE,
                (agent_radius_map, agent_radius_map))
            inflated_map = cv.erode(map_img_np.astype(np.uint8), elm)
            cv.imwrite(inflated_map_fname, inflated_map)
            self.map_img = tuple([tuple(row) for row in inflated_map])
        g, _ = make_graph_and_flann(pos, self.map_img, n, rng)

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


class CVT(RoadmapToTest):
    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)
        from roadmaps.cvt.build.libcvt import CVT
        from roadmaps.var_odrm_torch.var_odrm_torch import (
            make_graph_and_flann, optimize_poses)

        # prepare args
        cvt_kwargs = self.roadmap_specific_kwargs.copy()

        # run cvt
        cvt = CVT()
        nodes, runtime_points = cvt.run(
            mapFile=map_fname,
            **cvt_kwargs,
            plot=False,
        )
        pos = torch.Tensor(nodes) / roadmap_specific_kwargs["resolution"]
        # swap x and y
        pos = pos[:, [1, 0]]
        pos.requires_grad = True
        n = pos.shape[0]
        g, _ = make_graph_and_flann(pos, self.map_img, n, rng)

        # swap x and y
        nx.set_node_attributes(g,
                               {i: (p[1],
                                    p[0]) for i, p in nx.get_node_attributes(
                                   g, POS).items()}, POS)
        self._set_graph(g)


class SPARS2(RoadmapToTest):

    size_stats = {}

    def __init__(self,
                 map_fname: str,
                 rng: Random,
                 roadmap_specific_kwargs):
        super().__init__(map_fname, rng, roadmap_specific_kwargs)

        from roadmaps.SPARS.build.libsparspy import Spars
        target_n = self.roadmap_specific_kwargs['target_n']
        spars_kwargs = self.roadmap_specific_kwargs.copy()
        if 'target_n' in spars_kwargs:
            spars_kwargs.pop('target_n')
        if 'ideal_n' in spars_kwargs:
            spars_kwargs.pop('ideal_n')
        dense_to_sparse_multiplier = spars_kwargs.pop(
            'dense_to_sparse_multiplier')
        stretch_factor = spars_kwargs.pop('stretchFactor')
        dense_delta = 2.5
        sparse_delta = dense_delta * dense_to_sparse_multiplier
        n_nodes = 0

        def _goal_check(target_n, actual_n):
            return abs(target_n - actual_n) < 0.1 * target_n

        if map_fname not in SPARS2.size_stats:
            SPARS2.size_stats[map_fname] = {}
        if target_n not in SPARS2.size_stats[map_fname]:
            SPARS2.size_stats[map_fname][target_n] = ([], [], [], [])
        dense_deltas, sparse_deltas, stretch_factors, ns = SPARS2.size_stats[map_fname][target_n]
        dense_delta = 1.5
        sparse_delta = dense_delta * dense_to_sparse_multiplier

        while not _goal_check(target_n, n_nodes):
            print(f"Got {n_nodes} nodes, target was {target_n}.")
            print(f"Trying denseDelta={dense_delta}, "
                  f"sparseDelta={sparse_delta}...")

            spars_kwargs['denseDelta'] = dense_delta
            spars_kwargs['sparseDelta'] = sparse_delta
            spars_kwargs['stretchFactor'] = stretch_factor
            s = Spars()
            edges, self.runtime_ms = s.run(
                mapFile=self.map_fname,
                seed=rng.randint(0, 2**16),
                **spars_kwargs,
            )

            # to networkx
            g = nx.Graph()
            for (a, ax, ay, b, bx, by) in edges:
                # positions to unit square
                ax /= len(self.map_img)
                ay /= len(self.map_img)
                bx /= len(self.map_img)
                by /= len(self.map_img)
                # if self._check_line((ay, ax), (by, bx)):
                g.add_edge(
                    a, b, **{DISTANCE: np.sqrt((ax - bx)**2 + (ay - by)**2)})
                g.add_node(a, **{POS: (ax, ay)})
                g.add_node(b, **{POS: (bx, by)})
            print("nodes", g.number_of_nodes())
            print("edges", g.number_of_edges())
            n_nodes = g.number_of_nodes()

            if _goal_check(target_n, n_nodes):
                break

            if len(ns) == 0:
                dense_deltas.append(dense_delta)
                sparse_deltas.append(sparse_delta)
                stretch_factors.append(stretch_factor)
                ns.append(n_nodes)
            else:
                # if we improved, keep the values
                if abs(target_n - n_nodes) < abs(target_n - ns[-1]):
                    dense_deltas.append(dense_delta)
                    sparse_deltas.append(sparse_delta)
                    stretch_factors.append(stretch_factor)
                    ns.append(n_nodes)
                # if we didn't improve, try again with the last value
                else: 
                    dense_delta = dense_deltas[-1]
                    sparse_delta = sparse_deltas[-1]
                    stretch_factor = stretch_factors[-1]

            # adjust deltas
            dense_delta *= rng.gauss(1, 0.1)
            sparse_delta *= rng.gauss(1, 0.1)
            stretch_factor *= rng.gauss(1, 0.1)
            
            print(f"{len(dense_deltas)=})")

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
        n = roadmap_specific_kwargs["target_n"]
        epochs = roadmap_specific_kwargs["epochs"]
        lr = roadmap_specific_kwargs["lr"]

        start_t = timeit.default_timer()
        pos = sample_points(n, self.map_img, self.rng)
        g, _ = make_graph_and_flann(pos, self.map_img, n, rng)

        optimizer = torch.optim.Adam([pos], lr=lr)

        for _ in tqdm(range(epochs)):
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
        from roadmaps.var_odrm_torch.var_odrm_torch import sample_points
        n = int(roadmap_specific_kwargs['target_n'])
        n_edges = roadmap_specific_kwargs['n_edges']
        radius = roadmap_specific_kwargs['start_radius']

        actual_n_edges = 0
        while actual_n_edges < n_edges:
            print(f"Trying radius={radius}...")
            g = nx.Graph()
            start_t = timeit.default_timer()
            pos = sample_points(n, self.map_img, self.rng)
            pos_np = pos.detach().numpy()
            for i in range(n):
                g.add_node(i, **{POS: (pos_np[i, 0], pos_np[i, 1])})
                for j in range(i):
                    length = np.linalg.norm(pos_np[i] - pos_np[j])
                    if length > radius:
                        continue
                    if self._check_line(
                        (pos_np[i, 0].item(), pos_np[i, 1].item()),
                            (pos_np[j, 0].item(), pos_np[j, 1].item())):
                        g.add_edge(i, j, **{DISTANCE: length})

            end_t = timeit.default_timer()
            actual_n_edges = g.number_of_edges()
            print(f"Got {actual_n_edges} edges, target was {n_edges}.")
            radius *= 1.1
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
        target_n = roadmap_specific_kwargs["target_n"]
        n_side = ceil(sqrt(target_n))
        start_t = timeit.default_timer()
        g = self._make_gridmap(n_side)
        while g.number_of_nodes() < target_n:
            n_side += 1
            g = self._make_gridmap(n_side)
        end_t = timeit.default_timer()
        self.runtime_ms = (end_t - start_t) * 1000
        self._set_graph(g)

    def _make_gridmap(self, _):
        raise NotImplementedError("Implement in subclass.")


class GridMap4(GridMap):
    def _make_gridmap(self, n_side):
        edge_length = 1 / (n_side + 1)
        g = nx.Graph()
        grid = np.full((n_side, n_side), -1)
        for x, y in product(range(n_side), range(n_side)):
            i_to_add = len(g)
            coords = (
                x * edge_length + edge_length / 2,
                y * edge_length + edge_length / 2
            )
            if is_coord_free(self.map_img, coords):
                g.add_node(i_to_add, **{POS: coords})
                grid[x, y] = i_to_add
                if x > 0 and grid[x - 1, y] != -1 and self._check_line(
                        coords,
                        g.nodes[grid[x - 1, y]][POS]):
                    g.add_edge(i_to_add, grid[x - 1, y], **{
                        DISTANCE: edge_length
                    })
                if y > 0 and grid[x, y - 1] != -1 and self._check_line(
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


class GridMap8(GridMap):
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
                deltas = [
                    [0, -1],  # left
                    [-1, -1],  # up left
                    [-1, 0],  # up
                    [-1, 1],  # up right
                ]
                for dx, dy in deltas:
                    other_x, other_y = x + dx, y + dy
                    if (
                            other_x < 0 or
                            other_x >= n_side or
                            other_y < 0 or
                            other_y >= n_side):
                        continue
                    if grid[other_x, other_y] != -1 and self._check_line(
                            coords,
                            g.nodes[grid[other_x, other_y]][POS]):
                        g.add_edge(i_to_add, grid[other_x, other_y], **{
                            DISTANCE: np.linalg.norm(
                                np.array([dx, dy])) * edge_length
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
    ns = [270, 530, 1000]
    trials = [
        # (CVT, {
        #     'DA': 0.14,
        #     'DB': 0.06,
        #     'f': 0.035,
        #     'k': 0.065,
        #     'delta_t': 1.0,
        #     'iterations': 10000,
        #     'target_n': ns[0],
        #     'plot': True,
        # }),
        # (CVT, {
        #     'DA': 0.14,
        #     'DB': 0.06,
        #     'f': 0.035,
        #     'k': 0.065,
        #     'delta_t': 1.0,
        #     'iterations': 10000,
        #     'target_n': ns[1],
        # }),
        # (CVT, {
        #     'DA': 0.14,
        #     'DB': 0.06,
        #     'f': 0.035,
        #     'k': 0.065,
        #     'delta_t': 1.0,
        #     'iterations': 10000,
        #     'target_n': ns[2],
        # }),
        (GSRM, {
            'DA': 0.14,
            'DB': 0.06,
            'f': 0.035,
            'k': 0.065,
            'delta_t': 1.0,
            'iterations': 10000,
            'target_n': ns[0],
            'plot': True,
        }),
        (GSRM, {
            'DA': 0.14,
            'DB': 0.06,
            'f': 0.035,
            'k': 0.065,
            'delta_t': 1.0,
            'iterations': 10000,
            'target_n': ns[1],
        }),
        (GSRM, {
            'DA': 0.14,
            'DB': 0.06,
            'f': 0.035,
            'k': 0.065,
            'delta_t': 1.0,
            'iterations': 10000,
            'target_n': ns[2],
        }),
        (SPARS2, {
            'target_n': ns[0],
            'dense_to_sparse_multiplier': 50,
            'stretchFactor': 5,
            'maxFailures': 1000,
            'maxTime': 4.,  # ignored
            'maxIter': 40000,
        }),
        (SPARS2, {
            'target_n': ns[1],
            'dense_to_sparse_multiplier': 50,
            'stretchFactor': 5,
            'maxFailures': 1000,
            'maxTime': 4.,  # ignored
            'maxIter': 25000,
        }),
        (SPARS2, {
            'target_n': ns[2],
            'dense_to_sparse_multiplier': 50,
            'stretchFactor': 5,
            'maxFailures': 1000,
            'maxTime': 4.,  # ignored
            'maxIter': 20000,
        }),
        (ORM, {
            'target_n': ns[0],
            'lr': 5e-5,
            'epochs': 50,
        }),
        (ORM, {
            'target_n': ns[1],
            'lr': 5e-5,
            'epochs': 50,
        }),
        (ORM, {
            'target_n': ns[2],
            'lr': 5e-5,
            'epochs': 50,
        }),
        (PRM, {
            'target_n': ns[0],
            'start_radius': 0.06,
        }),
        (PRM, {
            'target_n': ns[1],
            'start_radius': 0.035,
        }),
        (PRM, {
            'target_n': ns[2],
            'start_radius': 0.025,
        }),
        # (GridMap4, {
        #     'target_n': ns[0],
        # }),
        # (GridMap4, {
        #     'target_n': ns[1],
        # }),
        # (GridMap4, {
        #     'target_n': ns[2],
        # }),
        (GridMap8, {
            'target_n': ns[0],
        }),
        (GridMap8, {
            'target_n': ns[1],
        }),
        (GridMap8, {
            'target_n': ns[2],
        })
    ]
    if not os.path.exists(EXAMPLE_FOLDER):
        os.makedirs(EXAMPLE_FOLDER)
    if len(os.listdir(EXAMPLE_FOLDER)) != 0:
        # delete all files in folder
        for f in os.listdir(EXAMPLE_FOLDER):
            os.remove(os.path.join(EXAMPLE_FOLDER, f))
    params_to_run = []
    df['i'] = np.nan
    df.set_index('i', inplace=True)
    for _cls, _args in trials:
        for map_name in MAP_NAMES:
            args = _args.copy()
            if args['target_n'] < 250 and _cls == SPARS2:
                continue # skip spars2 for small target_n
            if 'plot' in args:
                if map_name == PLOT_GSRM_ON_MAP and args['plot']:
                    print(f'Plotting Trial {len(df) + 1} on {map_name}')
                else:
                    args['plot'] = False

            for seed in range(N_SEEDS):
                i = len(df) + 1  # new experiment in new row
                df.at[i, 'i'] = i
                df.at[i, 'map'] = map_name
                df.at[i, 'roadmap'] = _cls.__name__
                df.at[i, 'seed'] = seed
                df.at[i, 'target_n'] = args['target_n']
                params_to_run.append((_cls, args, map_name, seed, i))
    df = df.copy()

    for i_p, ptr in enumerate(tqdm(params_to_run)):
        _cls, args, map_name, seed, i = ptr
        if 'ideal_n' in args:
            target_n = args['ideal_n']
        else:
            target_n = args['target_n']
        # for prm, we want to have the same n_edges as gsrm
        if _cls == PRM:
            ptr[1]['n_edges'] = df[
                df['roadmap'] == GSRM.__name__][
                df['map'] == map_name][
                df['GSRM_target_n'] == target_n
            ]['n_edges'].values[0]
        if _cls != GSRM:
            data_target_ns = df[
                df['roadmap'] == GSRM.__name__][
                df['map'] == map_name][
                df['target_n'] == target_n
            ]
            # assume that gsrm was already run
            assert len(
                data_target_ns) == N_SEEDS, f"Run gsrm first for {map_name=}, {args['target_n']=}"
            # get target_n from gsrm
            new_target_n = data_target_ns['n_nodes'].mean()
            if 'ideal_n' not in args:
                ptr[1]['ideal_n'] = target_n
            ptr[1]['target_n'] = new_target_n
        _, data = _run_proxy(ptr)
        for k, v in data.items():
            df.at[i, k] = v
        for k, v in args.items():
            df.at[i, _cls.__name__ + "_" + k] = v
        if i_p % 100 == 0:
            print("cleaning ...")
            df = df.copy()
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
    t.plot_example(EXAMPLE_FOLDER, i)
    if ('plot' in args and
        args['plot'] and
        seed == 0 and
            map_name == PLOT_GSRM_ON_MAP):
        t.plot_example(GSRM_EXAMPLE_FOLDER, i)
    data = t.evaluate()
    return (i, data)


def _get_i_from_column_name(col_name):
    return int(col_name.split("_")[-1])


def _get_cols_by_prefix(df, prefix):
    return filter(
        # must math {prefix}XX
        lambda x: re.match(f"^({prefix})[0-9]+$", x),
        df.columns)


def _make_sure_folder_exists_and_is_empty(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))


def prepare_mean():
    """Read the csv and prepare a new csv with additional columns
    that contain the mean of path_length and the success rate."""
    df = pd.read_csv(CSV_PATH)
    # mean only over successful runs
    maps_in_data = df.map.unique()
    roadmaps_in_data = df.roadmap.unique()
    target_ns_in_data = df.target_n.unique()
    target_ns_in_data = target_ns_in_data[
        np.logical_not(np.isnan(target_ns_in_data))]
    seeds_in_data = df.seed.unique()
    for map_, roadmap, target_n, seed in product(
        maps_in_data, roadmaps_in_data, target_ns_in_data, seeds_in_data
    ):
        mask_this_data = np.logical_and(
            df.map == map_,
            np.logical_and(
                df.seed == seed,
                np.logical_and(
                    df.target_n == target_n,
                    df.roadmap == roadmap)))
        if not mask_this_data.any():
            continue
        this_i = int(df[mask_this_data].index[0])
        mask_all_roadmaps = np.logical_and(
            df.map == map_,
            np.logical_and(
                df.seed == seed,
                df.target_n == target_n))
        success_cols = _get_cols_by_prefix(df, 'success_')
        # Filter out colums where all roadmaps where successful:
        success_cols = [
            col for col in success_cols if df[mask_all_roadmaps][col].all()]
        if len(success_cols) == 0:
            continue
        success_idx = [int(col.split("_")[-1]) for col in success_cols]
        df.at[this_i, 'path_length_mean'] = df[mask_this_data][
            [f'path_length_{i:03d}' for i in success_idx]
        ].mean(axis=1).values[0]
        # df.at[this_i, 'rel_straight_length_mean'] = df[mask_this_data][
        #     [f'rel_straight_length_{i:03d}' for i in success_idx
        #     ]].mean(axis=1)
        df.at[this_i, 'visited_nodes_mean'] = df[mask_this_data][
            [f'visited_nodes_{i:03d}' for i in success_idx]
        ].mean(axis=1).values[0]
        df.at[this_i, 'success_rate'] = df[mask_this_data][
            _get_cols_by_prefix(df, 'success_')
        ].mean(axis=1).values[0]
    df.to_csv(CSV_MEAN_PATH)


def plot():
    interesting_vars = [
        'path_length_mean',
        # 'rel_straight_length_mean',
        'n_nodes',
        'n_edges',
        'visited_nodes_mean',
        'success_rate',
        'runtime_ms'
    ]

    df = pd.read_csv(CSV_MEAN_PATH)
    sns.set_theme(style='whitegrid')
    sns.set(rc={'figure.dpi': DPI})
    _make_sure_folder_exists_and_is_empty(PLOT_FOLDER)

    print("Plotting results for all maps")
    sns.pairplot(
        df,
        hue='roadmap',
        vars=interesting_vars,
        markers='.',
        plot_kws={
            'alpha': 0.7,
            'edgecolor': 'none'},
    )
    plt.savefig(os.path.join(
        PLOT_FOLDER,
        os.path.basename(CSV_PATH).replace('.csv', '.png')))
    plt.close('all')

    print("Plotting results per map")
    for map_name in df.map.unique():
        df_map = df[df.map == map_name]
        sns.set_theme(style='whitegrid')
        sns.pairplot(
            df_map,
            hue='roadmap',
            vars=interesting_vars,
            markers='.',
            plot_kws={
                'alpha': 0.7,
                'edgecolor': 'none'},
        )
        plt.savefig(os.path.join(
            PLOT_FOLDER,
            os.path.basename(CSV_PATH).replace(
                '.csv', f'_map-{map_name}.png')))
        plt.close('all')

    print("Plotting results per roadmap")
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
        plt.savefig(os.path.join(
            PLOT_FOLDER,
            os.path.basename(CSV_PATH).replace(
                '.csv', f'_rm-{roadmap}.png')))
        plt.close('all')

    print("Making scatter plots")
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
                    other, our = (
                        df_compare.iloc[row][f'{prefix}{i:03d}'],
                        df_roadmap.iloc[row][f'{prefix}{i:03d}'],
                    )
                    if np.isnan(other) or np.isnan(our):
                        continue
                    data.append([other, our])
                    map_names.append(map_name)
        assert len(data_len) == len(map_names_len), \
            f'{len(data_len)=} != {len(map_names_len)=}'
        assert len(data_rel) == len(map_names_rel), \
            f'{len(data_rel)=} != {len(map_names_rel)=}'
        data_len_np = np.array(data_len)
        data_rel_np = np.array(data_rel)
        _, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=DPI)
        all_map_names = set()
        all_map_names.update(map_names_len)
        all_map_names.update(map_names_rel)
        sorted_map_names = [
            map_name for map_name in MAP_NAMES if map_name in all_map_names
        ]
        for i, (d, mns) in enumerate([
            (data_len_np, map_names_len),
            # (data_rel_np, map_names_rel),
        ]):
            colors_per_data = [
                plt.cm.get_cmap('hsv')(
                    sorted_map_names.index(map_name) / len(sorted_map_names)
                )
                for map_name in mns
            ]
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

        plt.savefig(os.path.join(
            PLOT_FOLDER,
            os.path.basename(CSV_PATH).replace('.csv', f'_scatter_{roadmap}.png')))
        plt.close('all')

    print('Done')


def _group_n_nodes(df, n_n_nodes):
    df_new = pd.DataFrame()
    n_nodes_per_rm = {}
    for roadmap in df.roadmap.unique():
        df_roadmap = df[df.roadmap == roadmap]
        n_nodes = df_roadmap.n_nodes.unique()
        n_nodes_per_goal = len(n_nodes) // n_n_nodes
        n_nodes_s_goal = [
            np.mean(n_nodes[i * n_nodes_per_goal: (i + 1) * n_nodes_per_goal])
            for i in range(n_n_nodes)
        ]
        n_nodes_per_rm[roadmap] = n_nodes_s_goal
    for i_row in range(len(df)):
        row = df.iloc[i_row].copy()
        n_nodes = n_nodes_per_rm[row.roadmap][
            np.argmin(np.abs(row.n_nodes - n_nodes_per_rm[row.roadmap]))]
        row['n_nodes'] = n_nodes
        df_new = pd.concat([df_new, row.to_frame().T])
    return df_new


def _remove_numbers_after_mapnames_and_cap(map_name: str) -> str:
    """For example, 'plain34' -> 'Plain' and 'Berlin_1_256' -> 'Berlin'"""
    map_name = re.sub(r'\d+', '', map_name)
    map_name = map_name.replace('_', '')
    return map_name.capitalize()


def plots_for_paper():
    df = pd.read_csv(CSV_MEAN_PATH)
    sns.set_theme(style='whitegrid')
    # sns.set(rc={'figure.dpi': DPI})
    _make_sure_folder_exists_and_is_empty(PLOT_FOLDER_PAPER)

    interesting_maps = [
        'plain',
        'den520',
        'rooms_30',
        'slam'
    ]
    n_plots = len(interesting_maps)
    n_n_nodes = len(df[df.roadmap == 'GSRM'].GSRM_target_n.unique())
    min_n = min(df.target_n.unique())
    print(f'{n_plots=}, {n_n_nodes=}')
    df = _group_n_nodes(df, n_n_nodes)
    legend_i = 1

    # SPARS2 does not perform well on small maps
    df = df[np.logical_or(
        df.roadmap != 'SPARS2',
        df.target_n != min_n)]

    for key, title in [
        ('visited_nodes_mean', 'Visited Vertices'),
        ('runtime_ms', 'Runtime Generation (ms)'),
        ('success_rate', 'Success Rate')
    ]:
        fig, axs = plt.subplots(
            2,
            n_plots // 2,
            figsize=(4.5*(n_plots // 2), 7))
        axs = axs.flatten()
        for i, map_name in enumerate(interesting_maps):
            ax = axs[i]
            df_map = df[df.map == map_name]
            sns.lineplot(
                data=df_map,
                x='n_nodes',
                y=key,
                hue='roadmap',
                marker='.',
                ax=ax,
                legend=(i == legend_i),
            )
            ax.set_xlabel('Number of Vertices')
            ax.set_ylabel(title)
            if key == 'runtime_ms':
                ax.set_yscale('log')
            map_name_title = _remove_numbers_after_mapnames_and_cap(map_name)
            ax.set_title(f'Map {map_name_title}')
        axs[legend_i].legend(bbox_to_anchor=(1.1, 1.05))
        fig.tight_layout()
        for extension in ['png', 'pdf']:
            plt.savefig(os.path.join(
                PLOT_FOLDER_PAPER,
                os.path.basename(CSV_PATH).replace(
                    '.csv',
                    f'_paper_{key}.{extension}'))
            )


def plot_regret():
    interesting_maps = [
        'plain',
        'den520',
        'rooms_30',
        'slam'
    ]
    n_plots = len(interesting_maps)

    df = pd.read_csv(CSV_PATH)
    # regret only over successful runs
    maps_in_data = df.map.unique()
    roadmaps_in_data = df.roadmap.unique()
    target_ns_in_data = df.target_n.unique()
    min_n = min(target_ns_in_data)
    target_ns_in_data = target_ns_in_data[
        np.logical_not(np.isnan(target_ns_in_data))]
    seeds_in_data = df.seed.unique()
    for key, title in [
        ('path_length', 'Path Length Regret'),
        ('visited_nodes', 'Visited Vertices Regret'),
    ]:
        plot_df = pd.DataFrame()
        for map_, roadmap, target_n, seed in product(
            maps_in_data, roadmaps_in_data, target_ns_in_data, seeds_in_data
        ):
            if roadmap == 'SPARS2' and target_n == min_n:
                # SPARS2 does not perform well on small maps
                continue
            mask_this_data = np.logical_and(
                df.map == map_,
                np.logical_and(
                    df.seed == seed,
                    np.logical_and(
                        df.target_n == target_n,
                        df.roadmap == roadmap)))
            if not mask_this_data.any():
                continue
            mask_gsrm = np.logical_and(
                df.map == map_,
                np.logical_and(
                    df.seed == seed,
                    np.logical_and(
                        df.target_n == target_n,
                        df.roadmap == 'GSRM')))
            if not mask_gsrm.any():
                continue

            this_i = int(df[mask_this_data].index[0])
            mask_both = np.logical_or(mask_gsrm, mask_this_data)
            success_cols = _get_cols_by_prefix(df, 'success_')
            # Filter out colums where all roadmaps where successful:
            success_cols = [
                col for col in success_cols if df[mask_both][col].all()]
            if len(success_cols) == 0:
                continue
            success_idx = [int(col.split("_")[-1]) for col in success_cols]
            # calculate regret
            plot_df.at[this_i, f'{key}_regret'] = (df[mask_this_data][
                [f'{key}_{i:03d}' for i in success_idx]
            ].mean(axis=1).values[0] - df[mask_gsrm][
                [f'{key}_{i:03d}' for i in success_idx]
            ].mean(axis=1).values[0])/df[mask_this_data][
                [f'{key}_{i:03d}' for i in success_idx]
            ].mean(axis=1).values[0]
            plot_df.at[this_i, 'map'] = map_
            plot_df.at[this_i, 'roadmap'] = roadmap
            plot_df.at[this_i, 'n_nodes'] = target_n

        legend_i = 1
        fig, axs = plt.subplots(
            2,
            n_plots // 2,
            figsize=(4.5*(n_plots // 2), 7))
        axs = axs.flatten()
        for i, map_name in enumerate(interesting_maps):
            ax = axs[i]
            df_map = plot_df[plot_df.map == map_name]
            sns.lineplot(
                data=df_map,
                x='n_nodes',
                y=f'{key}_regret',
                hue='roadmap',
                marker='.',
                ax=ax,
                legend=(i == legend_i),
            )
            ax.set_xlabel('Number of Vertices')
            ax.set_ylabel(title)
            map_name_title = _remove_numbers_after_mapnames_and_cap(map_name)
            ax.set_title(f'Map {map_name_title}')
        axs[legend_i].legend(bbox_to_anchor=(1.1, 1.05))
        fig.tight_layout()
        for extension in ['png', 'pdf']:
            plt.savefig(os.path.join(
                PLOT_FOLDER_PAPER,
                os.path.basename(CSV_PATH).replace(
                    '.csv',
                    f'_paper_{key}_regret.{extension}'))
            )

def table_for_paper():
    """Make a latex formatted table that contains how much shorter
    the paths from the different roadmaps are compared to the GSRM
    on different maps.
    """
    import latextable
    from collections import OrderedDict
    from texttable import Texttable
    df = pd.read_csv(CSV_MEAN_PATH)

    interesting_maps = [
        'plain',
        'den520',
        'rooms_30',
        'slam'
    ]
    data = OrderedDict()
    for i_m, map_name in enumerate(interesting_maps):
        data[map_name] = OrderedDict()
        mask_gsrm = np.logical_and(
            df.map == map_name,
            df.roadmap == 'GSRM'
        )
        for i_r, roadmap in enumerate(df.roadmap.unique()):
            if roadmap == 'GSRM':
                continue
            mask_rm = np.logical_and(
                df.map == map_name,
                df.roadmap == roadmap
            )
            data[map_name][roadmap] = 100 * (
                df[mask_rm].path_length_mean.mean() -
                df[mask_gsrm].path_length_mean.mean()
            ) / df[mask_gsrm].path_length_mean.mean()
    print(data)

    roadmap_names = sorted(df.roadmap.unique())
    roadmap_names.remove('GSRM')

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t'] + ['f'] * len(roadmap_names))
    table.set_cols_align(['l'] + ['r'] * len(roadmap_names))

    table.add_row(["Map"] + roadmap_names)
    for map_name, roadmap_data in data.items():
        map_name_title = _remove_numbers_after_mapnames_and_cap(map_name)
        table.add_row([map_name_title] + [
            f'\SI{{{roadmap_data[roadmap]:.1f}}}{{\percent}}'
            for roadmap in roadmap_names
        ])
    print(table.draw())
    print(latextable.draw_latex(table))


def table_number_edges():
    """Make a latex formatted table that contains the number of edges
    for different roadmaps on different maps.
    """
    import latextable
    from collections import OrderedDict
    from texttable import Texttable
    df = pd.read_csv(CSV_MEAN_PATH)

    interesting_maps = [
        'plain',
        'den520',
        'rooms_30',
        'slam'
    ]
    data = OrderedDict()
    for i_m, map_name in enumerate(interesting_maps):
        data[map_name] = OrderedDict()
        for i_r, roadmap in enumerate(df.roadmap.unique()):
            mask_rm = np.logical_and(
                df.map == map_name,
                df.roadmap == roadmap
            )
            data[map_name][roadmap] = (
                df[mask_rm].n_edges.mean(),
                df[mask_rm].n_nodes.mean()
            )

    roadmap_names = sorted(df.roadmap.unique())

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t'] + ['i'] * len(roadmap_names))
    table.set_cols_align(['l'] + ['r'] * len(roadmap_names))

    table.add_row(["Map"] + roadmap_names)
    for map_name, roadmap_data in data.items():
        map_name_title = _remove_numbers_after_mapnames_and_cap(map_name)
        row_nodes = ['Vertices ' + map_name_title] + [
            int(roadmap_data[roadmap][1]) for roadmap in roadmap_names
        ]
        row_edges = ['Edges ' + map_name_title] + [
            int(roadmap_data[roadmap][0]) for roadmap in roadmap_names
        ]
        table.add_row(row_nodes)
        table.add_row(row_edges)
    print(latextable.draw_latex(table))


if __name__ == '__main__':
    # run()
    # prepare_mean()
    # plot()
    plots_for_paper()
    plot_regret()
    table_for_paper()
    table_number_edges()
    plt.close('all')
