import logging
import tracemalloc

import networkx as nx
import numpy as np
from humanfriendly import format_size
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)


N_LEARN_MAX = 1000
RADIUS = 0.001


class DaggerStrategy():
    """Implementation of DAgger
    (https://proceedings.mlr.press/v15/ross11a.html)"""

    def __init__(self, model, graph, n_episodes, n_agents,
                 batch_size, optimizer, prefix, rng):
        self.model = model
        self.graph = self._add_self_edges_to_graph(graph)
        self.n_episodes = n_episodes
        self.n_agents = n_agents
        self.batch_size = batch_size
        self.rng = rng
        self.optimizer = optimizer
        self.prefix = prefix

    def _add_self_edges_to_graph(self, graph: nx.Graph) -> nx.Graph:
        """Add self edges to the graph."""
        for node in graph.nodes():
            if not graph.has_edge(node, node):
                graph.add_edge(node, node)
        return graph

    def learn_dagger(self, epds):
        """Run the DAgger algorithm."""
        loss_s = []

        logger.debug("Memory usage, current: " +
                     str(format_size(tracemalloc.get_traced_memory()[0])) +
                     " peak: " +
                     str(format_size(tracemalloc.get_traced_memory()[1])))

        # learn
        loader = DataLoader(epds, batch_size=self.batch_size, shuffle=True)
        loss_s = []
        for _, batch in enumerate(loader):
            loss = self.model.learn(batch, self.optimizer)
            loss_s.append(loss)

        if len(loss_s) == 0:
            loss_s = [0]
        return self.model, np.mean(loss_s)
