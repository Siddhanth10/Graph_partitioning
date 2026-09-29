from __future__ import annotations

import networkx as nx
import numpy as np

from .algorithms.spectral import spectral_bipartition
from .features import node_features


def generate_graph_dataset(
    num_graphs: int = 100,
    nodes_per_graph: int = 40,
    probability: float = 0.12,
    seed: int = 42,
):
    """Generate graphs with spectral partitions used as supervised targets."""
    dataset = []
    for i in range(num_graphs):
        graph = nx.erdos_renyi_graph(
            nodes_per_graph, probability, seed=seed + i
        )
        if graph.number_of_edges() == 0:
            continue

        labels = spectral_bipartition(graph)
        nodes, features = node_features(graph)
        targets = np.asarray([labels[node] for node in nodes], dtype=np.int64)

        if len(np.unique(targets)) < 2:
            continue

        dataset.append(
            {
                "graph": graph,
                "features": features.astype(np.float32),
                "labels": targets,
            }
        )
    if not dataset:
        raise RuntimeError("No valid graphs were generated.")
    return dataset
