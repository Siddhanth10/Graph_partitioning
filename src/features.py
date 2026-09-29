from __future__ import annotations
import networkx as nx
import numpy as np

FEATURE_NAMES = ["degree", "weighted_degree", "clustering", "pagerank", "core_number", "average_neighbor_degree"]

def node_features(graph: nx.Graph) -> tuple[list[int], np.ndarray]:
    nodes = list(graph.nodes())
    if not nodes:
        return [], np.empty((0, len(FEATURE_NAMES)))
    degree = dict(graph.degree())
    weighted_degree = dict(graph.degree(weight="weight"))
    clustering = nx.clustering(graph, weight="weight")
    pagerank = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {n: 1 / len(nodes) for n in nodes}
    core = nx.core_number(graph) if graph.number_of_edges() else {n: 0 for n in nodes}
    avg_neighbor_degree = nx.average_neighbor_degree(graph, weight="weight")
    matrix = [[degree[n], weighted_degree[n], clustering[n], pagerank[n], core[n], avg_neighbor_degree[n]] for n in nodes]
    return nodes, np.asarray(matrix, dtype=float)
