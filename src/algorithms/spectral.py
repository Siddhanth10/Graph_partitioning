from __future__ import annotations
import numpy as np
import networkx as nx

def spectral_bipartition(graph: nx.Graph) -> dict[int, int]:
    """Partition a graph into two groups using the Fiedler vector."""
    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: 0}

    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=float)
    degree = np.sum(adjacency, axis=1)
    laplacian = np.diag(degree) - adjacency
    _, eigenvectors = np.linalg.eigh(laplacian)
    fiedler = eigenvectors[:, 1]
    labels = (fiedler > np.median(fiedler)).astype(int)

    if labels.min() == labels.max():
        order = np.argsort(fiedler)
        labels[order[:n // 2]] = 0
        labels[order[n // 2:]] = 1
    return {node: int(label) for node, label in zip(nodes, labels)}
