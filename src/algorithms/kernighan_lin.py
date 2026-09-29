from __future__ import annotations
import networkx as nx

def cut_size(graph: nx.Graph, partition: dict[int, int]) -> float:
    return sum(float(data.get("weight", 1.0)) for u, v, data in graph.edges(data=True) if partition[u] != partition[v])

def refine_bipartition(graph: nx.Graph, partition: dict[int, int], max_passes: int = 10) -> dict[int, int]:
    """Greedy local-search refinement for a two-way partition."""
    labels = dict(partition)
    nodes = list(graph.nodes())
    for _ in range(max_passes):
        current = cut_size(graph, labels)
        counts = [sum(labels[n] == p for n in nodes) for p in (0, 1)]
        best_gain, best_node, best_label = 0.0, None, None
        for node in nodes:
            old = labels[node]
            if counts[old] <= 1:
                continue
            new = 1 - old
            labels[node] = new
            candidate = cut_size(graph, labels)
            labels[node] = old
            gain = current - candidate
            if gain > best_gain:
                best_gain, best_node, best_label = gain, node, new
        if best_node is None:
            break
        labels[best_node] = best_label
    return labels
