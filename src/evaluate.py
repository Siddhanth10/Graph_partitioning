from __future__ import annotations
import networkx as nx

def cut_size(graph: nx.Graph, partition: dict[int, int]) -> float:
    return sum(float(d.get("weight", 1.0)) for u, v, d in graph.edges(data=True) if partition[u] != partition[v])

def balance_score(graph: nx.Graph, partition: dict[int, int]) -> float:
    n = graph.number_of_nodes()
    if n == 0:
        return 1.0
    sizes = [sum(partition[node] == label for node in graph.nodes()) for label in (0, 1)]
    return min(sizes) / (n / 2)

def normalized_cut(graph: nx.Graph, partition: dict[int, int]) -> float:
    if graph.number_of_nodes() == 0:
        return 0.0
    cut = cut_size(graph, partition)
    volumes = [0.0, 0.0]
    for node in graph.nodes():
        volumes[partition[node]] += sum(float(d.get("weight", 1.0)) for _, _, d in graph.edges(node, data=True))
    return sum(cut / volume for volume in volumes if volume > 0)
