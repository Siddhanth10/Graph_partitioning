from __future__ import annotations
import networkx as nx
import numpy as np
from .algorithms.kernighan_lin import refine_bipartition
from .algorithms.spectral import spectral_bipartition
from .features import node_features
from .model import build_model

def generate_training_examples(num_graphs=80, nodes_per_graph=40, probability=0.12, seed=42):
    all_x, all_y = [], []
    for i in range(num_graphs):
        graph = nx.erdos_renyi_graph(nodes_per_graph, probability, seed=seed + i)
        if graph.number_of_edges() == 0:
            continue
        labels = spectral_bipartition(graph)
        _, x = node_features(graph)
        y = np.asarray([labels[node] for node in graph.nodes()], dtype=int)
        if len(np.unique(y)) < 2:
            continue
        all_x.append(x)
        all_y.append(y)
    if not all_x:
        raise RuntimeError("Could not generate training examples")
    return np.vstack(all_x), np.concatenate(all_y)

def train_model(num_graphs=80, nodes_per_graph=40, probability=0.12, seed=42):
    x, y = generate_training_examples(num_graphs, nodes_per_graph, probability, seed)
    model = build_model()
    model.fit(x, y)
    return model

def predict_partition(model, graph: nx.Graph):
    nodes, x = node_features(graph)
    if not nodes:
        return {}
    prediction = model.predict(x)
    partition = {node: int(label) for node, label in zip(nodes, prediction)}
    if len(set(partition.values())) < 2 and len(nodes) > 1:
        order = sorted(nodes, key=lambda n: x[nodes.index(n), 0])
        half = len(order) // 2
        partition = {node: int(i >= half) for i, node in enumerate(order)}
    return refine_bipartition(graph, partition)
