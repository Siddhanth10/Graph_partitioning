import networkx as nx
import torch

from src.dataset import generate_graph_dataset
from src.gnn import GCNPartitioner, normalized_adjacency


def test_normalized_adjacency_shape():
    graph = nx.path_graph(6)
    adjacency = normalized_adjacency(graph)
    assert adjacency.shape == (6, 6)
    assert torch.isfinite(adjacency).all()


def test_gnn_forward_shape():
    dataset = generate_graph_dataset(num_graphs=2, nodes_per_graph=8, probability=0.3)
    sample = dataset[0]
    model = GCNPartitioner(sample["features"].shape[1])
    x = torch.tensor(sample["features"], dtype=torch.float32)
    logits = model(x, normalized_adjacency(sample["graph"]))
    assert logits.shape == (8, 2)
