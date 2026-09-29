from __future__ import annotations

import random

import networkx as nx
import numpy as np
import torch
from torch import nn


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.linear(adjacency @ x)


class GCNPartitioner(nn.Module):
    """Small two-layer GCN for binary node partition prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = GraphConvolution(input_dim, hidden_dim)
        self.conv2 = GraphConvolution(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x, adjacency))
        x = torch.dropout(x, p=0.15, train=self.training)
        x = torch.relu(self.conv2(x, adjacency))
        return self.classifier(x)


def normalized_adjacency(graph: nx.Graph) -> torch.Tensor:
    adjacency = nx.to_numpy_array(graph, dtype=np.float32)
    adjacency += np.eye(len(adjacency), dtype=np.float32)
    degree = adjacency.sum(axis=1)
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
    normalized = inv_sqrt @ adjacency @ inv_sqrt
    return torch.tensor(normalized, dtype=torch.float32)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_gnn(
    train_graphs,
    epochs: int = 60,
    learning_rate: float = 0.005,
    hidden_dim: int = 32,
    seed: int = 42,
):
    set_seed(seed)
    input_dim = train_graphs[0]["features"].shape[1]
    model = GCNPartitioner(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        random.shuffle(train_graphs)
        for sample in train_graphs:
            x = torch.tensor(sample["features"], dtype=torch.float32)
            y = torch.tensor(sample["labels"], dtype=torch.long)
            adjacency = normalized_adjacency(sample["graph"])

            optimizer.zero_grad()
            logits = model(x, adjacency)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def predict_gnn(model, graph_sample):
    model.eval()
    x = torch.tensor(graph_sample["features"], dtype=torch.float32)
    adjacency = normalized_adjacency(graph_sample["graph"])
    logits = model(x, adjacency)
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

    return {
        node: int(label)
        for node, label in zip(graph_sample["graph"].nodes(), predictions)
    }
