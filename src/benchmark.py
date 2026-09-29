from __future__ import annotations

import time

from sklearn.metrics import accuracy_score

from .algorithms.kernighan_lin import refine_bipartition
from .algorithms.spectral import spectral_bipartition
from .evaluate import balance_score, cut_size, normalized_cut
from .gnn import predict_gnn


def evaluate_partition(graph, partition):
    return {
        "cut_size": cut_size(graph, partition),
        "balance": balance_score(graph, partition),
        "normalized_cut": normalized_cut(graph, partition),
    }


def benchmark_gnn(model, test_graphs):
    rows = []
    for sample in test_graphs:
        graph = sample["graph"]

        start = time.perf_counter()
        predicted = predict_gnn(model, sample)
        predicted = refine_bipartition(graph, predicted)
        runtime = time.perf_counter() - start

        metrics = evaluate_partition(graph, predicted)
        target = sample["labels"]
        prediction_array = [predicted[node] for node in graph.nodes()]

        rows.append(
            {
                "accuracy_vs_spectral_labels": accuracy_score(target, prediction_array),
                "runtime_seconds": runtime,
                **metrics,
            }
        )
    return rows


def benchmark_spectral(test_graphs):
    rows = []
    for sample in test_graphs:
        graph = sample["graph"]
        start = time.perf_counter()
        partition = spectral_bipartition(graph)
        runtime = time.perf_counter() - start
        rows.append({"runtime_seconds": time.perf_counter() - start, **evaluate_partition(graph, partition)})
    return rows
