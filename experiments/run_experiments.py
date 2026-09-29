"""Reproduce the synthetic, real-network and GNN experiments.

Run from the repository root:
    python experiments/run_experiments.py
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.algorithms.spectral import spectral_bipartition
from src.evaluate import balance_score, cut_size, normalized_cut
from src.features import node_features
from src.gnn import predict_gnn, train_gnn
from experiments.compare_baselines import run as run_baselines


def generate(num_graphs=100, nodes_per_graph=40, probability=0.12, seed=42):
    dataset = []
    for i in range(num_graphs):
        graph = nx.erdos_renyi_graph(nodes_per_graph, probability, seed=seed + i)
        if graph.number_of_edges() == 0:
            continue
        labels = spectral_bipartition(graph)
        nodes, features = node_features(graph)
        targets = np.asarray([labels[node] for node in nodes], dtype=np.int64)
        if len(np.unique(targets)) == 2:
            dataset.append({"graph": graph, "features": features, "labels": targets})
    return dataset


def metrics(graph, partition):
    return {
        "cut_size": cut_size(graph, partition),
        "balance": balance_score(graph, partition),
        "normalized_cut": normalized_cut(graph, partition),
    }


def balanced_refinement(graph, partition, minimum_balance=0.90):
    """Greedy refinement that never drops below the requested balance."""
    partition = dict(partition)
    for _ in range(10):
        current_cut = cut_size(graph, partition)
        best = None

        for node in graph:
            old_label = partition[node]
            partition[node] = 1 - old_label
            gain = current_cut - cut_size(graph, partition)
            candidate_balance = balance_score(graph, partition)
            partition[node] = old_label

            if (
                gain > 0
                and candidate_balance >= minimum_balance
                and (best is None or gain > best[0])
            ):
                best = (gain, node)

        if best is None:
            break
        partition[best[1]] = 1 - partition[best[1]]

    return partition


def main():
    train = generate(100)
    test = generate(30, seed=1000)

    X_train = np.vstack([sample["features"] for sample in train])
    y_train = np.concatenate([sample["labels"] for sample in train])

    rf = RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced", n_jobs=-1
    ).fit(X_train, y_train)

    rf_rows = []
    spectral_rows = []
    for sample in test:
        graph = sample["graph"]
        nodes = list(graph)
        prediction = {
            node: int(label)
            for node, label in zip(nodes, rf.predict(sample["features"]))
        }
        prediction = balanced_refinement(graph, prediction)

        rf_rows.append({
            **metrics(graph, prediction),
            "accuracy_vs_spectral_labels": accuracy_score(
                sample["labels"], [prediction[node] for node in nodes]
            ),
        })
        spectral_rows.append(metrics(graph, spectral_bipartition(graph)))

    gnn = train_gnn(train[:80], epochs=40, hidden_dim=32)
    gnn_rows = []
    for sample in test:
        prediction = predict_gnn(gnn, sample)
        prediction = balanced_refinement(sample["graph"], prediction)
        gnn_rows.append(metrics(sample["graph"], prediction))

    hyperparameters = []
    for hidden_dim, epochs in [(16, 20), (32, 20), (32, 40), (64, 40)]:
        model = train_gnn(train[:80], epochs=epochs, hidden_dim=hidden_dim)
        scores = []
        for sample in test[:15]:
            prediction = predict_gnn(model, sample)
            prediction = balanced_refinement(sample["graph"], prediction)
            scores.append(normalized_cut(sample["graph"], prediction))
        hyperparameters.append({
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "mean_normalized_cut": float(np.mean(scores)),
        })

    real_world = []
    for name, loader in {
        "karate_club": nx.karate_club_graph,
        "les_miserables": nx.les_miserables_graph,
        "florentine_families": nx.florentine_families_graph,
    }.items():
        graph = loader()
        prediction = balanced_refinement(graph, spectral_bipartition(graph))
        real_world.append({
            "dataset": name,
            "nodes": len(graph),
            "edges": graph.number_of_edges(),
            **metrics(graph, prediction),
        })

    mean = lambda rows: {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }

    results = {
        "synthetic": {
            "train_graphs": len(train),
            "test_graphs": len(test),
            "spectral_mean": mean(spectral_rows),
            "random_forest_mean": mean(rf_rows),
            "gnn_mean": mean(gnn_rows),
        },
        "gnn_hyperparameters": hyperparameters,
        "real_world": real_world,
    }

    Path("results").mkdir(exist_ok=True)
    Path("results/experiment_results.json").write_text(
        json.dumps(results, indent=2)
    )
    run_baselines()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
