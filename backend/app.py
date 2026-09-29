from __future__ import annotations

from functools import lru_cache

import networkx as nx
from flask import Flask, jsonify, request
from flask_cors import CORS

from src.algorithms.kernighan_lin import refine_bipartition
from src.algorithms.spectral import spectral_bipartition
from src.evaluate import balance_score, cut_size, normalized_cut
from src.pipeline import predict_partition, train_model

app = Flask(__name__)
CORS(app)


def build_graph(node_count: int, edges: list) -> nx.Graph:
    if not isinstance(node_count, int) or not 2 <= node_count <= 500:
        raise ValueError("node_count must be an integer between 2 and 500.")

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    seen = set()

    for edge in edges:
        if not isinstance(edge, dict):
            raise ValueError("Each edge must be an object with source and target.")
        try:
            source = int(edge["source"])
            target = int(edge["target"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("Each edge needs numeric source and target values.")

        if not (0 <= source < node_count and 0 <= target < node_count):
            raise ValueError(f"Edge ({source}, {target}) uses a node outside 0..{node_count - 1}.")
        if source == target:
            raise ValueError("Self-loops are not supported.")

        weight = float(edge.get("weight", 1.0))
        if weight <= 0:
            raise ValueError("Edge weights must be greater than zero.")

        key = tuple(sorted((source, target)))
        if key in seen:
            raise ValueError(f"Duplicate edge detected: {source}-{target}.")
        seen.add(key)
        graph.add_edge(source, target, weight=weight)

    if graph.number_of_edges() == 0:
        raise ValueError("Add at least one edge.")

    return graph


@lru_cache(maxsize=1)
def get_ml_model():
    return train_model(num_graphs=40, nodes_per_graph=40, probability=0.12, seed=42)


def run_partition(graph: nx.Graph, algorithm: str) -> dict[int, int]:
    if algorithm == "spectral":
        return spectral_bipartition(graph)

    if algorithm == "kernighan-lin":
        first, second = nx.algorithms.community.kernighan_lin_bisection(
            graph, weight="weight", seed=42
        )
        return {node: (0 if node in first else 1) for node in graph.nodes()}

    if algorithm == "ml":
        return predict_partition(get_ml_model(), graph)

    raise ValueError("algorithm must be one of: spectral, kernighan-lin, ml")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "service": "graph-partitioning-api"})


@app.post("/api/partition")
def partition():
    try:
        payload = request.get_json(silent=True) or {}
        graph = build_graph(int(payload.get("node_count", 0)), payload.get("edges", []))
        algorithm = payload.get("algorithm", "spectral")

        labels = run_partition(graph, algorithm)
        partition_a = sorted([node for node, label in labels.items() if label == 0])
        partition_b = sorted([node for node, label in labels.items() if label == 1])

        return jsonify({
            "algorithm": algorithm,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "partition": {"0": partition_a, "1": partition_b},
            "metrics": {
                "cut_size": round(cut_size(graph, labels), 4),
                "balance": round(balance_score(graph, labels), 4),
                "normalized_cut": round(normalized_cut(graph, labels), 4),
            },
            "edges": [
                {
                    "source": int(source),
                    "target": int(target),
                    "weight": float(data.get("weight", 1.0)),
                    "cut": labels[source] != labels[target],
                }
                for source, target, data in graph.edges(data=True)
            ],
        })
    except (ValueError, TypeError, nx.NetworkXException) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Unexpected partitioning error")
        return jsonify({"error": "Unexpected server error."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
