"""Compare spectral and NetworkX Kernighan-Lin bipartitioning on real networks."""
import json
import time
from pathlib import Path
import networkx as nx
from src.algorithms.spectral import spectral_bipartition
from src.evaluate import balance_score, cut_size, normalized_cut

DATASETS = {
    "karate_club": nx.karate_club_graph,
    "les_miserables": nx.les_miserables_graph,
    "florentine_families": nx.florentine_families_graph,
}

def evaluate(graph, partition):
    return {
        "cut_size": cut_size(graph, partition),
        "balance": balance_score(graph, partition),
        "normalized_cut": normalized_cut(graph, partition),
    }

def run():
    rows = []
    for name, loader in DATASETS.items():
        graph = loader()

        start = time.perf_counter()
        spectral = spectral_bipartition(graph)
        spectral_time = time.perf_counter() - start
        rows.append({
            "dataset": name, "nodes": len(graph), "edges": graph.number_of_edges(),
            "algorithm": "Spectral", **evaluate(graph, spectral),
            "runtime_seconds": spectral_time,
        })

        start = time.perf_counter()
        left, right = nx.algorithms.community.kernighan_lin_bisection(graph, seed=42)
        kl_time = time.perf_counter() - start
        partition = {node: 0 for node in left}
        partition.update({node: 1 for node in right})
        rows.append({
            "dataset": name, "nodes": len(graph), "edges": graph.number_of_edges(),
            "algorithm": "Kernighan-Lin", **evaluate(graph, partition),
            "runtime_seconds": kl_time,
        })

    Path("results").mkdir(exist_ok=True)
    Path("results/baseline_comparison.json").write_text(json.dumps(rows, indent=2))
    return rows

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
