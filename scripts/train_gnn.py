from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset import generate_graph_dataset
from src.gnn import train_gnn
from src.benchmark import benchmark_gnn, benchmark_spectral


def main():
    parser = argparse.ArgumentParser(description="Train the graph partitioning GNN")
    parser.add_argument("--graphs", type=int, default=100)
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--probability", type=float, default=0.12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    dataset = generate_graph_dataset(args.graphs, args.nodes, args.probability)
    split = int(len(dataset) * (1 - args.test_size))
    train_graphs, test_graphs = dataset[:split], dataset[split:]

    model = train_gnn(train_graphs, epochs=args.epochs)
    gnn_results = benchmark_gnn(model, test_graphs)
    spectral_results = benchmark_spectral(test_graphs)

    summary = {
        "graphs": len(dataset),
        "train_graphs": len(train_graphs),
        "test_graphs": len(test_graphs),
        "gnn": gnn_results,
        "spectral": spectral_results,
    }

    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/benchmark.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved benchmark to outputs/benchmark.json")
    print(f"GNN test graphs: {len(gnn_results)}")


if __name__ == "__main__":
    main()
