from __future__ import annotations
import argparse
import networkx as nx
from .algorithms.spectral import spectral_bipartition
from .evaluate import balance_score, cut_size, normalized_cut
from .pipeline import predict_partition, train_model

def main():
    parser = argparse.ArgumentParser(description="Graph partitioning ML demo")
    parser.add_argument("--nodes", type=int, default=40)
    parser.add_argument("--probability", type=float, default=0.12)
    args = parser.parse_args()
    graph = nx.erdos_renyi_graph(args.nodes, args.probability, seed=7)
    model = train_model(nodes_per_graph=args.nodes, probability=args.probability)
    ml = predict_partition(model, graph)
    spectral = spectral_bipartition(graph)
    print("Graph Partitioning for Network Optimization")
    print(f"Nodes: {graph.number_of_nodes()} | Edges: {graph.number_of_edges()}")
    for name, partition in [("ML + refinement", ml), ("Spectral baseline", spectral)]:
        print(f"\\n{name}")
        print(f"  Cut size: {cut_size(graph, partition):.2f}")
        print(f"  Balance: {balance_score(graph, partition):.3f}")
        print(f"  Normalized cut: {normalized_cut(graph, partition):.3f}")

if __name__ == "__main__":
    main()
