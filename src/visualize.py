from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


def save_partition_plot(graph, partition, output_path="outputs/partition.png", title="Graph Partition"):
    """Save a graph image with nodes grouped by predicted partition."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx_edges(graph, pos, alpha=0.35)
    for label in (0, 1):
        nodes = [n for n in graph.nodes() if partition[n] == label]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_size=80,
            label=f"Partition {label}",
        )
    plt.title(title)
    plt.axis("off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()
    return output
