import networkx as nx
from src.algorithms.spectral import spectral_bipartition
from src.evaluate import balance_score, cut_size

def test_spectral_returns_two_partitions():
    graph = nx.path_graph(8)
    partition = spectral_bipartition(graph)
    assert set(partition) == set(graph.nodes())
    assert set(partition.values()) == {0, 1}

def test_cut_is_positive():
    graph = nx.path_graph(8)
    assert cut_size(graph, spectral_bipartition(graph)) >= 1

def test_balance_is_bounded():
    graph = nx.path_graph(8)
    score = balance_score(graph, spectral_bipartition(graph))
    assert 0 <= score <= 1
