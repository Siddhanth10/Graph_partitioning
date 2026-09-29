# Experimental Results

## 1. Synthetic benchmark

Configuration: 100 training graphs + 30 held-out test graphs; 40 nodes per graph; Erdős–Rényi edge probability 0.12; random seed 42. Spectral partitions were used as supervised labels.

| Method | Mean cut size | Mean balance | Mean normalized cut |
|---|---:|---:|---:|
| Spectral | 28.23 | 0.965 | 0.618 |
| Random Forest + balance-constrained refinement | 33.70 | 0.857 | 0.775 |
| GCN + balance-constrained refinement | 26.30 | 0.902 | 0.629 |

Balance = 1.0 means an exactly even two-way split. The constrained Random Forest no longer obtains an artificially low cut by collapsing one partition. The GCN produces a lower mean cut than the spectral baseline while maintaining a balance above 0.90 on this particular benchmark, although its normalized cut remains slightly higher than spectral.

## 2. GNN hyperparameter experiment

| Hidden dimension | Epochs | Mean normalized cut |
|---:|---:|---:|
| 16 | 20 | 0.5680 |
| 32 | 20 | 0.6085 |
| 32 | 40 | 0.6071 |
| 64 | 40 | 0.5833 |

These are exploratory runs, not statistically significant model-selection results. Multiple seeds and a larger graph benchmark are needed for a stronger study.

## 3. Real-world network examples

The project was evaluated on NetworkX's built-in Karate Club, Les Misérables, and Florentine Families graph examples.

| Dataset | Nodes | Edges | Cut size | Balance | Normalized cut |
|---|---:|---:|---:|---:|---:|
| Karate Club | 34 | 78 | 22 | 0.941 | 0.191 |
| Les Misérables | 77 | 254 | 52 | 0.753 | 0.177 |
| Florentine Families | 15 | 20 | 4 | 0.933 | 0.404 |

## 4. Stronger classical baseline

| Dataset | Algorithm | Cut size | Balance | Normalized cut |
|---|---|---:|---:|---:|
| Karate Club | Spectral | 25 | 1.000 | 0.217 |
| Karate Club | Kernighan–Lin | 23 | 1.000 | 0.199 |
| Les Misérables | Spectral | 119 | 0.987 | 0.318 |
| Les Misérables | Kernighan–Lin | 105 | 0.987 | 0.265 |
| Florentine Families | Spectral | 5 | 0.933 | 0.501 |
| Florentine Families | Kernighan–Lin | 5 | 0.933 | 0.501 |

## 5. Conclusions

The experiments validate the end-to-end pipeline and expose an important optimization trade-off: minimizing crossing edges without enforcing balance can produce misleadingly low cut values. The project therefore reports cut size, balance, and normalized cut together.

On the synthetic benchmark, the balance-constrained GCN achieved a mean cut of 26.30 versus 28.23 for spectral partitioning, while maintaining mean balance of 0.902. This is an encouraging prototype result, not evidence of general superiority: the benchmark is small and the GNN is trained using spectral-generated labels.

The current GNN learns to approximate a classical heuristic rather than a globally optimal partition. A stronger research study should use multiple random seeds, larger graph families, and optimization-generated targets.
