# Graph Partitioning for Network Optimization

Machine learning project for graph partitioning and network optimization.

## Goal
Divide a graph into balanced partitions while minimizing communication/cut edges.

## Planned pipeline
1. Generate/load graphs
2. Extract graph features
3. Implement classical partitioning baselines
4. Train an ML model to predict node partitions
5. Refine predictions with local search
6. Compare cut size, balance, normalized cut, and runtime

## Initial algorithms
- Spectral partitioning
- Kernighan-Lin style local refinement
- Machine-learning baseline using Random Forest

## Run
```bash
pip install -r requirements.txt
python -m src.cli --nodes 40 --probability 0.12
```
