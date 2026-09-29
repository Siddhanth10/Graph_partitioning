# Graph Partitioning API

This Flask API connects the static frontend to the existing graph-partitioning project.

## Run locally

From the repository root:

    pip install -r requirements.txt
    python backend/app.py

The API starts on http://localhost:5000.

## Endpoints

- GET /api/health
- POST /api/partition

Example request:

    {
      "node_count": 5,
      "algorithm": "spectral",
      "edges": [
        {"source": 0, "target": 1},
        {"source": 1, "target": 2},
        {"source": 2, "target": 3},
        {"source": 3, "target": 4},
        {"source": 0, "target": 4, "weight": 2}
      ]
    }

Algorithms: spectral, kernighan-lin, ml.
