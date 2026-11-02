# Graph Clustering

This package clusters concept nodes in a line graph using their aligned semantic embeddings and graph adjacency.

## Setup

```bash
cd /scratch/Fabrice/sage_dev/Final/GraphClustering
python -m pip install -e .
```

Configure `paths.matrix`, `paths.graph`, `paths.node_ids`, `paths.output`, and `clustering.n_clusters` in `configurations/main.yaml`. The default method is **Ego**. Its parameters are in `configurations/method/ego.yaml`; each other method has its own YAML file.

```bash
python main.py
```

Select another method with a Hydra override:

```bash
python main.py method=hard clustering.n_clusters=8
```

By default, we provide implementations of `hard`, `ego`, `ego_mod`, and `gcc_soft`. Their formulations are detailed in the paper.

To add a new method, create `source/<name>.py` with a `cluster(...)` function and add its parameters in `configurations/method/<name>.yaml`. Then import it, add its name to the supported methods, and call it in `main.py`. The runner first calls `hard.cluster(...)`; an overlapping method then returns `(memberships, selected_parameter, achieved_rate)`. `memberships` maps each node's row index to a list of cluster IDs, with at least one ID for every node.

The runner uses `fea` and `W` from `line_graph.mat` and validates their alignment with `node_ids.json` and `line_graph.json`. It does not use `gnd`, because GraphConstruction writes placeholder ground-truth labels. Results are saved as `clusters.json` and `metrics.json` under `<paths.output>/<method>/`.
