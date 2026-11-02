# Graph Construction

This package converts educational text into a cleaned knowledge graph, a concept line graph, and aligned semantic features for downstream clustering.

## Setup

```bash
cd /scratch/Fabrice/sage_dev/Final/GraphConstruction
python -m pip install -e .
```

Configure the source text and destination in `configurations/main.yaml`, set the API key named by `kggen.api_key_env`, then run:

```bash
python main.py
```

Hydra overrides can be supplied without editing the configuration:

```bash
python main.py paths.text=/path/to/biology.txt kggen.context="Biology textbook"
```

The `paths.text` and `paths.destination` settings accept absolute paths or paths relative to `GraphConstruction`. Outputs are organized in a subdirectory named after the input file.

`main.py` runs KGGen and coordinates the pipeline. `clean.py` cleans the extracted graph and constructs the concept line graph. `embed.py` encodes line-graph nodes with the model selected by `embedding.model` and writes the feature artifacts.

To regenerate embeddings and the clustering matrix from an existing `line_graph.json`, change `embedding.model` in the configuration and run `python embed.py`. This step reads the saved graph and does not call KGGen.

Person-name filtering is disabled by default. Set `cleaning.remove_person_entities` to `true` in `configurations/main.yaml` to enable it. First install the optional dependency and the spaCy model selected by `cleaning.spacy_model`:

```bash
python -m pip install -e '.[person-filtering]'
python -m spacy download en_core_web_trf
```

Only entities whose complete text is identified as a person name are removed. For a single run, the setting can also be overridden with `python main.py cleaning.remove_person_entities=true`.

## Outputs

- `cleaned_graph.json`: normalized and deduplicated entity graph.
- `line_graph.json`: concept triplets represented as nodes and connected by the configured line-graph rule.
- `node_ids.json`: row order shared by all feature and adjacency files.
- `embeddings.npy`: float32 semantic vectors in `node_ids.json` order.
- `line_graph.mat`: the same vectors in `fea` and aligned adjacency in `W`, as required for clustering. `gnd` contains placeholder labels because source text has no ground truth cluster labels.
