# Student Simulation

This package evaluates how well the assessment pipeline recovers a simulated student's hidden mastery. It reads the graph from `GraphConstruction`, Ego clusters from `GraphClustering`, and cluster questions from `QuestionGeneration`.

## Configure and run

```bash
cd /scratch/Fabrice/sage_dev/Final/StudentSimulation
python -m pip install -e .
export OPENAI_API_KEY=...
python main.py
```

Set the input and output paths, model, student profile, random seed, and `simulation.max_questions` in `configurations/main.yaml`. Mastered and unmastered response examples are optional. The defaults are `propagation=heat_kernel` and `QuestionSelection=boundary_submodular_eig`.

Select other methods through Hydra:

```bash
python main.py propagation=ppnrp QuestionSelection=frontier_submodular_eig simulation.max_questions=50
```

## Question Selection

Question selection decides which question to ask next. We provide four adaptive methods—`submodular_eig`, `boundary_submodular_eig`, `cutmass_submodular_eig`, and `frontier_submodular_eig`—plus a `random` baseline. Their settings are in `configurations/QuestionSelection/`, and their implementations are in `source/questionselection/`.

To add an adaptive method, implement `select(data, beliefs, graph, asked, covered, config, rng)` in `source/questionselection/<name>.py`. It should return the selected cluster ID, question index, tested concept nodes, and reached graph nodes, or `None` if no question remains. Add `configurations/QuestionSelection/<name>.yaml` and register the method in `main.py`'s `select_question` function.

## Propagation

Propagation spreads the signal from observed responses beyond the queried nodes. We provide `ppnrp`, `heat_kernel`, `chebyshev`, and `distance_weighted`; the last is the no-diffusion baseline. Their settings are in `configurations/propagation/`, and their implementations are in `source/propagation/`.

To add a propagation method, implement `build_operator(...)` for the graph and method parameters, then `propagate(operator, values)` to return a signal aligned with the graph's node order. Add `configurations/propagation/<name>.yaml` and register the method in `main.py`'s `build_operator` and `propagate_signal` functions.

## Assessment

`initialization.py` loads and checks graph, cluster, and question alignment.

For each selected question, `main.py` obtains an answer and reasoning trace, asks the judge to classify cluster concepts, combines answer correctness and reasoning signs, updates Beta beliefs directly, and applies one propagation operator.

The run stops at `simulation.max_questions` or when no valid unasked questions remain.
