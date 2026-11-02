# Question Generation

This package turns line-graph clusters into multiple-choice questions. In addition, it optionally compare the generated questions with teacher questions using three local judge models and Earth Mover distance.

## Generate questions

```bash
cd /scratch/Fabrice/sage_dev/Final/QuestionGeneration
python -m pip install -e .
export OPENAI_API_KEY=...
python main.py
```

Set the cluster file, teacher examples, output file, model, question count, and generation prompt in `configurations/main.yaml`. Paths may be absolute or relative to the `QuestionGeneration` directory. The default cluster path matches the Ego output from `GraphClustering`; the other default paths use the same `../Data/input/` dataset folder. `model.id` is the model identifier accepted by the OpenAI-compatible API, e.g `gpt-4o-mini`. Set `model.base_url` for another compatible endpoint.

`generation.questions_per_cluster` defaults to 10. The generator prioritizes knowledge concepts used least often in earlier questions, limits the cluster concepts shown in one prompt with `generation.max_terms_per_prompt`, validates that reported `used_terms` were shown to the model, and saves after each question. Running `main.py` again resumes an incomplete output. If the cluster concepts, prompt, or model settings change, choose a new output path. Increasing only the question count can continue an existing run.

For another dataset, override all three paths together:

```bash
python main.py paths.clusters=../Data/biology/clusters/ego/clusters.json paths.few_shots=../Data/biology/few_shot_qstn.json paths.output=../Data/biology/questions/ego/questions.json generation.topic=Biology
```

The output is a JSON list of cluster records. Each question has a stem, four labeled choices, the correct answer letter, and the concept statements the model says it used. The generator checks structure and term labels; it does not independently verify the designated correct answer.

## Optional evaluation

Question generation does not start evaluation or load judge models. When the question set is ready, configure `configurations/evaluations.yaml` and run:

```bash
python -m pip install -e '.[evaluation]'
python evaluate.py
```

`evaluations.yaml` contains paths to generated and teacher questions, three judge model paths, the rubric prompts, and output paths. Models load one at a time on the devices visible to the process. The evaluation saves scores after each question and resumes only when the questions and judge configuration match the saved results.

For each model and rubric dimension, `earth_mover.json` reports the exact one-dimensional Wasserstein-1 distance between teacher and generated score distributions. A normalized similarity of `1 - distance / 5` is also reported. Different question counts are handled directly without random subsampling. The report records answer-choice counts so differences between teacher and generated question formats remain visible.
