"""Score questions with local models using normalized 0–5 next-token probabilities."""

from __future__ import annotations

from typing import Any


# Func: Load one configured judge model and tokenizer onto available devices.
# Input: Hugging Face model path, numerical dtype, and device placement.
# Output: Model, tokenizer, and model input device.
def load_judge(model_path: str, dtype_name: str, device_map: str) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_name, None)
    if dtype is None or not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown PyTorch dtype: {dtype_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map=device_map)
    model.eval()
    device = model.get_input_embeddings().weight.device
    return model, tokenizer, device


# Func: Validate that every score is one tokenizer token before probing.
# Input: Tokenizer and configured score token strings.
# Output: One vocabulary token ID per score.
def score_token_ids(tokenizer: Any, score_tokens: list[str]) -> list[int]:
    if len(score_tokens) < 2 or len(set(score_tokens)) != len(score_tokens):
        raise ValueError("At least two distinct score tokens are required")
    ids = [tokenizer.encode(token, add_special_tokens=False) for token in score_tokens]
    if any(len(item) != 1 for item in ids):
        raise ValueError("Judge score tokens must each tokenize to exactly one token")
    return [item[0] for item in ids]


# Func: Build one rubric prompt without relying on a teacher answer key.
# Input: Question record, rubric dimension, and judge system instructions.
# Output: System and user messages for scoring.
def judge_messages(question: dict[str, Any], template: str, system: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": template.format(
            question=question["question"],
            options="\n".join(question["options"]),
        )},
    ]


# Func: Probe the next-token distribution over rubric score choices.
# Input: Loaded judge, tokenizer, messages, token IDs, and model template settings.
# Output: Expected score under the normalized score-token probabilities.
def expected_score(
    model: Any,
    tokenizer: Any,
    device: Any,
    messages: list[dict[str, str]],
    token_ids: list[int],
    score_values: list[float],
    add_generation_prompt: bool,
    assistant_suffix: str,
) -> float:
    import torch

    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
    inputs = tokenizer(prompt + assistant_suffix, return_tensors="pt", add_special_tokens=False)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[0, -1, :].float()
    probabilities = torch.softmax(logits[token_ids], dim=0)
    values = torch.tensor(score_values, device=probabilities.device, dtype=probabilities.dtype)
    return float(torch.dot(probabilities, values).item())


# Func: Score one question on every configured evaluation dimension.
# Input: Loaded judge objects, model settings, score tokens, rubric, and question.
# Output: Expected score for each rubric dimension.
def score_question(
    model: Any,
    tokenizer: Any,
    device: Any,
    model_cfg: dict[str, Any],
    evaluation: dict[str, Any],
    question: dict[str, Any],
) -> dict[str, float]:
    tokens = [str(token) for token in evaluation["score_tokens"]]
    token_ids = score_token_ids(tokenizer, tokens)
    values = [float(token) for token in tokens]
    results: dict[str, float] = {}
    for dimension, template in evaluation["dimensions"].items():
        messages = judge_messages(question, str(template), str(evaluation["system"]))
        results[str(dimension)] = expected_score(
            model, tokenizer, device, messages, token_ids, values,
            bool(model_cfg["add_generation_prompt"]), str(model_cfg["assistant_suffix"]),
        )
    return results
