"""Call an OpenAI-compatible model for simulated student responses and concept judgments."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


# Func: Load optional mastered or unmastered response examples.
# Input: Optional path to a JSON list of example records.
# Output: Example records or an empty list.
def load_examples(path: Path | None) -> list[Any]:
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Student example file does not exist: {path}")
    examples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(examples, list):
        raise ValueError("Student examples must be a JSON list")
    return examples


# Func: Format one response exemplar without modifying its content.
# Input: String or dictionary example record and its display number.
# Output: Readable example text.
def format_example(example: Any, index: int) -> str:
    if isinstance(example, str):
        return f"Example {index}\n{example.strip()}"
    if not isinstance(example, dict):
        raise ValueError("Student example must be a string or object")
    lines = [f"Example {index}"]
    question = str(example.get("qstn", example.get("question", ""))).strip()
    if question:
        lines.append(f"Question: {question}")
    options = [str(option).strip() for option in example.get("options", []) if str(option).strip()]
    if options:
        lines.extend(["Options:", *options])
    answer = str(example.get("answer_letter", "")).strip().upper()
    if answer:
        lines.append(f"Answer: {answer}")
    used = [str(term).strip() for term in example.get("used_terms", []) if str(term).strip()]
    if used:
        lines.extend(["Concepts used:", *(f"- {term}" for term in used)])
    reasoning = str(example.get("reasoning", "")).strip()
    if reasoning:
        lines.append(f"Reasoning: {reasoning}")
    return "\n".join(lines)


# Func: Present one MCQ as a stem followed by its four choices.
# Input: Normalized assessment question record.
# Output: Question text for student and judge prompts.
def format_question(question: dict) -> str:
    return question["question"] + "\n\n" + "\n".join(question["options"])


class LLMResponder:
    # Func: Initialize one OpenAI-compatible client and model settings.
    # Input: API key and configured model settings.
    # Output: A responder ready for student and judge calls.
    def __init__(self, api_key: str, model: dict[str, Any]):
        from openai import OpenAI

        options = {"api_key": api_key}
        if model.get("base_url"):
            options["base_url"] = str(model["base_url"])
        self.client = OpenAI(**options)
        self.model = model

    # Func: Submit one chat request using configured model and sampling settings.
    # Input: System and user prompt text, temperature, and token budget.
    # Output: Nonempty model response text.
    def complete(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=str(self.model["id"]),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content
        if not text:
            raise ValueError("Model returned an empty response")
        return text

    # Func: Simulate one mastered or unmastered student's answer and reasoning.
    # Input: Profile, question, cluster concepts, exemplars, prompt config, topic, and RNG.
    # Output: Raw simulated student text.
    def respond(
        self, profile: str, question: dict, nodes: list[str], examples: list[Any],
        prompts: dict, topic: str, count: int, rng: random.Random,
    ) -> str:
        selected = rng.sample(examples, min(count, len(examples))) if examples else []
        prompt = prompts[profile]
        return self.complete(
            str(prompt["system"]).format(topic=topic),
            str(prompt["user"]).format(
                examples="\n\n".join(format_example(example, index) for index, example in enumerate(selected, 1)),
                nodes="\n".join(f"- {node}" for node in nodes),
                question=format_question(question),
            ),
            float(self.model.get("student_temperature", self.model["temperature"])),
            int(self.model.get("student_max_tokens", self.model["max_tokens"])),
        )

    # Func: Judge concept-level understanding from the simulated response.
    # Input: Question, answer, reasoning, cluster concepts, scoring prompt, and topic.
    # Output: Raw concept-classification text.
    def judge(self, question: dict, answer: str, reasoning: str, nodes: list[str], scoring: dict, topic: str) -> str:
        return self.complete(
            str(scoring["judge_system"]).format(topic=topic),
            str(scoring["judge_user"]).format(
                question=format_question(question), answer=answer, reasoning=reasoning,
                nodes="\n".join(f"- {node}" for node in nodes),
            ),
            float(self.model.get("judge_temperature", self.model["temperature"])),
            int(self.model.get("judge_max_tokens", self.model["max_tokens"])),
        )
