from __future__ import annotations

import json
import random
import re
import time
from typing import Optional

from openai import OpenAI


class LLM:
    def __init__(self, model: str = "gpt-5.2"):
        self.client = OpenAI()
        self.model = model

    def prompt(self, text: str, retries: int = 4) -> str:
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": text}],
                    store=False,
                )
                return response.choices[0].message.content or ""
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""


_llm: Optional[LLM] = None
_MODEL = "gpt-5.2"


def configure(model: str) -> None:
    global _MODEL, _llm
    _MODEL = model
    _llm = None


def _get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM(_MODEL)
    return _llm


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


_LETTERS = ["A", "B", "C", "D", "E", "F"]


def render_multiple_choice(
    facts: str,
    correct_text: str,
    distractor_texts: list[str],
    prompt_template: str,
    rng: random.Random,
    use_gpt: bool = True,
    fallback_question: str = "Select the correct answer.",
) -> dict:

    items = [(correct_text, True)] + [(d, False) for d in distractor_texts]
    rng.shuffle(items)
    answer_idx = next(i for i, (_, ok) in enumerate(items) if ok)
    answer_letter = _LETTERS[answer_idx]

    if use_gpt:
        rendered = _try_gpt_mc(facts, items, prompt_template)
        if rendered is not None:
            question, option_texts = rendered
            options = [f"{_LETTERS[i]}. {t}" for i, t in enumerate(option_texts)]
            return {
                "question": question,
                "options": options,
                "answer": answer_letter,
                "rendered_by": "gpt",
            }


    options = [f"{_LETTERS[i]}. {t}" for i, (t, _) in enumerate(items)]
    return {
        "question": fallback_question,
        "options": options,
        "answer": answer_letter,
        "rendered_by": "template",
    }


def _try_gpt_mc(
    facts: str,
    items: list[tuple[str, bool]],
    prompt_template: str,
) -> Optional[tuple[str, list[str]]]:
    ordered_texts = [t for t, _ in items]
    labeled = "\n".join(f"{_LETTERS[i]}. {t}" for i, t in enumerate(ordered_texts))
    prompt = prompt_template.format(
        facts=facts.strip(),
        options=labeled,
        correct=next(t for t, ok in items if ok),
        distractors="; ".join(t for t, ok in items if not ok),
    )
    try:
        raw = _get_llm().prompt(prompt)
        data = json.loads(_strip_json_fence(raw))
    except Exception:
        return None

    question = data.get("question")
    opts = data.get("options")
    if not isinstance(question, str) or not question.strip():
        return None


    if isinstance(opts, dict):
        opts = [opts.get(_LETTERS[i], "") for i in range(len(ordered_texts))]
    if not isinstance(opts, list) or len(opts) != len(ordered_texts):
        return None
    if any(not isinstance(o, str) or not o.strip() for o in opts):
        return None
    return question.strip(), [o.strip() for o in opts]


def render_summary(facts: str, prompt_template: str) -> dict:
    prompt = prompt_template.format(facts=facts.strip())
    text = _get_llm().prompt(prompt).strip()
    return {"answer": text, "rendered_by": "gpt"}
