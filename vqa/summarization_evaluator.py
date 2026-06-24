from openai import OpenAI
import json
import logging
import os
import re
import sys
import tqdm

try:
    import yaml
except ImportError:
    yaml = None


def _judge_model() -> str:
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
    if yaml and os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                return yaml.safe_load(f).get("models", {}).get("judge", "gpt-5.2")
        except Exception:
            pass
    return "gpt-5.2"


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class LLM:
    def __init__(self, model="gpt-5.2"):
        self.client = OpenAI()
        self.model = model

    def prompt(self, p):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": p}],
            store=False,
        )
        return response.choices[0].message.content or ""


class ConceptualSimilarityError(Exception):
    pass


_SYSTEM_PROMPT = (
    "You are an intelligent chatbot designed for evaluating the conceptual "
    "and semantic similarity of generated summaries against a reference text. "
    "Your task is to compare the Generated Text with the Reference Text and "
    "determine how well the core concepts, facts, and conclusions are preserved. "
    "Here is how you accomplish the task:\n"
    "------\n"
    "##INSTRUCTIONS:\n"
    "- Focus on whether the same core facts, entities, numerical values, causal "
    "claims, and key conclusions are present. Ignore surface differences in "
    "vocabulary, grammar, word order, or sentence length.\n"
    "- Consider synonyms and paraphrases as valid matches "
    "(e.g. '20%' and 'one-fifth' are equivalent).\n"
    "- Penalise the Generated Text if it introduces hallucinations (information "
    "absent from the Reference), omits critical facts, or contradicts the Reference.\n"
    "- Do NOT reward verbosity. A longer Generated Text is not better unless the "
    "extra content is supported by the Reference.\n"
    "- Provide a single similarity score that reflects conceptual fidelity."
)

_USER_PROMPT = (
    "Please evaluate the following summarization pair:\n\n"
    "Reference Text (ground truth):\n{ref}\n\n"
    "Generated Text (model output):\n{gen}\n\n"
    "First, write ONE sentence of feedback explaining your score and citing "
    "specific agreements or gaps (e.g. hallucinations, omissions).\n"
    "Then output your score as: <score>NUMBER</score>\n\n"
    "The NUMBER must be a decimal between 0.0 and 1.0, where:\n"
    "  1.00 = perfect - all core concepts captured, no hallucinations or omissions\n"
    "  0.75 = high    - primary concepts present, only minor details missing\n"
    "  0.50 = moderate - some overlap but at least one major concept missing or inaccurate\n"
    "  0.25 = low     - largely unrelated or contradicts the reference\n"
    "  0.00 = none    - no semantic overlap\n"
    "Scores may be any value in [0.00, 1.00], not just the anchors above.\n"
    "Do not place any text after the closing </score> tag."
)


_llm = LLM(model=_judge_model())


def compare(a: str, b: str) -> float:
    full_prompt = f"{_SYSTEM_PROMPT}\n\n{_USER_PROMPT.format(ref=a.strip(), gen=b.strip())}"
    response = _llm.prompt(full_prompt)
    logger.debug("LLM raw output:\n%s", response)

    match = re.search(r"<score>\s*([0-9]*\.?[0-9]+)\s*</score>", response)
    if match:
        return max(0.0, min(1.0, float(match.group(1))))

    logger.warning("Score tag not found; falling back to last float in response.")
    floats = re.findall(r"\b(0\.\d+|1\.0|0\.0|1|0)\b", response)
    if floats:
        return max(0.0, min(1.0, float(floats[-1])))

    raise ConceptualSimilarityError(
        f"Could not parse a valid score from LLM output:\n{response}"
    )


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <summarization.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    total = 0.0
    n = 0
    for entry in tqdm.tqdm(data.values(), total=len(data)):
        gen = entry.get("generated_answer", "")
        ref = entry.get("reference_answer", "")
        if not gen or not ref:
            continue
        try:
            total += compare(ref, gen)
            n += 1
        except ConceptualSimilarityError as ex:
            logger.warning("skipping entry: %s", ex)

    print(total / n if n else 0.0)


if __name__ == "__main__":
    main()
