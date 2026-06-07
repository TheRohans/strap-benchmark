"""Score a model response against a task's expected output."""
import re
from openai import OpenAI


def _normalise(text: str) -> str:
    """Strip markdown formatting and normalise numbers for comparison."""
    # remove bold/italic markers
    text = re.sub(r"\*{1,3}|_{1,3}", "", text)
    # remove inline code
    text = re.sub(r"`[^`]*`", lambda m: m.group().strip("`"), text)
    # remove currency symbols
    text = re.sub(r"[$€£¥]", "", text)
    # remove thousands separators (commas between digits)
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    return text

_judge_client: OpenAI | None = None

_DEFAULT_JUDGE_URL   = "http://localhost:11434/v1"
_DEFAULT_JUDGE_MODEL = "qwen2.5:14b"


def _get_judge() -> tuple[OpenAI, str]:
    import os
    global _judge_client
    base_url = os.environ.get("JUDGE_BASE_URL", _DEFAULT_JUDGE_URL)
    model    = os.environ.get("JUDGE_MODEL",    _DEFAULT_JUDGE_MODEL)
    api_key  = os.environ.get("JUDGE_API_KEY",  "ollama")
    if _judge_client is None:
        _judge_client = OpenAI(base_url=base_url, api_key=api_key)
    return _judge_client, model


def score(task: dict, response: str) -> dict:
    method = task["eval"]

    if method == "exact":
        return _exact(task, response)
    elif method == "contains":
        return _contains(task, response)
    elif method == "llm_judge":
        return _llm_judge(task, response)
    else:
        raise ValueError(f"unknown eval method: {method!r}")


def _exact(task: dict, response: str) -> dict:
    expected = str(task["expected"]).strip()
    normalised = _normalise(response)
    passed = expected.lower() in normalised.lower()
    return {"pass": passed, "method": "exact", "expected": expected, "got": response[:200]}


def _contains(task: dict, response: str) -> dict:
    needles = task["expected"] if isinstance(task["expected"], list) else [task["expected"]]
    normalised = _normalise(response)
    missing = [n for n in needles if n.lower() not in normalised.lower()]
    return {
        "pass": len(missing) == 0,
        "method": "contains",
        "missing": missing,
        "got": response[:200],
    }


def _llm_judge(task: dict, response: str) -> dict:
    client, model = _get_judge()

    # build context for the judge
    if task["type"] == "memory":
        conversation = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in task["turns"]
        )
        context = f"Conversation so far:\n{conversation}\n\nModel response:\n{response}"
    else:
        context = f"Task: {task.get('prompt', '')}\n\nModel response:\n{response}"

    judge_prompt = (
        f"{context}\n\n"
        f"Evaluation criteria: {task['judge_prompt']}\n\n"
        "Your response MUST begin with the single word PASS or FAIL (all caps), "
        "followed by a colon and one short reason. Example: 'PASS: answer matches.'"
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=150,
    )
    verdict_text = (resp.choices[0].message.content or "").strip()
    passed = _parse_verdict(verdict_text)
    return {"pass": passed, "method": "llm_judge", "verdict": verdict_text, "got": response[-300:]}


def _parse_verdict(text: str) -> bool:
    """Find the last PASS/FAIL keyword in the verdict; fall back to positive-language detection."""
    hits = [(m.start(), m.group().upper()) for m in re.finditer(r"\b(PASS|FAIL)\b", text, re.IGNORECASE)]
    if hits:
        return hits[-1][1] == "PASS"
    # Judge didn't use the keywords — look for conclusive positive language
    positive = re.search(r"\b(correct|right|accurate|yes)\b", text, re.IGNORECASE)
    negative = re.search(r"\b(incorrect|wrong|missing|no\b|not correct)", text, re.IGNORECASE)
    if positive and not negative:
        return True
    return False
