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


def _get_judge(base_url: str = "http://localhost:11434/v1", model: str = "qwen2.5:14b") -> tuple[OpenAI, str]:
    global _judge_client
    if _judge_client is None:
        _judge_client = OpenAI(base_url=base_url, api_key="ollama")
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
        "Answer with exactly PASS or FAIL, then a one-sentence reason."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=100,
    )
    verdict_text = (resp.choices[0].message.content or "").strip()
    passed = verdict_text.upper().startswith("PASS")
    return {"pass": passed, "method": "llm_judge", "verdict": verdict_text, "got": response[:200]}
