"""GSM8K adapter — grade-school math, exact numeric answers."""
import re
from datasets import load_dataset


def load(split: str = "test", max_samples: int | None = None) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    tasks = []
    for i, ex in enumerate(ds):
        answer = _extract_answer(ex["answer"])
        if not answer:
            continue
        tasks.append({
            "id": f"gsm8k_{i:04d}",
            "type": "reasoning",
            "eval": "contains",
            "source": "gsm8k",
            "prompt": ex["question"],
            "expected": [answer],
        })
    return tasks


def _extract_answer(raw: str) -> str:
    # GSM8K answers end with  #### <number>
    m = re.search(r"####\s*([\d,\.\-]+)", raw)
    if not m:
        return ""
    return m.group(1).replace(",", "").strip()
