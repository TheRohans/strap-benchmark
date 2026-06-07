"""MATH-500 adapter — competition math, LLM-judged since answers are LaTeX expressions."""
from datasets import load_dataset

# Subjects available: Algebra, Counting & Probability, Geometry,
# Intermediate Algebra, Number Theory, Prealgebra, Precalculus
ALL_SUBJECTS = None  # None = all

JUDGE_PROMPT = (
    "The model was asked to solve a math problem. "
    "The correct answer is: {expected}\n"
    "Check the model's response — especially any \\boxed{{}} expression at the end — "
    "and decide whether it contains the correct answer, even if formatted differently "
    "(e.g. equivalent fractions, simplified radicals, different LaTeX notation). "
    "Answer PASS if correct, FAIL if wrong or missing."
)

_ANSWER_SUFFIX = (
    "\n\nReason step by step, then state your final answer on its own line "
    "enclosed in \\boxed{} notation, like: \\boxed{42}"
)


def load(
    max_samples: int | None = None,
    max_level: int = 5,
    subjects: list[str] | None = ALL_SUBJECTS,
) -> list[dict]:
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

    if subjects:
        ds = ds.filter(lambda ex: ex["subject"] in subjects)
    if max_level < 5:
        ds = ds.filter(lambda ex: ex["level"] <= max_level)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    tasks = []
    for i, ex in enumerate(ds):
        tasks.append({
            "id": f"math_{ex['unique_id'].replace('/', '_')}",
            "type": "reasoning",
            "eval": "llm_judge",
            "source": "math500",
            "subject": ex["subject"],
            "level": ex["level"],
            "prompt": ex["problem"] + _ANSWER_SUFFIX,
            "expected": ex["answer"],
            "judge_prompt": JUDGE_PROMPT.format(expected=ex["answer"]),
        })
    return tasks
