"""Baseline runner: direct Ollama API call, no harness, no tools."""
import os
from openai import OpenAI

_DEFAULT_BASE_URL = "http://localhost:11434/v1"
_DEFAULT_MODEL    = "qwen2.5:14b"


class QwenRunner:
    def __init__(
        self,
        base_url: str = os.environ.get("QWEN_BASE_URL", _DEFAULT_BASE_URL),
        model:    str = os.environ.get("QWEN_MODEL",    _DEFAULT_MODEL),
    ):
        self.client = OpenAI(base_url=base_url, api_key="ollama")
        self.model = model

    def run(self, task: dict) -> str:
        messages = self._build_messages(task)
        resp = self.client.chat.completions.create(model=self.model, messages=messages)
        return resp.choices[0].message.content or ""

    def reset(self) -> None:
        pass  # stateless — nothing to clear

    def close(self) -> None:
        pass

    def _build_messages(self, task: dict) -> list[dict]:
        if task["type"] == "reasoning":
            return [{"role": "user", "content": task["prompt"]}]
        # multi-turn: send full conversation history
        return [{"role": t["role"], "content": t["content"]} for t in task["turns"]]
