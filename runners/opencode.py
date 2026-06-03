"""OpenCode runner.

OpenCode is a TUI application. Check whether it has a non-interactive/headless
mode by running: opencode --help

If it supports piping (e.g. `echo "prompt" | opencode`), set PIPE_MODE = True.
If it has an HTTP API mode, implement _run_api() and switch to that instead.
"""
import subprocess
import textwrap


PIPE_MODE = True  # set False and implement _run_api() if opencode has an API

# The model flag — check opencode --help for the exact flag name
MODEL_FLAG = "--model"
MODEL = "qwen2.5:14b"


class OpenCodeRunner:
    def __init__(self, binary: str = "opencode", model: str = MODEL):
        self.binary = binary
        self.model = model

    def run(self, task: dict) -> str:
        prompt = self._build_prompt(task)
        if PIPE_MODE:
            return self._run_pipe(prompt)
        return self._run_api(prompt)

    def reset(self) -> None:
        pass  # stateless between calls in pipe mode

    def close(self) -> None:
        pass

    def _build_prompt(self, task: dict) -> str:
        if task["type"] == "reasoning":
            return task["prompt"]
        # flatten multi-turn into a single prompt
        # opencode doesn't have a session API we can replay turns into
        lines = []
        for t in task["turns"]:
            lines.append(f"{t['role'].upper()}: {t['content']}")
        return "\n".join(lines) + "\nPlease respond as the assistant."

    def _run_pipe(self, prompt: str) -> str:
        # TODO: verify exact invocation — this is a best guess
        result = subprocess.run(
            [self.binary, MODEL_FLAG, self.model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"opencode failed: {result.stderr}")
        return result.stdout.strip()

    def _run_api(self, prompt: str) -> str:
        # implement if opencode exposes an HTTP API
        raise NotImplementedError("set PIPE_MODE=True or implement _run_api()")
