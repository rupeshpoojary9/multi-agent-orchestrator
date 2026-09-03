"""Thin LLM wrapper — optional, and the system never depends on it.

If ANTHROPIC_API_KEY is set and `anthropic` is installed, `get_llm()` returns a
client that returns parsed JSON for a system+user prompt. Otherwise it returns
None and the planner/verifier/synthesizer fall back to their deterministic
heuristics. This keeps the whole project runnable and testable with no key while
leaving a real, drop-in model path for the interesting behavior.

Cost accounting is centralized here so every model call is metered the same way.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

DEFAULT_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "claude-sonnet-5")

# Rough per-token USD (input, output) for cost budgeting; override as needed.
_PRICE_PER_MTOK = {"input": 3.0, "output": 15.0}


@dataclass
class LLMResult:
    data: dict
    cost_usd: float


class LLM:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        import anthropic  # imported lazily; only when a key is present

        self.model = model
        self._client = anthropic.Anthropic()

    def complete_json(self, system: str, prompt: str, max_tokens: int = 1500) -> LLMResult:
        """Ask the model for a JSON object and parse it. Raises on invalid JSON."""
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        cost = (
            msg.usage.input_tokens / 1_000_000 * _PRICE_PER_MTOK["input"]
            + msg.usage.output_tokens / 1_000_000 * _PRICE_PER_MTOK["output"]
        )
        return LLMResult(data=_extract_json(text), cost_usd=cost)


def get_llm() -> LLM | None:
    """Return a live LLM if configured, else None (offline heuristic mode)."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        return LLM()
    except Exception:  # anthropic not installed, or client init failed
        return None


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    return json.loads(text)
