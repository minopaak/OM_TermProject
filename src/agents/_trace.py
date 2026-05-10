"""에이전트 실행 추적: LLM 호출 토큰·비용·시간을 JSONL 로 기록.

사용:
    tracer = WorkflowTracer(run_id="...")
    callbacks = [tracer.callback_handler()]
    response = llm.invoke(messages, config={"callbacks": callbacks})
    tracer.write(log_path)
    summary = tracer.summary()
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

# OpenAI 가격 (per 1M tokens, 2026-05 기준)
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.00},
}


def _estimate_cost(model: str | None, prompt_tokens: int, completion_tokens: int) -> float:
    """간단한 USD 비용 추정. unknown 모델은 0."""
    if not model:
        return 0.0
    p = PRICING.get(model)
    if p is None:
        # prefix matching (예: 'gpt-4o-mini-...')
        for k, v in PRICING.items():
            if model.startswith(k):
                p = v
                break
    if p is None:
        return 0.0
    return (prompt_tokens * p["input"] + completion_tokens * p["output"]) / 1_000_000


@dataclass
class WorkflowTracer:
    """한 번의 워크플로 실행에 대한 추적 결과."""
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    events: list[dict[str, Any]] = field(default_factory=list)
    current_agent: str = "unknown"

    def set_agent(self, name: str) -> None:
        self.current_agent = name

    def add_event(self, event: dict[str, Any]) -> None:
        event = {"run_id": self.run_id, "agent": self.current_agent, **event}
        self.events.append(event)

    def callback_handler(self) -> BaseCallbackHandler:
        return _TracerCallback(self)

    def summary(self) -> dict[str, Any]:
        llm_events = [e for e in self.events if e["type"] == "llm_call"]
        total_in = sum(e.get("prompt_tokens", 0) for e in llm_events)
        total_out = sum(e.get("completion_tokens", 0) for e in llm_events)
        total_cost = sum(e.get("cost_usd", 0.0) for e in llm_events)
        total_llm_time = sum(e.get("latency_s", 0.0) for e in llm_events)

        per_agent: dict[str, dict[str, float]] = {}
        for e in llm_events:
            d = per_agent.setdefault(
                e["agent"],
                {"llm_calls": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0},
            )
            d["llm_calls"] += 1
            d["tokens_in"] += e.get("prompt_tokens", 0)
            d["tokens_out"] += e.get("completion_tokens", 0)
            d["cost_usd"] += e.get("cost_usd", 0.0)

        return {
            "run_id": self.run_id,
            "n_llm_calls": len(llm_events),
            "tokens_in": total_in,
            "tokens_out": total_out,
            "total_llm_time_s": round(total_llm_time, 2),
            "estimated_cost_usd": round(total_cost, 6),
            "per_agent": per_agent,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")


class _TracerCallback(BaseCallbackHandler):
    """LangChain LLM·tool 호출을 가로채 tracer에 기록."""

    def __init__(self, tracer: WorkflowTracer) -> None:
        self._tracer = tracer
        self._llm_starts: dict[UUID, float] = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        run_id = kwargs.get("run_id")
        self._llm_starts[run_id] = time.time()

    def on_chat_model_start(self, serialized, messages, **kwargs):
        run_id = kwargs.get("run_id")
        self._llm_starts[run_id] = time.time()

    def on_llm_end(self, response, **kwargs):
        run_id = kwargs.get("run_id")
        t0 = self._llm_starts.pop(run_id, None)
        latency = (time.time() - t0) if t0 is not None else 0.0

        usage: dict[str, Any] = {}
        model: str | None = None
        if response.llm_output:
            usage = response.llm_output.get("token_usage") or {}
            model = response.llm_output.get("model_name")
        # response.generations[0][0].message.usage_metadata 도 확인 (newer langchain)
        if not usage and response.generations:
            try:
                meta = response.generations[0][0].message.usage_metadata  # type: ignore[attr-defined]
                if meta:
                    usage = {
                        "prompt_tokens": meta.get("input_tokens", 0),
                        "completion_tokens": meta.get("output_tokens", 0),
                        "total_tokens": meta.get("total_tokens", 0),
                    }
            except Exception:
                pass

        prompt_t = int(usage.get("prompt_tokens", 0))
        completion_t = int(usage.get("completion_tokens", 0))
        cost = _estimate_cost(model, prompt_t, completion_t)

        self._tracer.add_event(
            {
                "type": "llm_call",
                "model": model,
                "latency_s": round(latency, 3),
                "prompt_tokens": prompt_t,
                "completion_tokens": completion_t,
                "cost_usd": round(cost, 6),
            }
        )

