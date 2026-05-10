"""LangGraph workflow.

Flow:
    package → context (Agent 1, deterministic)
            ┌→ timeseries (Agent 2a, mini → SpecialistOutput) ┐
            ├→ events     (Agent 2b, mini → SpecialistOutput) │ parallel
            └→ quality    (Agent 2c, mini → SpecialistOutput) ┘
              → decision  (Agent 3, deterministic merge)
              → apply     (deterministic compound: baseline × ∏ multipliers)
              → report    (Agent 4, gpt-4o, Korean rich narrative)
              → END

Specialists analyse data and propose multipliers based on judgment.
Decision compilation is deterministic. Final report is the only
LLM-heavy narrative step (gpt-4o).
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents._trace import WorkflowTracer
from src.agents.context_collector import collect_context
from src.agents.decision_maker import DecisionOutput, compile_decision
from src.agents.forecast_adjuster import FinalForecast, apply_insights
from src.agents.pattern_analyst import (
    SpecialistOutput,
    run_data_quality_analyst,
    run_event_analyst,
    run_time_series_analyst,
)
from src.agents.report_writer import run_report_writer
from src.forecasting.package import PredictionPackage


class State(TypedDict, total=False):
    package: PredictionPackage
    tracer: WorkflowTracer
    context: dict[str, Any]
    timeseries_output: SpecialistOutput
    event_output: SpecialistOutput
    quality_output: SpecialistOutput
    decision: DecisionOutput
    final_forecast: FinalForecast
    report_md: str


def _context_node(state: State) -> dict[str, Any]:
    state["tracer"].set_agent("context_collector")
    ctx = collect_context(state["package"])
    return {"context": ctx}


def _timeseries_node(state: State) -> dict[str, Any]:
    tr = state["tracer"]
    tr.set_agent("timeseries_analyst")
    output = run_time_series_analyst(
        state["package"], state["context"], callbacks=[tr.callback_handler()]
    )
    return {"timeseries_output": output}


def _events_node(state: State) -> dict[str, Any]:
    tr = state["tracer"]
    tr.set_agent("event_analyst")
    output = run_event_analyst(
        state["package"], state["context"], callbacks=[tr.callback_handler()]
    )
    return {"event_output": output}


def _quality_node(state: State) -> dict[str, Any]:
    tr = state["tracer"]
    tr.set_agent("data_quality_analyst")
    output = run_data_quality_analyst(
        state["package"], state["context"], callbacks=[tr.callback_handler()]
    )
    return {"quality_output": output}


def _decision_node(state: State) -> dict[str, Any]:
    state["tracer"].set_agent("decision_maker")
    decision = compile_decision(
        state["package"],
        state["context"],
        state["timeseries_output"],
        state["event_output"],
        state["quality_output"],
    )
    return {"decision": decision}


def _apply_node(state: State) -> dict[str, Any]:
    state["tracer"].set_agent("apply")  # deterministic compound
    out = apply_insights(state["package"], state["decision"].selected_insights)
    return {"final_forecast": out}


def _report_node(state: State) -> dict[str, Any]:
    tr = state["tracer"]
    tr.set_agent("report_writer")
    md = run_report_writer(
        state["package"],
        state["context"],
        state["timeseries_output"].narrative,
        state["event_output"].narrative,
        state["quality_output"].narrative,
        state["decision"].selected_insights,
        state["final_forecast"],
        callbacks=[tr.callback_handler()],
    )
    return {"report_md": md}


def build_graph():
    g = StateGraph(State)
    g.add_node("context", _context_node)
    g.add_node("timeseries", _timeseries_node)
    g.add_node("events", _events_node)
    g.add_node("quality", _quality_node)
    g.add_node("decision", _decision_node)
    g.add_node("apply", _apply_node)
    g.add_node("report", _report_node)

    g.add_edge(START, "context")
    g.add_edge("context", "timeseries")
    g.add_edge("context", "events")
    g.add_edge("context", "quality")
    g.add_edge("timeseries", "decision")
    g.add_edge("events", "decision")
    g.add_edge("quality", "decision")
    g.add_edge("decision", "apply")
    g.add_edge("apply", "report")
    g.add_edge("report", END)

    return g.compile()


def run_workflow(
    package: PredictionPackage,
    run_id: str | None = None,
) -> tuple[State, WorkflowTracer]:
    """전체 파이프라인 실행. (final_state, tracer) 반환."""
    tracer = WorkflowTracer(run_id=run_id) if run_id else WorkflowTracer()
    app = build_graph()
    final = app.invoke({"package": package, "tracer": tracer})
    return final, tracer
