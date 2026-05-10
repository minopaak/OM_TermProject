"""Live analysis runner with step-by-step UI feedback.

Runs the same agents as `src.agents.workflow` but sequentially, surfacing
each step's progress through `st.status` blocks. The report writer is
streamed token-by-token so the user can watch the assistant compose it.
After completion the artifacts are written into a fresh run directory so
the review dashboard can load them.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from datetime import date, datetime

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents._llm import get_llm
from src.agents.batch import _build_state_payload
from src.agents.context_collector import collect_context
from src.agents.decision_maker import compile_decision
from src.agents.forecast_adjuster import apply_insights
from src.agents.pattern_analyst import (
    SpecialistOutput,
    run_data_quality_analyst,
    run_event_analyst,
    run_time_series_analyst,
)
from src.agents.prompts import REPORT_WRITER_SYSTEM
from src.agents.report_writer import _build_input_message, _strip_code_fence
from src.config import DATA_DIR


_SIGNAL_LABEL = {
    "level_weekday": "평일 level 보정",
    "level_weekend": "주말 level 보정",
    "event_peak": "이벤트 당일",
    "event_buildup": "이벤트 직전",
    "event_postlift": "이벤트 직후 상승",
    "event_antispike": "이벤트 직후 하락",
    "supply_outage_caveat": "공급 caveat",
}


def _stream_report_tokens(
    package,
    context,
    timeseries_narrative: str,
    event_narrative: str,
    quality_narrative: str,
    selected_insights,
    final_forecast,
) -> Iterator[str]:
    """Yield report markdown chunks from gpt-4o."""
    llm = get_llm(model="gpt-4o")
    user_msg = _build_input_message(
        package,
        context,
        timeseries_narrative,
        event_narrative,
        quality_narrative,
        selected_insights,
        final_forecast,
    )
    for chunk in llm.stream(
        [
            SystemMessage(content=REPORT_WRITER_SYSTEM),
            HumanMessage(content=user_msg),
        ]
    ):
        text = chunk.content
        if text:
            yield text


def _render_specialist_body(out: SpecialistOutput) -> None:
    st.markdown("**분석 narrative**")
    st.markdown(out.narrative)
    if out.proposed_signals:
        st.markdown(f"**제안 신호 ({len(out.proposed_signals)}개)**")
        for s in out.proposed_signals:
            label = _SIGNAL_LABEL.get(s.type, s.type)
            delta = (s.multiplier - 1.0) * 100
            st.markdown(
                f"- `{s.type}` ({label}) ×{s.multiplier:.2f} "
                f"({delta:+.0f}%) · {len(s.dates)}일 · conf={s.confidence}"
            )
            st.caption(s.rationale)
    else:
        st.caption("제안 신호 없음")


def _save_run(
    sku_id: str,
    input_end_date: date,
    package,
    context,
    ts_out: SpecialistOutput,
    ev_out: SpecialistOutput,
    q_out: SpecialistOutput,
    decision,
    final_forecast,
    report_md: str,
    run_id: str,
) -> str:
    """Persist artifacts in the same layout `src.agents.batch` produces."""
    out_dir = DATA_DIR / "runs" / run_id
    for sub in ("reports", "intermediates", "forecasts", "state", "traces"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    (out_dir / "reports" / f"{sku_id}.md").write_text(report_md, encoding="utf-8")

    forecast_df = pd.DataFrame(
        [
            {
                "date": d.date,
                "yhat_baseline": d.yhat_baseline,
                "yhat_final": d.yhat_final,
                "applied": d.applied,
            }
            for d in final_forecast.final_forecast
        ]
    )
    forecast_df.to_parquet(
        out_dir / "forecasts" / f"{sku_id}.parquet", index=False
    )

    insights_md = "\n".join(
        f"- **{si.type}** (×{si.multiplier:.3f}): {si.dates} — {si.reason}"
        for si in decision.selected_insights
    )

    def _render(out) -> str:
        lines = [out.narrative if out else "(no output)", ""]
        if out and out.proposed_signals:
            lines.append("### Proposed signals")
            for s in out.proposed_signals:
                lines.append(
                    f"- `{s.type}` ×{s.multiplier:.3f} (conf={s.confidence}): "
                    f"dates={s.dates}"
                )
                lines.append(f"  - {s.rationale}")
        return "\n".join(lines)

    (out_dir / "intermediates" / f"{sku_id}.md").write_text(
        f"# Agent 1 Context\n\n```json\n{json.dumps(context, ensure_ascii=False, indent=2)}\n```\n\n"
        f"# Agent 2a 시계열 분석\n\n{_render(ts_out)}\n\n"
        f"---\n\n# Agent 2b 이벤트·인과 분석\n\n{_render(ev_out)}\n\n"
        f"---\n\n# Agent 2c 데이터 품질 분석\n\n{_render(q_out)}\n\n"
        f"---\n\n# Agent 3 Selected Insights (compiled)\n\n{insights_md}\n",
        encoding="utf-8",
    )

    class _CaseShim:
        def __init__(self, sku_id, input_end_date):
            self.sku_id = sku_id
            self.input_end_date = input_end_date

    state_payload = _build_state_payload(
        case=_CaseShim(sku_id, input_end_date),
        package=package,
        context=context,
        decision=decision,
        forecast_df=forecast_df,
    )
    (out_dir / "state" / f"{sku_id}.json").write_text(
        json.dumps(state_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return run_id


def run_analysis_with_cascade(
    package,
    sku_id: str,
    input_end_date: date,
) -> str:
    """Run the 4-agent workflow with `st.status` cascade visible inline.

    The PredictionPackage is built upstream (so the dashboard can show the
    pre-analysis chart immediately). This routine runs:
      context → 3 specialists → decision → apply → report (streaming)
    Persists the run dir and returns its `run_id`.
    """
    pkg = package
    run_id = f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sku_id}"
    overall_t0 = time.time()

    # --- Step 1: Context collector ----------------------------------------
    with st.status(
        "📅 컨텍스트 수집 중 (이벤트 · 요일 · SNAP)...",
        expanded=False,
        state="running",
    ) as status:
        t0 = time.time()
        ctx = collect_context(pkg)
        events = ctx.get("events", [])
        ev_str = (
            ", ".join(f"{e['name']}({e['date']})" for e in events) if events else "없음"
        )
        st.caption(f"이벤트: {ev_str}")
        st.caption(f"forecast 기간 SNAP 일수: {len(ctx.get('snap_days', []))}일")
        status.update(
            label=f"✅ 컨텍스트 수집 완료 ({time.time()-t0:.1f}s)",
            state="complete",
            expanded=False,
        )

    # --- Step 3a/b/c: 3 specialists ---------------------------------------
    with st.status(
        "📈 시계열 분석 중 (gpt-4o-mini)...",
        expanded=True,
        state="running",
    ) as status:
        t0 = time.time()
        ts_out = run_time_series_analyst(pkg, ctx)
        _render_specialist_body(ts_out)
        status.update(
            label=f"✅ 시계열 분석 ({len(ts_out.proposed_signals)}개 신호 제안 · {time.time()-t0:.1f}s)",
            state="complete",
            expanded=False,
        )

    with st.status(
        "🎉 이벤트·인과 분석 중 (gpt-4o-mini)...",
        expanded=True,
        state="running",
    ) as status:
        t0 = time.time()
        ev_out = run_event_analyst(pkg, ctx)
        _render_specialist_body(ev_out)
        status.update(
            label=f"✅ 이벤트·인과 분석 ({len(ev_out.proposed_signals)}개 신호 제안 · {time.time()-t0:.1f}s)",
            state="complete",
            expanded=False,
        )

    with st.status(
        "🧪 데이터 품질 분석 중 (gpt-4o-mini)...",
        expanded=True,
        state="running",
    ) as status:
        t0 = time.time()
        q_out = run_data_quality_analyst(pkg, ctx)
        _render_specialist_body(q_out)
        status.update(
            label=f"✅ 데이터 품질 분석 ({len(q_out.proposed_signals)}개 신호 제안 · {time.time()-t0:.1f}s)",
            state="complete",
            expanded=False,
        )

    # --- Step 4: Decision merge -------------------------------------------
    with st.status(
        "⚖️ 결정 합성 중 (deterministic merge)...",
        expanded=True,
        state="running",
    ) as status:
        t0 = time.time()
        decision = compile_decision(pkg, ctx, ts_out, ev_out, q_out)
        if decision.selected_insights:
            for si in decision.selected_insights:
                label = _SIGNAL_LABEL.get(si.type, si.type)
                delta = (si.multiplier - 1.0) * 100
                st.markdown(
                    f"- `{si.type}` ({label}) ×{si.multiplier:.3f} "
                    f"({delta:+.0f}%) · {len(si.dates)}일"
                )
                st.caption(si.reason)
        else:
            st.caption("채택된 신호 없음")
        status.update(
            label=f"✅ 결정 합성 완료 ({len(decision.selected_insights)}개 신호 채택 · {time.time()-t0:.1f}s)",
            state="complete",
            expanded=False,
        )

    # --- Step 5: Apply compound -------------------------------------------
    with st.status(
        "🔢 보정 적용 중 (deterministic compound)...",
        expanded=False,
        state="running",
    ) as status:
        t0 = time.time()
        final = apply_insights(pkg, decision.selected_insights)
        base_total = sum(d.yhat_baseline for d in final.final_forecast)
        final_total = sum(d.yhat_final for d in final.final_forecast)
        delta_pct = (final_total / base_total - 1.0) * 100 if base_total else 0
        st.caption(
            f"baseline 합계 {base_total:.0f} → 보정 후 {final_total:.0f} ({delta_pct:+.1f}%)"
        )
        status.update(
            label=f"✅ 보정 적용 완료 ({final.summary} · {time.time()-t0:.1f}s)",
            state="complete",
            expanded=False,
        )

    # --- Step 6: Report writer (streaming) --------------------------------
    with st.status(
        "📝 보고서 작성 중 (gpt-4o, 토큰 스트리밍)...",
        expanded=True,
        state="running",
    ) as status:
        t0 = time.time()

        def _gen():
            yield from _stream_report_tokens(
                pkg, ctx, ts_out.narrative, ev_out.narrative, q_out.narrative,
                decision.selected_insights, final,
            )

        accumulated = st.write_stream(_gen())
        report_md = _strip_code_fence(accumulated or "")
        status.update(
            label=f"✅ 보고서 작성 완료 ({time.time()-t0:.1f}s)",
            state="complete",
            expanded=True,
        )

    # --- Persist run dir --------------------------------------------------
    saved_id = _save_run(
        sku_id=sku_id,
        input_end_date=input_end_date,
        package=pkg,
        context=ctx,
        ts_out=ts_out,
        ev_out=ev_out,
        q_out=q_out,
        decision=decision,
        final_forecast=final,
        report_md=report_md,
        run_id=run_id,
    )
    st.success(
        f"🎯 분석 완료 (총 {time.time()-overall_t0:.1f}s) — "
        f"`data/runs/{saved_id}/` 에 저장됨"
    )
    return saved_id
