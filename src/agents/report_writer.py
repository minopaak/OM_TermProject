"""Agent 4: Report Writer (gpt-4o, Korean).

Receives:
  - the prediction package (input window, baseline)
  - context (events, weekdays, SNAP)
  - the 3 specialists' English narratives
  - the deterministic decision (selected_insights)
  - the deterministic final forecast (baseline × multipliers)

Produces a Korean manager-facing markdown report with concrete numbers,
event box, applied/excluded tables, and reviewer checkpoints.
"""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agents._llm import get_llm
from src.agents.decision_maker import SelectedInsight
from src.agents.forecast_adjuster import FinalForecast
from src.agents.prompts import REPORT_WRITER_SYSTEM
from src.forecasting.package import PredictionPackage


_WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]


def _to_date(d) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(str(d), "%Y-%m-%d").date()


def _format_forecast_table(final_fc: FinalForecast) -> str:
    """일자별 baseline·multiplier·final 표 (markdown)."""
    lines = [
        "| 일자 | 요일 | baseline | multiplier | 보정 후 | 적용 신호 |",
        "|------|------|----------|------------|---------|-----------|",
    ]
    for d in final_fc.final_forecast:
        dow = _WEEKDAY_KO[_to_date(d.date).weekday()]
        if d.yhat_baseline > 0:
            mult = d.yhat_final / d.yhat_baseline
        else:
            mult = 1.0
        m_str = f"×{mult:.3f}" if abs(mult - 1.0) > 1e-9 else "—"
        applied_label = d.applied or "—"
        lines.append(
            f"| {d.date} | {dow} | {d.yhat_baseline:.2f} | {m_str} | "
            f"{d.yhat_final:.2f} | {applied_label} |"
        )
    return "\n".join(lines)


def _format_insights_table(insights: list[SelectedInsight]) -> str:
    if not insights:
        return "(채택된 insight 없음 — 보정 미적용)"
    lines = [
        "| type | dates | multiplier | reason |",
        "|------|-------|------------|--------|",
    ]
    for si in insights:
        dates_str = (
            f"{si.dates[0]} ({len(si.dates)} days)"
            if len(si.dates) > 3
            else ", ".join(si.dates)
        )
        lines.append(
            f"| `{si.type}` | {dates_str} | ×{si.multiplier:.3f} | {si.reason} |"
        )
    return "\n".join(lines)


def _build_input_message(
    package: PredictionPackage,
    context: dict[str, Any],
    timeseries_analysis: str,
    event_analysis: str,
    quality_analysis: str,
    selected_insights: list[SelectedInsight],
    final_forecast: FinalForecast,
) -> str:
    fc_dates = [str(d) for d in package.forecast_window.dates]
    fc_values = [round(float(v), 2) for v in package.forecast_window.values]
    in_dates = [str(d) for d in package.input_window.dates]
    in_values = [round(float(v), 2) for v in package.input_window.values]

    base_total = sum(fc_values)
    final_total = sum(d.yhat_final for d in final_forecast.final_forecast)
    delta_pct = (final_total / base_total - 1.0) * 100 if base_total > 0 else 0.0

    return f"""\
## 대상 SKU
- sku_id: `{package.sku_id}` (cat={package.cat_id}, dept={package.dept_id}, store={package.store_id}, state={package.state_id})

## input window (직전 28일 실제 sales)
- 기간: {in_dates[0]} ~ {in_dates[-1]}
- 합계 {sum(in_values):.0f}, 평균 {sum(in_values)/28:.2f}
- 일자별: {list(zip(in_dates, in_values))}

## baseline forecast (다음 28일)
- 기간: {fc_dates[0]} ~ {fc_dates[-1]}
- 합계 {base_total:.2f}, 평균 {base_total/28:.2f}

## 보정 후 forecast (최종)
- 합계 {final_total:.2f}, 평균 {final_total/28:.2f}
- baseline 대비 {delta_pct:+.1f}%
- 일자별 baseline·multiplier·final 표:

{_format_forecast_table(final_forecast)}

## 적용된 insight 들

{_format_insights_table(selected_insights)}

## Agent 1 출력 (이벤트·요일·SNAP)
```json
{json.dumps(context, ensure_ascii=False, indent=2)}
```

---

## 시계열 전문가 분석
{timeseries_analysis}

---

## 이벤트·인과 전문가 분석
{event_analysis}

---

## 데이터 품질 전문가 분석
{quality_analysis}

---

## 당신의 작업

위 모든 정보를 종합해 매니저용 rich narrative markdown 보고서를 작성한다.

핵심 요구:
- baseline 합계 {base_total:.0f} → 보정 후 {final_total:.0f} ({delta_pct:+.1f}%) 처럼 *구체 숫자* 인용.
- 적용된 multiplier 와 일자를 표로 명시.
- 이벤트는 forecast 기간 상단에 박스/표로 가시화 (채택 안 한 이벤트도 *왜* 안 했는지 명시).
- SKU-specific narrative — 카테고리·점포·기간 컨텍스트와 연결.
- 내부 type 이름 (`level_bias_weekday` 등) 본문 노출 X — 풀어서 매니저 언어로.
- **시나리오 섹션 작성 X**.
"""


def _strip_code_fence(text: str) -> str:
    """LLM 이 markdown 본문을 ```markdown ... ``` 으로 감싸서 출력하는 경우 벗김."""
    s = text.strip()
    if s.startswith("```"):
        # 첫 줄 (```markdown 또는 ```) 제거
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        # 끝 ``` 제거
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip() + "\n"
    return s


def run_report_writer(
    package: PredictionPackage,
    context: dict[str, Any],
    timeseries_analysis: str,
    event_analysis: str,
    quality_analysis: str,
    selected_insights: list[SelectedInsight],
    final_forecast: FinalForecast,
    callbacks: list | None = None,
) -> str:
    """rich narrative markdown 보고서를 생성. 문자열 반환."""
    llm = get_llm(model="gpt-4o")
    user_msg = _build_input_message(
        package,
        context,
        timeseries_analysis,
        event_analysis,
        quality_analysis,
        selected_insights,
        final_forecast,
    )
    result = llm.invoke(
        [
            SystemMessage(content=REPORT_WRITER_SYSTEM),
            HumanMessage(content=user_msg),
        ],
        config={"callbacks": callbacks or []},
    )
    return _strip_code_fence(result.content)
