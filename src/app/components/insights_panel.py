"""Per-insight toggle + multiplier slider panel."""
from __future__ import annotations

import streamlit as st

from src.app.loaders import RunBundle
from src.app.state import ManagerOverrides

_TYPE_LABEL = {
    "level_weekday": "평일 level 보정",
    "level_weekend": "주말 level 보정",
    "event_peak": "이벤트 당일",
    "event_buildup": "이벤트 직전",
    "event_postlift": "이벤트 직후 상승",
    "event_antispike": "이벤트 직후 하락",
    "supply_outage_caveat": "공급 단절 caveat",
}


def _label_for(type_: str) -> str:
    return _TYPE_LABEL.get(type_, type_)


def render_insights_panel(
    bundle: RunBundle,
    overrides: ManagerOverrides,
) -> None:
    if not bundle.insights:
        st.info("에이전트가 제안한 보정 신호가 없습니다.")
        return

    st.caption("각 신호를 켜고 끄거나 multiplier 를 조정할 수 있습니다.")

    for ins in bundle.insights:
        cur_enabled = overrides.insight_enabled.get(ins.id, True)
        cur_mult = overrides.insight_multiplier.get(ins.id, ins.multiplier)
        delta_pct = (cur_mult - 1.0) * 100

        title = (
            f"**{_label_for(ins.type)}** — ×{cur_mult:.2f} "
            f"({delta_pct:+.0f}%) · {len(ins.dates)}일 적용"
        )
        with st.expander(title, expanded=False):
            cols = st.columns([1, 3])
            with cols[0]:
                new_enabled = st.checkbox(
                    "적용",
                    value=cur_enabled,
                    key=f"chk_{ins.id}",
                )
            with cols[1]:
                new_mult = st.slider(
                    "multiplier",
                    min_value=0.30,
                    max_value=3.00,
                    value=float(cur_mult),
                    step=0.05,
                    key=f"slider_{ins.id}",
                )
            st.caption(f"적용 일자: {', '.join(ins.dates)}")
            st.caption(f"근거: {ins.reason}")

            overrides.insight_enabled[ins.id] = new_enabled
            overrides.insight_multiplier[ins.id] = float(new_mult)
