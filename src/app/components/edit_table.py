"""Daily forecast edit table.

`st.data_editor` with one editable column (`매니저 최종`). Edits flow
into `overrides.cell_overrides` so a manual entry wins over the
compound recompute.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src.app.loaders import RunBundle
from src.app.state import ManagerOverrides

_WEEKDAY_KO = ["월", "화", "수", "목", "금", "토", "일"]


def render_edit_table(
    bundle: RunBundle,
    overrides: ManagerOverrides,
    manager_values: list[float],
    applied_labels: list[str],
    key: str = "edit_table",
) -> None:
    rows = []
    for date_s, base, agent, mgr, label in zip(
        bundle.baseline_dates,
        bundle.baseline_values,
        bundle.agent_final_values,
        manager_values,
        applied_labels,
    ):
        wd = _WEEKDAY_KO[datetime.strptime(date_s, "%Y-%m-%d").weekday()]
        rows.append(
            {
                "일자": date_s,
                "요일": wd,
                "baseline": round(float(base), 2),
                "에이전트 보정": round(float(agent), 2),
                "매니저 최종": round(float(mgr), 2),
                "적용": label,
            }
        )
    df = pd.DataFrame(rows)

    edited = st.data_editor(
        df,
        key=key,
        hide_index=True,
        column_config={
            "일자": st.column_config.TextColumn("일자", disabled=True, width="small"),
            "요일": st.column_config.TextColumn("요일", disabled=True, width="small"),
            "baseline": st.column_config.NumberColumn(
                "baseline", disabled=True, format="%.1f", width="small"
            ),
            "에이전트 보정": st.column_config.NumberColumn(
                "에이전트 보정", disabled=True, format="%.1f", width="small"
            ),
            "매니저 최종": st.column_config.NumberColumn(
                "매니저 최종", min_value=0.0, format="%.1f", width="small"
            ),
            "적용": st.column_config.TextColumn("적용", disabled=True, width="medium"),
        },
        width="stretch",
        height=min(420, 38 * (len(df) + 1) + 6),
    )

    # Diff: what changed in 매니저 최종 column → push into cell_overrides.
    for date_s, mgr_new, mgr_old in zip(
        edited["일자"], edited["매니저 최종"], df["매니저 최종"]
    ):
        if pd.isna(mgr_new):
            continue
        new_v = float(mgr_new)
        if abs(new_v - float(mgr_old)) > 1e-3:
            overrides.cell_overrides[date_s] = new_v
