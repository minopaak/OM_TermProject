"""Streamlit review app entry point.

Single unified dashboard. Sequence:

    1. user picks SKU + input 마감일 in sidebar (input/output windows are
       fixed at 28 + 28 days by the time-series model)
    2. PredictionPackage builds → chart shows input + baseline (2 traces)
    3. user clicks "▶ AI 에이전트 분석 시작" — the 4-agent workflow runs in
       place with `st.status` cascade, the report is streamed token-by-token
    4. on completion, chart expands to 4 traces, edit/signal/report tabs
       and chat assistant become available; manager can adjust and 컨펌

Run with:
    .\\.venv\\Scripts\\streamlit.exe run app.py
"""
from __future__ import annotations

import json as _json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app.chat_agent import ChatTurn, chat_step
from src.app.components.chart import build_chart
from src.app.components.edit_table import render_edit_table
from src.app.components.insights_panel import render_insights_panel
from src.app.live import run_analysis_with_cascade
from src.app.loaders import (
    RunBundle, list_run_ids, list_skus, load_actuals, load_bundle,
)
from src.app.state import ManagerOverrides, compute_manager_final
from src.config import DATA_DIR
from src.forecasting.model import get_backend_name
from src.forecasting.package import build_prediction_package


# ---------------------------------------------------------------- helpers


def _save_manager_final(
    bundle: RunBundle,
    overrides: ManagerOverrides,
    manager_values: list[float],
    applied_labels: list[str],
) -> Path:
    out_dir = bundle.run_dir / "manager_final"
    out_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(
        {
            "date": bundle.baseline_dates,
            "yhat_baseline": bundle.baseline_values,
            "yhat_agent_final": bundle.agent_final_values,
            "yhat_manager_final": manager_values,
            "applied": applied_labels,
        }
    )
    out_path = out_dir / f"{bundle.sku_id}.parquet"
    df.to_parquet(out_path, index=False)
    (out_dir / f"{bundle.sku_id}.meta.json").write_text(
        _json.dumps(
            {
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "global_multiplier": overrides.global_multiplier,
                "insight_enabled": overrides.insight_enabled,
                "insight_multiplier": overrides.insight_multiplier,
                "cell_overrides": overrides.cell_overrides,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_path


def _ensure_overrides(bundle: RunBundle) -> ManagerOverrides:
    cur = (bundle.run_id, bundle.sku_id)
    if st.session_state.get("loaded_key") != cur:
        st.session_state["loaded_key"] = cur
        st.session_state["overrides"] = ManagerOverrides.from_bundle(bundle)
        st.session_state["chat_history"] = []
        st.session_state["last_actions"] = []
    return st.session_state["overrides"]


def _ensure_package(sku_id: str, input_end_date: date):
    """Rebuild the package when SKU/date change. Clears any active bundle."""
    pkg_key = (sku_id, str(input_end_date))
    if st.session_state.get("pkg_key") == pkg_key:
        return st.session_state["package"]

    with st.spinner(f"baseline 모델로 forecast 생성 중 ({sku_id})..."):
        pkg = build_prediction_package(sku_id, input_end_date)
    st.session_state["pkg_key"] = pkg_key
    st.session_state["package"] = pkg
    # SKU/date changed — old analysis is stale.
    st.session_state.pop("active_run_id", None)
    st.session_state.pop("loaded_key", None)
    st.session_state.pop("overrides", None)
    st.session_state.pop("chat_history", None)
    return pkg


# ---------------------------------------------------------------- sidebar


def render_sidebar() -> tuple[str, date]:
    st.sidebar.markdown("### 분석 대상")

    sku_id = st.sidebar.text_input(
        "SKU id",
        value=st.session_state.get("sidebar_sku", "FOODS_3_295_CA_1"),
        key="sidebar_sku",
    )
    input_end_date = st.sidebar.date_input(
        "input 마감일",
        value=st.session_state.get("sidebar_date", date(2016, 1, 28)),
        min_value=date(2011, 2, 25),
        max_value=date(2016, 5, 22),
        key="sidebar_date",
    )
    st.sidebar.caption(
        f"input 28일 → forecast 28일 (시계열 모델 고정) · "
        f"baseline = **{get_backend_name().upper()}**"
    )

    st.sidebar.divider()
    with st.sidebar.expander("기존 분석 불러오기", expanded=False):
        runs = list_run_ids()
        if runs:
            picked = st.selectbox("Run", runs, key="sidebar_pick_run")
            skus = list_skus(picked) if picked else []
            picked_sku = st.selectbox("SKU", skus, key="sidebar_pick_sku") if skus else None
            if picked_sku and st.button("이 분석 불러오기", width="stretch"):
                st.session_state["pending_load"] = (picked, picked_sku)
                st.rerun()
        else:
            st.caption("저장된 run 이 없습니다.")

    return sku_id, input_end_date


# ---------------------------------------------------------------- header


def _mae(pred: list[float], actual: list[float] | None) -> float | None:
    if not actual or len(pred) != len(actual):
        return None
    return sum(abs(float(p) - float(a)) for p, a in zip(pred, actual)) / len(pred)


def _render_header(
    pkg,
    bundle: RunBundle | None,
    manager_values: list[float] | None,
    actual_values: list[float] | None,
):
    sku_id = pkg.sku_id
    cat_id = pkg.cat_id
    store_id = pkg.store_id
    state_id = pkg.state_id
    fc_dates = [str(d) for d in pkg.forecast_window.dates]

    st.markdown(f"## {sku_id}")
    st.caption(
        f"`{cat_id}` · `{store_id}` · `{state_id}` "
        f" |  baseline = **{get_backend_name().upper()}** "
        f" |  input {pkg.input_window.dates[0]} ~ {pkg.input_window.dates[-1]} "
        f" |  forecast {fc_dates[0]} ~ {fc_dates[-1]}"
    )

    base_total = sum(pkg.forecast_window.values)
    actual_total = sum(actual_values) if actual_values else None
    base_mae = _mae(list(pkg.forecast_window.values), actual_values)

    if bundle is None:
        if actual_total is not None:
            cols = st.columns(3)
            cols[0].metric("input 합계", f"{sum(pkg.input_window.values):.0f}")
            cols[1].metric(
                "baseline 합계",
                f"{base_total:.0f}",
                delta=f"{(base_total - actual_total):+.0f} vs 정답",
                delta_color="inverse",
            )
            cols[2].metric("정답 합계", f"{actual_total:.0f}")
        else:
            cols = st.columns(2)
            cols[0].metric("input 합계", f"{sum(pkg.input_window.values):.0f}")
            cols[1].metric("baseline 합계", f"{base_total:.0f}")
    else:
        agent_total = sum(bundle.agent_final_values)
        mgr_total = sum(manager_values or bundle.agent_final_values)
        agent_mae = _mae(list(bundle.agent_final_values), actual_values)
        mgr_mae = _mae(list(manager_values or bundle.agent_final_values), actual_values)

        if actual_total is not None:
            cols = st.columns(4)
            cols[0].metric(
                "baseline 합계",
                f"{base_total:.0f}",
                delta=f"{(base_total - actual_total):+.0f} vs 정답",
                delta_color="inverse",
            )
            cols[1].metric(
                "에이전트 보정",
                f"{agent_total:.0f}",
                delta=f"{(agent_total - actual_total):+.0f} vs 정답",
                delta_color="inverse",
            )
            cols[2].metric(
                "매니저 최종",
                f"{mgr_total:.0f}",
                delta=f"{(mgr_total - actual_total):+.0f} vs 정답",
                delta_color="inverse",
                help="매니저 조정 결과의 정답 대비 차이",
            )
            cols[3].metric("정답 합계", f"{actual_total:.0f}")
        else:
            cols = st.columns(3)
            cols[0].metric("baseline 합계", f"{base_total:.0f}")
            cols[1].metric(
                "에이전트 보정",
                f"{agent_total:.0f}",
                delta=f"{(agent_total - base_total):+.0f}",
            )
            cols[2].metric(
                "매니저 최종",
                f"{mgr_total:.0f}",
                delta=f"{(mgr_total - agent_total):+.0f}",
                help="에이전트 보정 대비 매니저 조정 차이",
            )

    if actual_values:
        line_parts = [f"baseline MAE = **{base_mae:.2f}**"]
        if bundle is not None:
            line_parts.append(f"에이전트 MAE = **{agent_mae:.2f}**")
            line_parts.append(f"매니저 MAE = **{mgr_mae:.2f}**")
        st.caption(" · ".join(line_parts) + "  (일자별 |예측−정답| 평균)")


# ---------------------------------------------------------------- chat


def _render_chat_panel(bundle: RunBundle, overrides: ManagerOverrides) -> None:
    st.markdown("#### 💬 어시스턴트")
    st.caption("자연어로 질문·요청 (예: \"슈퍼볼 보정 빼줘\", \"15일 값 35로 바꿔줘\")")

    history: list[ChatTurn] = st.session_state.get("chat_history", [])
    for turn in history:
        with st.chat_message("user" if turn.role == "user" else "assistant"):
            st.markdown(turn.content)

    user_msg = st.chat_input("메시지 입력")
    if user_msg:
        history.append(ChatTurn(role="user", content=user_msg))
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                try:
                    reply, actions = chat_step(history[:-1], user_msg, bundle, overrides)
                except Exception as exc:  # noqa: BLE001
                    reply = f"오류: {exc}"
                    actions = []
            if actions:
                st.success("적용한 변경: " + " · ".join(actions))
            st.markdown(reply or "_(빈 응답)_")
        history.append(ChatTurn(role="assistant", content=reply))
        st.session_state["chat_history"] = history
        st.session_state["last_actions"] = actions
        st.rerun()


# ---------------------------------------------------------------- dashboard body


def _render_post_analysis(bundle: RunBundle) -> None:
    overrides = _ensure_overrides(bundle)
    manager_values, applied_labels = compute_manager_final(bundle, overrides)

    left, right = st.columns([6, 4], gap="large")

    with left:
        tab_edit, tab_signals, tab_report = st.tabs(
            ["일자별 수정", "보정 신호", "보고서"]
        )

        with tab_edit:
            global_m = st.slider(
                "글로벌 multiplier (전체 일자에 곱)",
                min_value=0.50,
                max_value=2.00,
                value=float(overrides.global_multiplier),
                step=0.01,
                key="global_mult_slider",
            )
            overrides.global_multiplier = float(global_m)

            render_edit_table(bundle, overrides, manager_values, applied_labels)

            btn_cols = st.columns([1, 1, 4])
            if btn_cols[0].button("셀 직접입력 해제", width="stretch"):
                overrides.cell_overrides.clear()
                st.rerun()
            if btn_cols[1].button("매니저 조작 초기화", width="stretch"):
                overrides.reset(bundle)
                st.rerun()

            st.divider()
            new_mgr_values, new_labels = compute_manager_final(bundle, overrides)
            saved_path = None
            if st.button("✅ 최종 컨펌 및 저장", type="primary", width="stretch"):
                saved_path = _save_manager_final(
                    bundle, overrides, new_mgr_values, new_labels
                )
            if saved_path is not None:
                st.success(f"저장됨: `{saved_path.relative_to(DATA_DIR.parent)}`")

        with tab_signals:
            render_insights_panel(bundle, overrides)

        with tab_report:
            if bundle.report_md:
                st.markdown(bundle.report_md)
            else:
                st.info("보고서 파일이 없습니다.")

    with right:
        _render_chat_panel(bundle, overrides)


# ---------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(
        page_title="수요예측 검토",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("📊 수요예측 검토 시스템")

    # Honor a pending load from the sidebar expander.
    pending = st.session_state.pop("pending_load", None)
    if pending is not None:
        run_id_pl, sku_pl = pending
        try:
            bundle_pl = load_bundle(run_id_pl, sku_pl)
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        # Align sidebar inputs with the loaded bundle so the package matches.
        st.session_state["sidebar_sku"] = bundle_pl.sku_id
        st.session_state["sidebar_date"] = datetime.strptime(
            bundle_pl.input_dates[-1], "%Y-%m-%d"
        ).date()
        st.session_state["pkg_key"] = (
            bundle_pl.sku_id,
            bundle_pl.input_dates[-1],
        )
        # Build a package compatible with the loaded bundle.
        st.session_state["package"] = build_prediction_package(
            bundle_pl.sku_id,
            datetime.strptime(bundle_pl.input_dates[-1], "%Y-%m-%d").date(),
        )
        st.session_state["active_run_id"] = run_id_pl
        st.session_state.pop("loaded_key", None)
        st.session_state.pop("overrides", None)
        st.session_state.pop("chat_history", None)
        st.rerun()

    sku_id, input_end_date = render_sidebar()

    try:
        pkg = _ensure_package(sku_id, input_end_date)
    except Exception as exc:  # noqa: BLE001
        st.error(f"PredictionPackage 생성 실패: {exc}")
        return

    active_run_id = st.session_state.get("active_run_id")
    bundle = None
    if active_run_id is not None:
        try:
            bundle = load_bundle(active_run_id, pkg.sku_id)
        except FileNotFoundError:
            st.session_state.pop("active_run_id", None)
            bundle = None

    # Header
    manager_values_for_header = None
    if bundle is not None:
        overrides_preview = _ensure_overrides(bundle)
        mvs, _ = compute_manager_final(bundle, overrides_preview)
        manager_values_for_header = mvs

    # Chart (always shown). Pre-analysis: 2 traces. Post-analysis: 4 traces.
    # The "정답 (test 실측)" trace is added when test data exists for the
    # forecast horizon; hidden by default, click the legend to reveal.
    in_dates = [str(d) for d in pkg.input_window.dates]
    fc_dates = [str(d) for d in pkg.forecast_window.dates]
    actual_values = load_actuals(pkg.sku_id, fc_dates)

    _render_header(pkg, bundle, manager_values_for_header, actual_values)
    if bundle is None:
        chart = build_chart(
            input_dates=in_dates,
            input_values=list(pkg.input_window.values),
            baseline_dates=fc_dates,
            baseline_values=list(pkg.forecast_window.values),
            actual_values=actual_values,
        )
    else:
        chart = build_chart(
            input_dates=in_dates,
            input_values=list(pkg.input_window.values),
            baseline_dates=fc_dates,
            baseline_values=list(pkg.forecast_window.values),
            agent_values=list(bundle.agent_final_values),
            manager_values=manager_values_for_header,
            actual_values=actual_values,
        )
    st.plotly_chart(chart, width="stretch")

    # Action zone
    if bundle is None:
        st.info(
            "위는 시계열 모델의 baseline 예측입니다. 아래 버튼을 누르면 4개 AI 에이전트가 "
            "차례로 baseline 을 분석하고 보정 신호를 제안합니다."
        )
        if st.button("▶ AI 에이전트 분석 시작", type="primary", width="stretch"):
            try:
                run_id = run_analysis_with_cascade(pkg, sku_id, input_end_date)
            except Exception as exc:  # noqa: BLE001
                st.error(f"분석 실패: {exc}")
                return
            st.session_state["active_run_id"] = run_id
            st.rerun()
        return

    # Post-analysis controls
    cols = st.columns([1, 5])
    if cols[0].button("🔄 다시 분석", width="stretch"):
        st.session_state.pop("active_run_id", None)
        st.session_state.pop("loaded_key", None)
        st.session_state.pop("overrides", None)
        st.session_state.pop("chat_history", None)
        st.rerun()

    _render_post_analysis(bundle)


if __name__ == "__main__":
    main()
