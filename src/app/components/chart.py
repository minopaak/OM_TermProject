"""Plotly time-series chart.

Two-trace mode (pre-analysis): input + baseline.
Four-trace mode (post-analysis): input + baseline + agent-final + manager-final.
"""
from __future__ import annotations

import plotly.graph_objects as go


def build_chart(
    input_dates: list,
    input_values: list[float],
    baseline_dates: list,
    baseline_values: list[float],
    agent_values: list[float] | None = None,
    manager_values: list[float] | None = None,
    actual_values: list[float] | None = None,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=input_dates,
            y=input_values,
            name="과거 실적 (input 28일)",
            mode="lines+markers",
            line=dict(color="#2563EB", width=2),
            marker=dict(size=5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_dates,
            y=baseline_values,
            name="모델 baseline",
            mode="lines+markers",
            line=dict(color="#94A3B8", width=2, dash="dot"),
            marker=dict(size=5),
        )
    )
    if agent_values is not None:
        fig.add_trace(
            go.Scatter(
                x=baseline_dates,
                y=agent_values,
                name="에이전트 보정",
                mode="lines+markers",
                line=dict(color="#F59E0B", width=2, dash="dash"),
                marker=dict(size=5),
            )
        )
    if manager_values is not None:
        fig.add_trace(
            go.Scatter(
                x=baseline_dates,
                y=manager_values,
                name="매니저 최종",
                mode="lines+markers",
                line=dict(color="#16A34A", width=3),
                marker=dict(size=7),
            )
        )
    if actual_values is not None:
        fig.add_trace(
            go.Scatter(
                x=baseline_dates,
                y=actual_values,
                name="정답 (test 실측)",
                mode="lines+markers",
                line=dict(color="#DC2626", width=2.5),
                marker=dict(size=6, symbol="diamond"),
                visible="legendonly",  # 범례 클릭 시 표시
            )
        )

    if input_dates and baseline_dates:
        boundary = baseline_dates[0]
        fig.add_shape(
            type="line",
            x0=boundary,
            x1=boundary,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(color="#475569", width=1, dash="dot"),
        )
        fig.add_annotation(
            x=boundary,
            y=1.0,
            xref="x",
            yref="paper",
            text="forecast 시작",
            showarrow=False,
            yshift=10,
            font=dict(size=11, color="#475569"),
        )

    fig.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="", tickangle=-45),
        yaxis=dict(title="판매량 (units)"),
        hovermode="x unified",
    )
    return fig
