"""Manager override state + manager-final recompute.

`ManagerOverrides` lives in Streamlit `session_state`. Every UI element
(insight toggle, slider, edit-table cell, chat tool-call) mutates this
single object and the chart/table re-render from the recomputed values.

Recompute order per forecast day d:
  1. if d in cell_overrides → use that absolute value (bypasses cap),
  2. otherwise factor = global_multiplier × ∏ insight_multiplier[id]
     for every enabled insight that includes d,
  3. clip factor to [_FACTOR_MIN, _FACTOR_MAX] and multiply baseline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.app.loaders import Insight, RunBundle

_FACTOR_MIN = 0.50
_FACTOR_MAX = 2.00


@dataclass
class ManagerOverrides:
    insight_enabled: dict[str, bool] = field(default_factory=dict)
    insight_multiplier: dict[str, float] = field(default_factory=dict)
    cell_overrides: dict[str, float] = field(default_factory=dict)
    global_multiplier: float = 1.0

    @classmethod
    def from_bundle(cls, bundle: RunBundle) -> "ManagerOverrides":
        return cls(
            insight_enabled={ins.id: True for ins in bundle.insights},
            insight_multiplier={ins.id: ins.multiplier for ins in bundle.insights},
        )

    def reset(self, bundle: RunBundle) -> None:
        self.insight_enabled = {ins.id: True for ins in bundle.insights}
        self.insight_multiplier = {ins.id: ins.multiplier for ins in bundle.insights}
        self.cell_overrides.clear()
        self.global_multiplier = 1.0


def compute_manager_final(
    bundle: RunBundle,
    overrides: ManagerOverrides,
) -> tuple[list[float], list[str]]:
    """Return (per-day final values, per-day applied label)."""
    insights_by_id: dict[str, Insight] = {i.id: i for i in bundle.insights}
    finals: list[float] = []
    labels: list[str] = []
    for d, base in zip(bundle.baseline_dates, bundle.baseline_values):
        if d in overrides.cell_overrides:
            finals.append(float(overrides.cell_overrides[d]))
            labels.append("manual")
            continue

        factor = overrides.global_multiplier
        applied: list[str] = []
        for ins_id, ins in insights_by_id.items():
            if not overrides.insight_enabled.get(ins_id, True):
                continue
            if d not in ins.dates:
                continue
            m = float(overrides.insight_multiplier.get(ins_id, ins.multiplier))
            factor *= m
            sign = "+" if m >= 1.0 else "-"
            applied.append(f"{ins.type} {sign}{abs(m - 1.0)*100:.0f}%")
        factor = max(_FACTOR_MIN, min(_FACTOR_MAX, factor))
        finals.append(max(0.0, base * factor))
        labels.append(" × ".join(applied) if applied else "")
    return finals, labels
