"""Chat assistant for the review app.

A small OpenAI function-calling loop. The LLM has read access to the
report and selected insights, and write access to manager overrides via
tool calls. Each turn returns:
  * `text`              — assistant message to display in the chat,
  * `applied_actions`   — short human-readable list of mutations done.

The Streamlit page renders both, then re-renders chart/table from the
mutated overrides.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openai import OpenAI

from src.app.loaders import RunBundle
from src.app.state import ManagerOverrides

CHAT_MODEL = os.getenv("CHAT_AGENT_MODEL", "gpt-4o-mini")


_SYSTEM_PROMPT = """\
당신은 수요예측 검토 매니저를 돕는 어시스턴트입니다. 매니저는 한 SKU 의 28일
forecast 를 검토 중이며, 에이전트가 baseline forecast 에 보정 신호 (insight) 를
이미 적용한 상태입니다. 매니저가 그 결과를 받아들이거나, 일부 신호를 끄고 켜고
multiplier 를 조정하거나, 특정 일자의 값을 직접 입력해 최종 예측을 만듭니다.

당신의 역할:
1. 매니저의 질문에 보고서·신호 데이터를 근거로 *간결하게* 답한다.
2. 매니저가 보정 변경을 *요청* 하면 (\"슈퍼볼 보정 빼줘\", \"15일 35로 바꿔\",
   \"전체 5% 높여줘\" 등) 해당 도구를 호출해 적용한다. 적용 후 어떤 변경을 했는지
   매니저 언어로 1-2 줄 요약한다.
3. 매니저가 명시적으로 요청하지 않은 변경은 *임의로 하지 않는다*. 제안만 가능.

출력 언어: 한국어. 도구 호출은 필요한 만큼만.
"""


def _bundle_brief(bundle: RunBundle, overrides: ManagerOverrides) -> str:
    lines = [
        f"SKU: {bundle.sku_id} (cat={bundle.metadata.get('cat_id')}, "
        f"store={bundle.metadata.get('store_id')}, state={bundle.metadata.get('state_id')})",
        f"forecast 기간: {bundle.baseline_dates[0]} ~ {bundle.baseline_dates[-1]}",
        f"baseline 합계: {sum(bundle.baseline_values):.1f}, "
        f"에이전트 보정 합계: {sum(bundle.agent_final_values):.1f}",
        "",
        "## 현재 신호 상태",
    ]
    if not bundle.insights:
        lines.append("(신호 없음)")
    for ins in bundle.insights:
        enabled = overrides.insight_enabled.get(ins.id, True)
        m = overrides.insight_multiplier.get(ins.id, ins.multiplier)
        lines.append(
            f"- id={ins.id} type={ins.type} ×{m:.2f} {'ON' if enabled else 'OFF'} "
            f"dates={ins.dates} reason={ins.reason}"
        )
    lines.extend(
        [
            "",
            f"## 글로벌 multiplier: ×{overrides.global_multiplier:.2f}",
            "",
            f"## 일자별 셀 override: {overrides.cell_overrides or '(없음)'}",
        ]
    )
    return "\n".join(lines)


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "toggle_insight",
            "description": "특정 신호를 켜거나 끈다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "insight_id": {"type": "string"},
                    "enabled": {"type": "boolean"},
                },
                "required": ["insight_id", "enabled"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_insight_multiplier",
            "description": "특정 신호의 multiplier 값을 변경한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "insight_id": {"type": "string"},
                    "multiplier": {"type": "number"},
                },
                "required": ["insight_id", "multiplier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_global_multiplier",
            "description": "모든 일자에 적용되는 글로벌 multiplier 를 설정한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "multiplier": {"type": "number"},
                },
                "required": ["multiplier"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_day_value",
            "description": "특정 일자의 매니저 최종값을 직접 지정한다 (compound 무시).",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "YYYY-MM-DD"},
                    "value": {"type": "number"},
                },
                "required": ["date", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear_day_override",
            "description": "특정 일자의 직접 입력값을 제거하고 compound 로 되돌린다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string"},
                },
                "required": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_all",
            "description": "모든 매니저 조작을 초기 상태(에이전트 결과)로 되돌린다.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant"
    content: str


def _apply_tool_call(
    name: str,
    args: dict,
    bundle: RunBundle,
    overrides: ManagerOverrides,
    actions: list[str],
) -> str:
    valid_ids = {ins.id for ins in bundle.insights}
    valid_dates = set(bundle.baseline_dates)

    if name == "toggle_insight":
        ins_id = str(args.get("insight_id"))
        if ins_id not in valid_ids:
            return json.dumps({"ok": False, "error": f"unknown insight_id: {ins_id}"})
        overrides.insight_enabled[ins_id] = bool(args.get("enabled"))
        actions.append(
            f"신호 `{ins_id}` {'적용' if overrides.insight_enabled[ins_id] else '제외'}"
        )
        return json.dumps({"ok": True})

    if name == "set_insight_multiplier":
        ins_id = str(args.get("insight_id"))
        if ins_id not in valid_ids:
            return json.dumps({"ok": False, "error": f"unknown insight_id: {ins_id}"})
        m = float(args.get("multiplier"))
        m = max(0.30, min(3.00, m))
        overrides.insight_multiplier[ins_id] = m
        actions.append(f"신호 `{ins_id}` multiplier ×{m:.2f}")
        return json.dumps({"ok": True, "multiplier": m})

    if name == "set_global_multiplier":
        m = float(args.get("multiplier"))
        m = max(0.50, min(2.00, m))
        overrides.global_multiplier = m
        actions.append(f"글로벌 multiplier ×{m:.2f}")
        return json.dumps({"ok": True, "multiplier": m})

    if name == "set_day_value":
        d = str(args.get("date"))
        if d not in valid_dates:
            return json.dumps({"ok": False, "error": f"date {d} not in forecast window"})
        v = max(0.0, float(args.get("value")))
        overrides.cell_overrides[d] = v
        actions.append(f"{d} 값 = {v:.1f}")
        return json.dumps({"ok": True, "value": v})

    if name == "clear_day_override":
        d = str(args.get("date"))
        overrides.cell_overrides.pop(d, None)
        actions.append(f"{d} 직접입력 해제")
        return json.dumps({"ok": True})

    if name == "reset_all":
        overrides.reset(bundle)
        actions.append("모든 매니저 조작 초기화")
        return json.dumps({"ok": True})

    return json.dumps({"ok": False, "error": f"unknown tool: {name}"})


def chat_step(
    history: list[ChatTurn],
    user_message: str,
    bundle: RunBundle,
    overrides: ManagerOverrides,
) -> tuple[str, list[str]]:
    """Run one chat turn. Returns (assistant_text, applied_action_log)."""
    client = OpenAI()

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "system",
            "content": (
                "## 보고서 (한국어 매니저용)\n\n"
                + (bundle.report_md or "(없음)")
                + "\n\n## 현재 상태\n\n"
                + _bundle_brief(bundle, overrides)
            ),
        },
    ]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": user_message})

    actions: list[str] = []
    for _ in range(6):  # tool-loop guard
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "", actions

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            }
        )
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _apply_tool_call(tc.function.name, args, bundle, overrides, actions)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

    return "(도구 호출 한도 초과)", actions
