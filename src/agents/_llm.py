"""LLM 클라이언트 factory. 모든 에이전트가 공유."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import OPENAI_MODEL, OPENAI_TEMPERATURE


def get_llm(model: str | None = None, temperature: float | None = None) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE if temperature is None else temperature,
    )
