"""
Factory that returns a LangChain `BaseChatModel` for the configured provider.

Every provider here supports `bind_tools()` natively, so the rest of the
LLM pipeline (system prompts, tool dispatch, history conversion) is provider-
agnostic. To add a new provider, add an enum value in `config.py` and a
branch here — nothing else has to change.

Imports are done lazily inside each branch so that users who only install
the optional SDK they actually need don't pay an ImportError penalty for
providers they don't use.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import LLMProvider, Settings

logger = logging.getLogger(__name__)


def build_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider
    logger.info(
        "Initialising LLM: provider=%s model=%s",
        provider.value,
        settings.active_model,
    )

    if provider == LLMProvider.OLLAMA:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_ctx=settings.ollama_num_ctx,
        )

    if provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.llm_temperature,
        )

    if provider == LLMProvider.OPENAI:
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": settings.openai_model,
            "api_key": settings.openai_api_key,
            "temperature": settings.llm_temperature,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)

    if provider == LLMProvider.GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=settings.llm_temperature,
        )

    if provider == LLMProvider.GROQ:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")
