"""
Hardcoded OpenAI GPT-4o-mini configuration.

This module provides a simplified interface for using only OpenAI's GPT-4o-mini model.
"""

from typing import Optional
from langchain_openai import ChatOpenAI


def create_llm_model(
    api_key: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    system_prompt: Optional[str] = None,
    streaming: bool = True,
    **kwargs  # Ignore any additional parameters for compatibility
) -> ChatOpenAI:
    """Create a hardcoded GPT-4o-mini model."""
    params = {
        "model": "gpt-4o-mini",
        "openai_api_key": api_key,
        "temperature": temperature,
        "streaming": streaming,
    }
    
    if max_tokens:
        params["max_tokens"] = max_tokens
    
    if timeout:
        params["timeout"] = timeout
    
    llm = ChatOpenAI(**params)
    
    if system_prompt:
        llm._system_prompt = system_prompt
    
    return llm