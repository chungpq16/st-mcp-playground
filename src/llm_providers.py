"""
LLM Farm Integration for MCP Client.

This module provides interface for using LLM Farm with OpenAI-compatible API.
Supports both direct OpenAI and LLM Farm configurations.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Try to import langchain_openai with fallback
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
    print("✅ LangChain OpenAI available")
except ImportError as e:
    print(f"⚠️ LangChain OpenAI not available: {e}")
    print("💡 Please install: pip install langchain-openai")
    OPENAI_AVAILABLE = False
    
    # Create a mock ChatOpenAI class for graceful degradation
    class ChatOpenAI:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "langchain_openai is not installed. "
                "Please install it with: pip install langchain-openai"
            )

# Load environment variables
load_dotenv()


def create_llm_model(
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    timeout: Optional[float] = None,
    system_prompt: Optional[str] = None,
    streaming: bool = True,
    model: str = "gpt-4o-mini",
    **kwargs  # Additional parameters for compatibility
) -> ChatOpenAI:
    """
    Create LLM model using LLM Farm or OpenAI.
    
    Args:
        api_key: API key (will use env vars if not provided)
        temperature: Model temperature (0.0-2.0)
        max_tokens: Maximum tokens to generate
        timeout: Request timeout in seconds
        system_prompt: System prompt to use
        streaming: Enable streaming responses
        model: Model name to use
        **kwargs: Additional parameters
        
    Returns:
        Configured ChatOpenAI instance
        
    Raises:
        ImportError: If langchain_openai is not installed
        ValueError: If API key is not provided
    """
    # Check if OpenAI is available
    if not OPENAI_AVAILABLE:
        raise ImportError(
            "langchain_openai is not installed. "
            "Please install the required packages:\n"
            "pip install langchain-openai python-dotenv\n"
            "Or install all requirements:\n"
            "pip install -r requirements.txt"
        )
    
    # Get configuration from environment variables
    llm_farm_url = os.getenv('LLM_FARM_URL')
    env_api_key = os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY')
    
    # Use provided API key or fall back to environment
    final_api_key = api_key or env_api_key
    
    if not final_api_key:
        raise ValueError(
            "API key not found. Please provide api_key parameter or set "
            "API_KEY/OPENAI_API_KEY in environment variables"
        )
    
    # Configure parameters
    params = {
        "model": model,
        "openai_api_key": final_api_key,
        "temperature": temperature,
        "streaming": streaming,
    }
    
    # Add optional parameters
    if max_tokens:
        params["max_tokens"] = max_tokens
    
    if timeout:
        params["timeout"] = timeout
    
    # Configure for LLM Farm if URL is provided
    if llm_farm_url:
        params["openai_api_base"] = llm_farm_url
        params["default_headers"] = {
            "genaiplatform-farm-subscription-key": final_api_key
        }
        # Add API version for LLM Farm
        params["model_kwargs"] = {
            "extra_query": {"api-version": "2024-08-01-preview"}
        }
        print(f"🚜 Using LLM Farm: {llm_farm_url}")
        print(f"📋 Model: {model}")
    else:
        print(f"🤖 Using OpenAI directly")
        print(f"📋 Model: {model}")
    
    # Create the LLM instance
    llm = ChatOpenAI(**params)
    
    # Attach system prompt if provided
    if system_prompt:
        llm._system_prompt = system_prompt
    
    return llm