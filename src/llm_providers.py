"""
LLM Farm Integration for MCP Client.

This module provides interface for using LLM Farm with OpenAI-compatible API.
Supports both direct OpenAI and LLM Farm configurations.
"""

import os
from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

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
    """
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


def health_check() -> bool:
    """
    Check if the configured LLM service is accessible.
    
    Returns:
        True if healthy, False otherwise
    """
    try:
        # Create a test model instance
        test_llm = create_llm_model()
        
        # Simple test completion
        from langchain_core.messages import HumanMessage
        response = test_llm.invoke([HumanMessage(content="Hello")])
        
        print(f"✅ LLM Health Check: SUCCESS")
        return True
        
    except Exception as e:
        print(f"❌ LLM Health Check: FAILED - {str(e)}")
        return False


def get_llm_info() -> dict:
    """
    Get information about the current LLM configuration.
    
    Returns:
        Dictionary with LLM configuration details
    """
    llm_farm_url = os.getenv('LLM_FARM_URL')
    api_key = os.getenv('API_KEY') or os.getenv('OPENAI_API_KEY')
    
    return {
        "provider": "LLM Farm" if llm_farm_url else "OpenAI",
        "base_url": llm_farm_url or "https://api.openai.com/v1",
        "api_key_configured": bool(api_key),
        "api_key_preview": f"{api_key[:8]}..." if api_key else "Not configured"
    }