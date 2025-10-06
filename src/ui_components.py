"""
UI components and interface functions for the LangChain MCP Client.

This module contains all the user interface components including
sidebar, tabs, and various UI utility functions.
"""

import streamlit as st
import json
import datetime
import traceback
import aiohttp
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple

from .database import PersistentStorageManager
from .llm_providers import create_llm_model
from .mcp_client import MCPConnectionManager
from .agent_manager import create_agent_with_tools
from .utils import run_async, reset_connection_state, format_error_message, model_supports_tools, create_download_data

def render_sidebar():
    """Render the main application sidebar with all configuration options."""
    with st.sidebar:
        st.title("LangChain MCP Client")
        st.divider()
        st.header("Configuration")
        
        # Load environment variables and set up API key silently
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Store API key in session state for internal use
        st.session_state.api_key = api_key or ""
        st.session_state.llm_provider = "OpenAI"
        st.session_state.selected_model = "gpt-4o-mini"
        
        # Create a simple llm_config for other functions
        llm_config = {
            "provider": "OpenAI",
            "api_key": api_key or "",
            "model": "gpt-4o-mini"
        }
        
        # Set streaming default (always enabled for GPT-4o-mini)
        st.session_state.enable_streaming = True
        
        # Memory configuration
        memory_config = render_memory_configuration()
        
        # Store configs in session state for automatic MCP connection
        st.session_state.llm_config = llm_config
        st.session_state.memory_config = memory_config
        
        st.divider()
        
        # Display available tools
        render_available_tools()

def render_memory_configuration() -> Dict:
    """Render memory configuration section."""
    st.header("Memory Settings")
    
    memory_enabled = st.checkbox(
        "Enable Conversation Memory",
        value=st.session_state.get('memory_enabled', False),
        help="Enable persistent conversation memory across interactions",
        key="sidebar_memory_enabled"
    )
    st.session_state.memory_enabled = memory_enabled
    
    memory_config = {"enabled": memory_enabled}
    
    if memory_enabled:
        # Memory type selection
        memory_type = st.selectbox(
            "Memory Type",
            options=["Short-term (Session)", "Persistent (Cross-session)"],
            index=0 if st.session_state.get('memory_type', 'Short-term (Session)') == 'Short-term (Session)' else 1,
            help="Short-term: Remembers within current session\nPersistent: Remembers across sessions using SQLite database",
        )
        st.session_state.memory_type = memory_type
        memory_config["type"] = memory_type
        
        # Initialize persistent storage if needed
        if memory_type == "Persistent (Cross-session)":
            if 'persistent_storage' not in st.session_state:
                st.session_state.persistent_storage = PersistentStorageManager()
            
            render_persistent_storage_section()
        
        # Thread ID management
        thread_id = st.text_input(
            "Conversation ID",
            value=st.session_state.get('thread_id', 'default'),
            help="Unique identifier for this conversation thread"
        )
        st.session_state.thread_id = thread_id
        memory_config["thread_id"] = thread_id
        
        # Memory management options
        render_memory_management_section()
    
    # Reset connection when memory settings change
    if st.session_state.get('_last_memory_enabled') != memory_enabled:
        reset_connection_state()
        st.session_state._last_memory_enabled = memory_enabled
    
    return memory_config


def render_persistent_storage_section():
    """Render persistent storage configuration and management."""
    with st.expander("💾 Database Settings"):
        if hasattr(st.session_state, 'persistent_storage'):
            db_stats = st.session_state.persistent_storage.get_database_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Conversations", db_stats.get('conversation_count', 0))
                st.metric("Total Messages", db_stats.get('total_messages', 0))
            with col2:
                st.metric("Database Size", f"{db_stats.get('database_size_mb', 0)} MB")
                st.text(f"Path: {db_stats.get('database_path', 'N/A')}")
            
            # Conversation browser
            render_conversation_browser()


def render_conversation_browser():
    """Render conversation browser for persistent storage."""
    if not hasattr(st.session_state, 'persistent_storage'):
        return
    
    conversations = st.session_state.persistent_storage.list_conversations()
    if conversations:
        st.subheader("Saved Conversations")
        for conv in conversations[:5]:  # Show last 5 conversations
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    display_title = conv.get('title') or conv['thread_id']
                    if len(display_title) > 30:
                        display_title = display_title[:30] + "..."
                    st.write(f"**{display_title}**")
                    last_msg = conv.get('last_message', '')
                    if last_msg and len(last_msg) > 50:
                        last_msg = last_msg[:50] + "..."
                    st.caption(f"{conv.get('message_count', 0)} messages • {last_msg}")
                
                with col2:
                    if st.button("📂 Load", key=f"load_{conv['thread_id']}"):
                        st.session_state.thread_id = conv['thread_id']
                        st.session_state.chat_history = []
                        # Load conversation messages from database
                        if hasattr(st.session_state, 'persistent_storage'):
                            try:
                                loaded_messages = st.session_state.persistent_storage.load_conversation_messages(conv['thread_id'])
                                if loaded_messages:
                                    st.session_state.chat_history = loaded_messages
                            except Exception as e:
                                st.warning(f"Could not load conversation history: {str(e)}")
                        st.success(f"Loaded conversation: {conv['thread_id']}")
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Del", key=f"del_{conv['thread_id']}"):
                        if st.session_state.persistent_storage.delete_conversation(conv['thread_id']):
                            st.success("Conversation deleted")
                            st.rerun()
                
                st.divider()


def render_memory_management_section():
    """Render memory management options."""
    with st.expander("Memory Management"):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Memory"):
                if hasattr(st.session_state, 'checkpointer') and st.session_state.checkpointer:
                    try:
                        st.session_state.chat_history = []
                        if hasattr(st.session_state, 'agent') and st.session_state.agent:
                            st.session_state.agent = None
                        st.success("Memory cleared successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error clearing memory: {str(e)}")
        
        with col2:
            max_messages = st.number_input(
                "Max Messages",
                min_value=10,
                max_value=1000,
                value=st.session_state.get('max_messages', 100),
                help="Maximum messages to keep in memory"
            )
            st.session_state.max_messages = max_messages
        
        # Memory status
        if 'chat_history' in st.session_state:
            current_messages = len(st.session_state.chat_history)
            st.info(f"Current conversation: {current_messages} messages")
        
        # Persistent storage actions
        render_persistent_storage_actions()


def render_persistent_storage_actions():
    """Render persistent storage action buttons."""
    memory_type = st.session_state.get('memory_type', '')
    if (memory_type == "Persistent (Cross-session)" and 
        hasattr(st.session_state, 'persistent_storage')):
        
        st.subheader("Persistent Storage Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Current Conversation"):
                if st.session_state.chat_history:
                    # Generate a title from the first user message
                    title = None
                    for msg in st.session_state.chat_history[:3]:
                        if msg.get('role') == 'user':
                            title = msg.get('content', '')[:50] + "..." if len(msg.get('content', '')) > 50 else msg.get('content', '')
                            break
                    
                    thread_id = st.session_state.get('thread_id', 'default')
                    st.session_state.persistent_storage.update_conversation_metadata(
                        thread_id=thread_id,
                        title=title,
                        message_count=len(st.session_state.chat_history),
                        last_message=st.session_state.chat_history[-1].get('content', '') if st.session_state.chat_history else ''
                    )
                    st.success("Conversation metadata saved!")
                else:
                    st.warning("No conversation to save")
        
        with col2:
            if st.button("📤 Export Conversation"):
                thread_id = st.session_state.get('thread_id', 'default')
                export_data = st.session_state.persistent_storage.export_conversation(thread_id)
                if export_data:
                    json_str, filename = create_download_data(export_data, f"conversation_{thread_id}")
                    st.download_button(
                        label="📁 Download Export",
                        data=json_str,
                        file_name=filename,
                        mime="application/json"
                    )
                else:
                                        st.error("Failed to export conversation")


def render_mcp_configuration_tab():
    """Render MCP server configuration tab with automatic connection."""
    st.header("🔗 MCP Server Configuration")
    
    # Get configs from session state
    llm_config = st.session_state.get('llm_config', {})
    memory_config = st.session_state.get('memory_config', {})
    
    # Load configuration from JSON file
    config_file_path = "mcp_config.json"
    mcp_servers_config = load_mcp_config(config_file_path)
    
    if mcp_servers_config:
        st.success(f"📁 Configuration loaded from {config_file_path}")
        
        # Display configured servers
        with st.expander("📋 Server Configuration Details", expanded=True):
            for server_name, server_config in mcp_servers_config.items():
                st.write(f"**{server_name}:**")
                if server_config.get('transport') == 'stdio':
                    st.write(f"  • Transport: STDIO")
                    st.write(f"  • Command: {server_config.get('command', 'N/A')}")
                    st.write(f"  • Args: {server_config.get('args', [])}")
                else:
                    st.write(f"  • Transport: {server_config.get('transport', 'http')}")
                    st.write(f"  • URL: {server_config.get('url', 'N/A')}")
        
        # Auto-connect to MCP servers
        st.info("🔄 Automatically connecting to MCP servers...")
        
        # Check if already connected
        mcp_manager = st.session_state.get('mcp_manager')
        if not mcp_manager or not mcp_manager.is_connected:
            # Attempt automatic connection
            connection_result = handle_json_config_connection(llm_config, memory_config, mcp_servers_config)
            
            if connection_result.get('connected', False):
                st.success("✅ Successfully connected to MCP servers!")
            else:
                st.error("❌ Failed to connect automatically. Please check your configuration.")
                
                # Manual connect button as fallback
                if st.button("🔄 Retry Connection", type="secondary"):
                    st.rerun()
        else:
            st.success("✅ Already connected to MCP servers!")
            
            # Show connection details
            tools_count = len(st.session_state.get('tools', []))
            st.info(f"🔧 {tools_count} tools available from {len(mcp_servers_config)} servers")
            
            # Option to reconnect
            if st.button("🔄 Reconnect", type="secondary"):
                # Reset connection
                if mcp_manager:
                    try:
                        run_async(lambda: mcp_manager.close())
                    except Exception:
                        pass
                st.session_state.mcp_manager = None
                st.session_state.tools = []
                st.rerun()
    
    else:
        # Show error if no config file
        st.error(f"❌ No MCP configuration found at {config_file_path}")
        st.info("Please create a valid MCP configuration file to use this application.")
        
        with st.expander("📝 How to create configuration", expanded=True):
            st.write("Create a `mcp_config.json` file with your MCP server configurations:")
            st.code('''
{
  "weather": {
    "url": "http://localhost:8000/sse",
    "transport": "sse",
    "timeout": 30
  },
  "math": {
    "command": "python",
    "args": ["/path/to/math_server.py"],
    "transport": "stdio"
  }
}
            ''', language="json")
            
            st.write("**Supported transport types:**")
            st.write("• `stdio` - For local Python scripts")
            st.write("• `sse` - For Server-Sent Events HTTP endpoints")
            st.write("• `streamable_http` - For HTTP streaming endpoints")


def render_mcp_configuration_tab():
    """Render MCP server configuration tab with automatic connection."""
    st.header("🔗 MCP Server Configuration")
    
    # Get configs from session state
    llm_config = st.session_state.get('llm_config', {})
    memory_config = st.session_state.get('memory_config', {})
    
    # Load configuration from JSON file
    config_file_path = "mcp_config.json"
    mcp_servers_config = load_mcp_config(config_file_path)
    
    if mcp_servers_config:
        st.success(f"📁 Configuration loaded from {config_file_path}")
        
        # Display configured servers
        with st.expander("📋 Server Configuration Details", expanded=True):
            for server_name, server_config in mcp_servers_config.items():
                st.write(f"**{server_name}:**")
                if server_config.get('transport') == 'stdio':
                    st.write(f"  • Transport: STDIO")
                    st.write(f"  • Command: {server_config.get('command', 'N/A')}")
                    st.write(f"  • Args: {server_config.get('args', [])}")
                else:
                    st.write(f"  • Transport: {server_config.get('transport', 'http')}")
                    st.write(f"  • URL: {server_config.get('url', 'N/A')}")
        
        # Auto-connect to MCP servers
        st.info("🔄 Automatically connecting to MCP servers...")
        
        # Check if already connected
        mcp_manager = st.session_state.get('mcp_manager')
        if not mcp_manager or not mcp_manager.is_connected:
            # Attempt automatic connection
            connection_result = handle_json_config_connection(llm_config, memory_config, mcp_servers_config)
            
            if connection_result.get('connected', False):
                st.success("✅ Successfully connected to MCP servers!")
            else:
                st.error("❌ Failed to connect automatically. Please check your configuration.")
                
                # Manual connect button as fallback
                if st.button("🔄 Retry Connection", type="secondary"):
                    st.rerun()
        else:
            st.success("✅ Already connected to MCP servers!")
            
            # Show connection details
            tools_count = len(st.session_state.get('tools', []))
            st.info(f"🔧 {tools_count} tools available from {len(mcp_servers_config)} servers")
            
            # Option to reconnect
            if st.button("🔄 Reconnect", type="secondary"):
                # Reset connection
                if mcp_manager:
                    try:
                        run_async(lambda: mcp_manager.close())
                    except Exception:
                        pass
                st.session_state.mcp_manager = None
                st.session_state.tools = []
                st.rerun()
    
    else:
        # Show error if no config file
        st.error(f"❌ No MCP configuration found at {config_file_path}")
        st.info("Please create a valid MCP configuration file to use this application.")
        
        with st.expander("📝 How to create configuration", expanded=True):
            st.write("Create a `mcp_config.json` file with your MCP server configurations:")
            st.code('''
{
  "weather": {
    "url": "http://localhost:8000/sse",
    "transport": "sse",
    "timeout": 30
  },
  "math": {
    "command": "python",
    "args": ["/path/to/math_server.py"],
    "transport": "stdio"
  }
}
            ''', language="json")
            
            st.write("**Supported transport types:**")
            st.write("• `stdio` - For local Python scripts")
            st.write("• `sse` - For Server-Sent Events HTTP endpoints")
            st.write("• `streamable_http` - For HTTP streaming endpoints")


def render_mcp_configuration_tab():
    """Render MCP server configuration tab with automatic connection."""
    st.header("🔗 MCP Server Configuration")
    
    # Get configs from session state
    llm_config = st.session_state.get('llm_config', {})
    memory_config = st.session_state.get('memory_config', {})
    
    # Load configuration from JSON file
    config_file_path = "mcp_config.json"
    mcp_servers_config = load_mcp_config(config_file_path)
    
    if mcp_servers_config:
        st.success(f"📁 Configuration loaded from {config_file_path}")
        
        # Display configured servers
        with st.expander("📋 Server Configuration Details", expanded=True):
            for server_name, server_config in mcp_servers_config.items():
                st.write(f"**{server_name}:**")
                if server_config.get('transport') == 'stdio':
                    st.write(f"  • Transport: STDIO")
                    st.write(f"  • Command: {server_config.get('command', 'N/A')}")
                    st.write(f"  • Args: {server_config.get('args', [])}")
                else:
                    st.write(f"  • Transport: {server_config.get('transport', 'http')}")
                    st.write(f"  • URL: {server_config.get('url', 'N/A')}")
        
        # Auto-connect to MCP servers
        st.info("🔄 Automatically connecting to MCP servers...")
        
        # Check if already connected
        mcp_manager = st.session_state.get('mcp_manager')
        if not mcp_manager or not mcp_manager.is_connected:
            # Attempt automatic connection
            connection_result = handle_json_config_connection(llm_config, memory_config, mcp_servers_config)
            
            if connection_result.get('connected', False):
                st.success("✅ Successfully connected to MCP servers!")
            else:
                st.error("❌ Failed to connect automatically. Please check your configuration.")
                
                # Manual connect button as fallback
                if st.button("🔄 Retry Connection", type="secondary"):
                    st.rerun()
        else:
            st.success("✅ Already connected to MCP servers!")
            
            # Show connection details
            tools_count = len(st.session_state.get('tools', []))
            st.info(f"🔧 {tools_count} tools available from {len(mcp_servers_config)} servers")
            
            # Option to reconnect
            if st.button("🔄 Reconnect", type="secondary"):
                # Reset connection
                if mcp_manager:
                    try:
                        run_async(lambda: mcp_manager.close())
                    except Exception:
                        pass
                st.session_state.mcp_manager = None
                st.session_state.tools = []
                st.rerun()
    
    else:
        # Show error if no config file
        st.error(f"❌ No MCP configuration found at {config_file_path}")
        st.info("Please create a valid MCP configuration file to use this application.")
        
        with st.expander("📝 How to create configuration", expanded=True):
            st.write("Create a `mcp_config.json` file with your MCP server configurations:")
            st.code('''
{
  "weather": {
    "url": "http://localhost:8000/sse",
    "transport": "sse",
    "timeout": 30
  },
  "math": {
    "command": "python",
    "args": ["/path/to/math_server.py"],
    "transport": "stdio"
  }
}
            ''', language="json")
            
            st.write("**Supported transport types:**")
            st.write("• `stdio` - For local Python scripts")
            st.write("• `sse` - For Server-Sent Events HTTP endpoints")
            st.write("• `streamable_http` - For HTTP streaming endpoints")


def render_server_configuration(llm_config: Dict, memory_config: Dict) -> Dict:
    """Render MCP server configuration section with hardcoded JSON config."""
    st.header("MCP Server Configuration")
    
    # Load configuration from JSON file
    config_file_path = "mcp_config.json"
    mcp_servers_config = load_mcp_config(config_file_path)
    
    if mcp_servers_config:
        st.info(f"📁 Loaded configuration from {config_file_path}")
        
        # Display configured servers
        with st.expander("📋 Server Configuration Details"):
            for server_name, server_config in mcp_servers_config.items():
                st.write(f"**{server_name}:**")
                if server_config.get('transport') == 'stdio':
                    st.write(f"  • Transport: STDIO")
                    st.write(f"  • Command: {server_config.get('command', 'N/A')}")
                    st.write(f"  • Args: {server_config.get('args', [])}")
                else:
                    st.write(f"  • Transport: {server_config.get('transport', 'http')}")
                    st.write(f"  • URL: {server_config.get('url', 'N/A')}")
        
        # Single button to connect to all configured servers
        if st.button("🚀 Connect to MCP Servers", type="primary"):
            return handle_json_config_connection(llm_config, memory_config, mcp_servers_config)
        
        return {"mode": "json_config", "connected": False}
    else:
        # Show error if no config file
        st.error(f"❌ No MCP configuration found at {config_file_path}")
        st.info("Please create a valid MCP configuration file to use this application.")
        
        with st.expander("� How to create configuration"):
            st.write("Create a `mcp_config.json` file with your MCP server configurations:")
            st.code('''
{
  "weather": {
    "url": "http://localhost:8000/sse",
    "transport": "sse",
    "timeout": 30
  },
  "math": {
    "command": "python",
    "args": ["/path/to/math_server.py"],
    "transport": "stdio"
  }
}
            ''', language="json")
        
        return {"mode": "no_config", "connected": False}


def render_server_configuration(llm_config: Dict, memory_config: Dict) -> Dict:
    """Legacy function - MCP configuration now handled in dedicated tab."""
    # This function is kept for compatibility but functionality moved to render_mcp_configuration_tab
    return {"mode": "tab_based", "connected": False}


def load_mcp_config(config_file_path: str) -> Optional[Dict]:
    """Load MCP server configuration from JSON file."""
    try:
        if not os.path.exists(config_file_path):
            # Create a default configuration file if it doesn't exist
            default_config = {
                "math": {
                    "command": "python",
                    "args": ["/path/to/math_server.py"],
                    "transport": "stdio"
                },
                "weather": {
                    "url": "http://localhost:8000/mcp",
                    "transport": "streamable_http"
                }
            }
            
            with open(config_file_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            st.info(f"📝 Created default configuration file at {config_file_path}")
            st.info("Please update the paths and URLs in the configuration file as needed.")
            return default_config
        
        with open(config_file_path, 'r') as f:
            config = json.load(f)
        
        if not config:
            st.warning("Configuration file is empty")
            return None
        
        # Validate configuration structure
        for server_name, server_config in config.items():
            if not isinstance(server_config, dict):
                st.error(f"Invalid configuration for server '{server_name}': must be an object")
                return None
            
            transport = server_config.get('transport', '')
            if transport == 'stdio':
                if 'command' not in server_config:
                    st.error(f"STDIO server '{server_name}' missing 'command' field")
                    return None
            elif transport in ['streamable_http', 'http', 'sse']:
                if 'url' not in server_config:
                    st.error(f"HTTP server '{server_name}' missing 'url' field")
                    return None
            else:
                st.error(f"Unknown transport type '{transport}' for server '{server_name}'")
                return None
        
        return config
        
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON in configuration file: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error loading configuration file: {str(e)}")
        return None


def handle_json_config_connection(llm_config: Dict, memory_config: Dict, mcp_servers_config: Dict) -> Dict:
    """Handle connection using JSON configuration file."""
    # Check API key requirement
    if not llm_config["api_key"]:
        st.error("❌ OpenAI API Key not found. Please add OPENAI_API_KEY to your .env file")
        return {"mode": "json_config", "connected": False}
    
    if not mcp_servers_config:
        st.error("No MCP servers configured")
        return {"mode": "json_config", "connected": False}
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        with st.spinner("Connecting to MCP servers..."):
            try:
                # Show progress steps
                progress_placeholder = st.empty()
                
                # Step 1: Initialize MCP connection manager
                progress_placeholder.info("🌐 Initializing MCP connection manager...")
                
                # Get or create the connection manager
                mcp_manager = MCPConnectionManager.get_instance()
                st.session_state.mcp_manager = mcp_manager
                
                # Step 2: Start connections with JSON config
                progress_placeholder.info("🔗 Connecting to configured servers...")
                try:
                    run_async(lambda: mcp_manager.start(mcp_servers_config))
                    if not mcp_manager.is_connected:
                        progress_placeholder.error("❌ Failed to start MCP connection manager")
                        return {"mode": "json_config", "connected": False}
                except Exception as e:
                    formatted_error = format_error_message(e)
                    progress_placeholder.error(f"❌ Failed to start MCP connection manager: {formatted_error}")
                    return {"mode": "json_config", "connected": False}
                
                # Step 3: Get tools
                progress_placeholder.info("🔍 Retrieving tools from servers...")
                try:
                    tools = run_async(lambda: mcp_manager.get_tools())
                    if tools is None:
                        tools = []
                except Exception as e:
                    formatted_error = format_error_message(e)
                    progress_placeholder.error(f"❌ Failed to retrieve tools: {formatted_error}")
                    return {"mode": "json_config", "connected": False}
                
                st.session_state.tools = tools
                
                # Step 4: Create agent
                progress_placeholder.info("🤖 Creating and configuring agent...")
                success = create_and_configure_agent(llm_config, memory_config, st.session_state.tools)
                
                if success:
                    progress_placeholder.empty()  # Clear progress messages
                    st.success(f"✅ Connected to {len(mcp_servers_config)} MCP servers! Found {len(st.session_state.tools)} tools.")
                    
                    with st.expander("🔧 Connection Details"):
                        st.write(f"**Servers connected:** {len(mcp_servers_config)}")
                        for server_name, server_config in mcp_servers_config.items():
                            transport = server_config.get('transport', 'unknown')
                            if transport == 'stdio':
                                st.write(f"  • {server_name}: STDIO ({server_config.get('command', 'N/A')})")
                            else:
                                st.write(f"  • {server_name}: {transport.upper()} ({server_config.get('url', 'N/A')})")
                        st.write(f"**Total tools found:** {len(st.session_state.tools)}")
                    
                    return {"mode": "json_config", "connected": True}
                else:
                    progress_placeholder.error("❌ Failed to configure agent")
                    return {"mode": "json_config", "connected": False}
                    
            except Exception as e:
                formatted_error = format_error_message(e)
                st.error(f"❌ Error connecting to MCP servers: {formatted_error}")
                
                # Show additional troubleshooting info
                with st.expander("🔍 Troubleshooting"):
                    st.write("**Common solutions:**")
                    st.write("• Check that all MCP servers are running and accessible")
                    st.write("• Verify all server URLs and paths in mcp_config.json are correct")
                    st.write("• For STDIO servers, ensure the command and script paths are valid")
                    st.write("• For HTTP servers, ensure they support the specified transport type")
                    
                    st.write("**Configured servers:**")
                    for server_name, server_config in mcp_servers_config.items():
                        transport = server_config.get('transport', 'unknown')
                        if transport == 'stdio':
                            st.write(f"  • {server_name}: {server_config.get('command', 'N/A')} {server_config.get('args', [])}")
                        else:
                            st.write(f"  • {server_name}: {server_config.get('url', 'N/A')}")
                    
                    st.write("**Technical details:**")
                    st.code(traceback.format_exc(), language="python")
                
                return {"mode": "json_config", "connected": False}

def create_and_configure_agent(llm_config: Dict, memory_config: Dict, mcp_tools: List) -> bool:
    """Create and configure the agent with the given parameters."""
    try:
        # Get configuration parameters from session state
        use_custom_config = st.session_state.get('config_use_custom_settings', False)
        
        if use_custom_config:
            # Use custom configuration parameters
            temperature = st.session_state.get('config_temperature', 0.7)
            max_tokens = st.session_state.get('config_max_tokens')
            timeout = st.session_state.get('config_timeout')
            system_prompt = st.session_state.get('config_system_prompt')
        else:
            # Use default parameters
            temperature = 0.7
            max_tokens = None
            timeout = None
            system_prompt = None
        
        # Create the language model with configuration
        llm = create_llm_model(
            api_key=llm_config["api_key"],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            system_prompt=system_prompt
        )
        
        # Get persistent storage if needed
        persistent_storage = None
        if (memory_config.get("enabled") and 
            memory_config.get("type") == "Persistent (Cross-session)" and
            hasattr(st.session_state, 'persistent_storage')):
            persistent_storage = st.session_state.persistent_storage
        
        # Create the agent
        agent, checkpointer = create_agent_with_tools(
            llm=llm,
            mcp_tools=mcp_tools,
            memory_enabled=memory_config.get("enabled", False),
            memory_type=memory_config.get("type", "Short-term (Session)"),
            persistent_storage=persistent_storage
        )
        
        st.session_state.agent = agent
        st.session_state.checkpointer = checkpointer
        
        return True
        
    except Exception as e:
        st.error(f"Error creating agent: {str(e)}")
        return False



def render_available_tools():
    """Render the available tools and prompts section with connection status."""
    # Show connection status
    mcp_manager = st.session_state.get('mcp_manager')
    if mcp_manager:
        if mcp_manager.is_connected:
            st.success("🟢 MCP Connection: Connected")
        elif mcp_manager.running:
            st.warning("🟡 MCP Connection: Reconnecting...")
        else:
            st.error("🔴 MCP Connection: Disconnected")
    else:
        st.info("⚫ MCP Connection: Not initialized")
    
    # Always show the tools and prompts header if we have a manager
    if mcp_manager or st.session_state.get('agent'):
        # Tools section
        render_tools_section()
        
        # Prompts section
        if mcp_manager and mcp_manager.is_connected:
            render_prompts_section()


def render_tools_section():
    """Render the tools section in sidebar."""
    st.header("Available Tools")
    
    # Add refresh button for tools
    mcp_manager = st.session_state.get('mcp_manager')
    if mcp_manager and st.button("🔄 Refresh Tools"):
        try:
            tools = run_async(lambda: mcp_manager.get_tools(force_refresh=True))
            st.session_state.tools = tools or []
            st.rerun()
        except Exception as e:
            st.error(f"Failed to refresh tools: {format_error_message(e)}")
    
    # Check if the current model supports tools
    model_name = st.session_state.get('selected_model', '')
    supports_tools = model_supports_tools(model_name)
    
    # Show total tool count including history tool
    mcp_tool_count = len(st.session_state.tools)
    memory_tool_count = 1 if st.session_state.get('memory_enabled', False) and supports_tools else 0
    total_tools = mcp_tool_count + memory_tool_count

    if st.session_state.tools or (st.session_state.agent and st.session_state.get('memory_enabled', False)):
        
        if not supports_tools and st.session_state.get('memory_enabled', False):
            st.info("🧠 Memory enabled (conversation history only - model doesn't support tool calling)")
            st.warning("⚠️ This model doesn't support tools, so the history tool is not available. Memory works through conversation context only.")
        elif mcp_tool_count > 0 and memory_tool_count > 0:
            st.info(f"🔧 {total_tools} tools available ({mcp_tool_count} MCP + {memory_tool_count} memory tool)")
        elif mcp_tool_count > 0:
            st.info(f"🔧 {mcp_tool_count} MCP tools available")
        elif memory_tool_count > 0:
            st.info(f"🔧 {memory_tool_count} memory tool available")
        else:
            st.info("📊 No MCP tools available")
        
        render_tool_selector()
    elif st.session_state.agent:
        # Agent exists but no tools - show appropriate message
        model_name = st.session_state.get('selected_model', '')
        if not model_supports_tools(model_name):
            st.info("💬 Simple chat mode - this model doesn't support tool calling")
        else:
            st.info("📊 No MCP tools available")


def render_prompts_section():
    """Render the prompts section in sidebar."""
    st.header("Available Prompts")
    
    mcp_manager = st.session_state.get('mcp_manager')
    if not mcp_manager or not mcp_manager.is_connected:
        st.info("📝 No prompts available - MCP not connected")
        return
    
    # Add refresh button for prompts
    if st.button("🔄 Refresh Prompts"):
        # Clear prompts cache to force refresh
        if hasattr(mcp_manager, 'prompts_cache'):
            mcp_manager.prompts_cache = {}
        st.rerun()
    
    # Get all prompts from all servers
    all_prompts = {}
    server_configs = mcp_manager.server_config or {}
    
    for server_name in server_configs.keys():
        try:
            prompts = run_async(lambda: mcp_manager.get_prompts(server_name))
            if prompts:
                all_prompts[server_name] = prompts
        except Exception as e:
            st.debug(f"Failed to get prompts from {server_name}: {e}")
            continue
    
    if not all_prompts:
        st.info("� No prompts available from connected servers")
        return
    
    # Count total prompts
    total_prompts = sum(len(prompts) for prompts in all_prompts.values())
    st.info(f"🎯 {total_prompts} prompts available from {len(all_prompts)} servers")
    
    # Render prompts by server
    for server_name, prompts in all_prompts.items():
        with st.expander(f"📡 {server_name} ({len(prompts)} prompts)", expanded=True):
            for prompt in prompts:
                render_prompt_selector(server_name, prompt)


def render_prompt_selector(server_name: str, prompt: dict):
    """Render individual prompt selector with arguments and generate button."""
    prompt_name = prompt['name']
    prompt_key = f"{server_name}_{prompt_name}"
    
    st.write(f"**🎯 {prompt_name}**")
    if prompt.get('description'):
        st.caption(prompt['description'])
    
    # Arguments form
    prompt_args = {}
    if prompt.get('arguments'):
        st.write("**Arguments:**")
        for arg in prompt['arguments']:
            arg_key = f"prompt_arg_{prompt_key}_{arg['name']}"
            
            if arg.get('required'):
                prompt_args[arg['name']] = st.text_input(
                    f"{arg['name']} *",
                    key=arg_key,
                    help=arg.get('description', ''),
                    placeholder="Required"
                )
            else:
                prompt_args[arg['name']] = st.text_input(
                    f"{arg['name']}",
                    key=arg_key,
                    help=arg.get('description', ''),
                    placeholder="Optional"
                )
    
    # Generate button
    if st.button(f"📝 Generate Prompt", key=f"generate_{prompt_key}"):
        # Validate required arguments
        missing_required = []
        for arg in prompt.get('arguments', []):
            if arg.get('required') and not prompt_args.get(arg['name']):
                missing_required.append(arg['name'])
        
        if missing_required:
            st.error(f"❌ Missing required arguments: {', '.join(missing_required)}")
        else:
            # Filter out empty optional arguments
            filtered_args = {k: v for k, v in prompt_args.items() if v}
            
            try:
                mcp_manager = st.session_state.get('mcp_manager')
                with st.spinner(f"Generating prompt '{prompt_name}'..."):
                    messages = run_async(lambda: mcp_manager.get_prompt(
                        server_name, 
                        prompt_name, 
                        filtered_args if filtered_args else None
                    ))
                
                # Extract content from messages and set in chat input
                if messages:
                    # Combine all message content
                    prompt_content = ""
                    for message in messages:
                        content = getattr(message, 'content', str(message))
                        prompt_content += content + "\n"
                    
                    # Set the prompt content in session state for chat input
                    st.session_state.generated_prompt = prompt_content.strip()
                    st.success(f"✅ Prompt '{prompt_name}' generated! Check the chat input.")
                    st.rerun()
                else:
                    st.warning("⚠️ No content received from prompt")
            
            except Exception as e:
                st.error(f"❌ Failed to generate prompt: {format_error_message(e)}")
    
    st.divider()


def render_tool_selector():
    """Render the tool selection dropdown and information."""
    # Check if the current model supports tools
    model_name = st.session_state.get('selected_model', '')
    supports_tools = model_supports_tools(model_name)
    
    # Add history tool to the dropdown when memory is enabled AND model supports tools
    tool_options = [tool.name for tool in st.session_state.tools]
    if st.session_state.get('memory_enabled', False) and supports_tools:
        tool_options.append("get_conversation_history (Memory)")
    
    # Only show tool selection if there are tools available
    if tool_options:
        selected_tool_name = st.selectbox(
            "Available Tools",
            options=tool_options,
            index=0 if tool_options else None
        )
        
        if selected_tool_name:
            render_tool_information(selected_tool_name)
    else:
        # No tools available
        if st.session_state.get('memory_enabled', False) and not supports_tools:
            st.info("💬 Memory is enabled but works through conversation context only (no tool interface)")
        else:
            st.info("� No MCP tools configured - please check your configuration")


def render_tool_information(selected_tool_name: str):
    """Render detailed information about the selected tool."""
    if selected_tool_name == "get_conversation_history (Memory)":
        st.write("**Description:** Retrieve conversation history from the current session with advanced filtering and search options")
        st.write("**Enhanced Features:**")
        st.write("• Timestamps and message IDs for precise referencing")
        st.write("• Date range filtering and flexible sorting")
        st.write("• Rich metadata including tool execution details")
        st.write("• Advanced search with boolean operators and regex support")
        
        st.write("**Parameters:**")
        st.code("message_type: string (optional) [default: all]")
        st.code("last_n_messages: integer (optional) [default: 10, max: 100]") 
        st.code("search_query: string (optional) - supports text, boolean ops, regex")
        st.code("sort_order: string (optional) [default: newest_first]")
        st.code("date_from: string (optional) [YYYY-MM-DD format]")
        st.code("date_to: string (optional) [YYYY-MM-DD format]")
        st.code("include_metadata: boolean (optional) [default: true]")
        
        with st.expander("🔍 Advanced Search Examples"):
            st.write("**Simple Text Search:**")
            st.code('search_query="weather"')
            
            st.write("**Boolean Operators:**")
            st.code('search_query="weather AND temperature"')
            st.code('search_query="sunny OR cloudy OR rainy"')
            st.code('search_query="weather NOT rain"')
            st.code('search_query="(weather OR climate) AND NOT error"')
            
            st.write("**Regex Patterns:**")
            st.code('search_query="regex:\\\\d{2}°[CF]"  # Find temperatures like "72°F"')
            st.code('search_query="regex:https?://\\\\S+"  # Find URLs')
            st.code('search_query="regex:\\\\$\\\\d+(\\\\.\\\\d{2})?"  # Find dollar amounts')
            st.code('search_query="regex:\\\\b\\\\d{4}-\\\\d{2}-\\\\d{2}\\\\b"  # Find dates')
            
            st.write("**Complex Queries:**")
            st.code('search_query="tool AND (success OR complete) NOT error"')
            st.code('search_query="regex:API.*key AND NOT expired"')
        
        st.info("💡 This enhanced tool provides enterprise-grade conversation history access with powerful search capabilities including boolean logic and regex pattern matching.")
    else:
        # Find the selected MCP tool
        selected_tool = next((tool for tool in st.session_state.tools if tool.name == selected_tool_name), None)
        
        if selected_tool:
            # Display tool information
            st.write(f"**Description:** {selected_tool.description}")
            
            # Display parameters if available
            if hasattr(selected_tool, 'args_schema'):
                st.write("**Parameters:**")
                render_tool_parameters(selected_tool)


def render_tool_parameters(tool):
    """Render tool parameters information."""
    # Get schema properties directly from the tool
    schema = getattr(tool, 'args_schema', {})
    if isinstance(schema, dict):
        properties = schema.get('properties', {})
        required = schema.get('required', [])
    else:
        # Handle Pydantic schema
        schema_dict = schema.schema()
        properties = schema_dict.get('properties', {})
        required = schema_dict.get('required', [])

    # Display each parameter with its details
    for param_name, param_info in properties.items():
        # Get parameter details
        param_type = param_info.get('type', 'string')
        param_title = param_info.get('title', param_name)
        param_default = param_info.get('default', None)
        is_required = param_name in required

        # Build parameter description
        param_desc = [
            f"{param_title}:",
            f"{param_type}",
            "(required)" if is_required else "(optional)"
        ]
        
        if param_default is not None:
            param_desc.append(f"[default: {param_default}]")

        # Display parameter info
        st.code(" ".join(param_desc), wrap_lines=True) 