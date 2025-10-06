"""
MCP (Model Context Protocol) client management and tool/prompt retrieval.

This module handles the setup and management of MCP clients,
server configurations, tool retrieval, and prompt handling.
"""

import asyncio
import atexit
import logging
import random
from typing import Dict, List, Optional, Any
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# Module-level singleton instance
_connection_manager_instance: Optional['MCPConnectionManager'] = None


class MCPConnectionManager:
    """
    Singleton manager for MCP connections with heartbeat, reconnection, tool caching, and prompt handling.
    
    Features:
    - Singleton pattern with thread-safe initialization
    - Automatic heartbeat to keep SSE connections alive
    - Exponential backoff reconnection with jitter
    - Tool caching with refresh capabilities
    - Prompt listing and retrieval capabilities
    - Proper async resource cleanup
    """
    
    def __init__(self):
        self.client: Optional[MultiServerMCPClient] = None
        self.server_config: Optional[Dict[str, Dict]] = None
        self.lock = asyncio.Lock()
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.reconnect_task: Optional[asyncio.Task] = None
        self.running = False
        self.tools_cache: List[BaseTool] = []
        self.prompts_cache: Dict[str, List[Dict[str, Any]]] = {}  # server_name -> list of prompts
        self.last_heartbeat_ok = False  # Start as False until first successful connection
        self._shutdown_registered = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    @classmethod
    def get_instance(cls) -> 'MCPConnectionManager':
        """Get or create the singleton instance."""
        global _connection_manager_instance
        if _connection_manager_instance is None:
            _connection_manager_instance = cls()
            # Register cleanup on process exit
            if not _connection_manager_instance._shutdown_registered:
                atexit.register(_connection_manager_instance._cleanup_on_exit)
                _connection_manager_instance._shutdown_registered = True
        return _connection_manager_instance
    
    def _cleanup_on_exit(self):
        """Cleanup method for atexit handler."""
        try:
            if self.running:
                # Prefer using the original loop that created tasks
                loop = self._loop
                if loop and not loop.is_closed():
                    if loop.is_running():
                        try:
                            fut = asyncio.run_coroutine_threadsafe(self.close(), loop)
                            fut.result(timeout=2)
                        except Exception as e:
                            logger.error(f"Error during MCP connection cleanup (thread-safe): {e}")
                    else:
                        loop.run_until_complete(self.close())
                else:
                    logger.debug("Skipping MCP cleanup: no active event loop available")
        except Exception as e:
            logger.error(f"Error during MCP connection cleanup: {e}")
    
    async def start(self, server_config: Dict[str, Dict]) -> None:
        """
        Start the MCP connection with the given configuration.
        
        Args:
            server_config: Server configuration dictionary
        """
        async with self.lock:
            # If already running with same config, no-op
            if self.running and self.server_config == server_config:
                logger.info("MCP connection already running with same configuration")
                return
            
            # If running but config changed, close and restart
            if self.running:
                logger.info("Configuration changed, restarting MCP connection")
                await self._close_internal()
            
            # Store config and create client
            self.server_config = server_config
            try:
                self.client = MultiServerMCPClient(server_config)
                
                # Note: We don't validate connectivity here because different servers
                # have different capabilities (some only support tools, others only prompts)
                # Individual operations will handle their own validation
                logger.info("MCP client created successfully")
                
                # Mark as successfully connected
                self.last_heartbeat_ok = True
                self.running = True
                
                # Clear caches to force refresh on next calls
                self.tools_cache = []
                self.prompts_cache = {}
                
                # Start heartbeat task
                self._loop = asyncio.get_running_loop()
                self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
            except Exception as e:
                logger.error(f"Failed to start MCP connection: {e}")
                await self._close_internal()
                raise
    
    async def get_tools(self, force_refresh: bool = False) -> List[BaseTool]:
        """
        Get tools from the MCP client, with caching.
        Gracefully handles servers that don't support tools.
        
        Args:
            force_refresh: If True, bypass cache and fetch fresh tools
            
        Returns:
            List of available tools from servers that support them
        """
        if not self.client:
            logger.warning("No MCP client available")
            return []
        
        if not self.tools_cache or force_refresh:
            try:
                logger.debug("Fetching tools from MCP client...")
                
                # Get tools from each server individually to handle failures gracefully
                all_tools = []
                for server_name in self.server_config.keys():
                    try:
                        server_tools = await self.client.get_tools(server_name=server_name)
                        all_tools.extend(server_tools)
                        logger.debug(f"Got {len(server_tools)} tools from server '{server_name}'")
                    except Exception as e:
                        # Log but don't fail - server might not support tools (like prompts-only servers)
                        logger.debug(f"Server '{server_name}' doesn't support tools or failed: {e}")
                        continue
                
                self.tools_cache = all_tools
                logger.info(f"Retrieved {len(self.tools_cache)} tools from {len(self.server_config)} servers")
            except Exception as e:
                logger.error(f"Failed to get tools from MCP client: {e}")
                if not self.tools_cache:  # Only return empty if no cached tools
                    return []
        else:
            logger.debug(f"Using cached tools: {len(self.tools_cache)} tools")
        
        return self.tools_cache.copy()
    
    async def get_prompts(self, server_name: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Get prompts from a specific MCP server, with caching.
        
        Args:
            server_name: Name of the server to get prompts from
            force_refresh: If True, bypass cache and fetch fresh prompts
            
        Returns:
            List of available prompts with their metadata
        """
        if not self.client:
            logger.warning("No MCP client available")
            return []
        
        if server_name not in self.server_config:
            logger.warning(f"Server '{server_name}' not found in configuration")
            return []
        
        if server_name not in self.prompts_cache or force_refresh:
            try:
                logger.debug(f"Fetching prompts from MCP server '{server_name}'...")
                # Use session to list prompts from specific server
                async with self.client.session(server_name) as session:
                    prompts_result = await session.list_prompts()
                    
                    server_prompts = []
                    for prompt in prompts_result.prompts:
                        prompt_dict = {
                            "name": prompt.name,
                            "title": getattr(prompt, 'title', None),
                            "description": getattr(prompt, 'description', ''),
                            "arguments": []
                        }
                        
                        # Extract argument information if available
                        if hasattr(prompt, 'arguments') and prompt.arguments:
                            for arg in prompt.arguments:
                                arg_dict = {
                                    "name": arg.name,
                                    "description": getattr(arg, 'description', ''),
                                    "required": getattr(arg, 'required', False)
                                }
                                prompt_dict["arguments"].append(arg_dict)
                        
                        server_prompts.append(prompt_dict)
                    
                    self.prompts_cache[server_name] = server_prompts
                    logger.info(f"Retrieved {len(server_prompts)} prompts from MCP server '{server_name}'")
            except Exception as e:
                logger.error(f"Failed to get prompts from MCP server '{server_name}': {e}")
                if server_name not in self.prompts_cache:  # Only return empty if no cached prompts
                    return []
        else:
            logger.debug(f"Using cached prompts for '{server_name}': {len(self.prompts_cache[server_name])} prompts")
        
        return self.prompts_cache.get(server_name, []).copy()
    
    async def get_prompt(
        self, 
        server_name: str, 
        prompt_name: str, 
        arguments: Optional[Dict[str, Any]] = None
    ) -> List[HumanMessage | AIMessage]:
        """
        Get a specific prompt by name from a specific server with arguments.
        
        Args:
            server_name: Name of the server to get the prompt from
            prompt_name: Name of the prompt to retrieve
            arguments: Optional arguments to pass to the prompt
            
        Returns:
            List of LangChain message objects representing the prompt
        """
        if not self.client:
            raise ValueError("No MCP client available")
        
        if server_name not in self.server_config:
            raise ValueError(f"Server '{server_name}' not found in configuration")
        
        try:
            logger.debug(f"Getting prompt '{prompt_name}' from server '{server_name}' with arguments: {arguments}")
            
            # Use the client's get_prompt method directly
            messages = await self.client.get_prompt(
                server_name=server_name,
                prompt_name=prompt_name, 
                arguments=arguments or {}
            )
            
            logger.info(f"Successfully retrieved prompt '{prompt_name}' from server '{server_name}'")
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get prompt '{prompt_name}' from server '{server_name}': {e}")
            raise
    
    async def close(self) -> None:
        """Close the MCP connection and cleanup resources."""
        async with self.lock:
            await self._close_internal()
    
    async def _close_internal(self) -> None:
        """Internal close method without locking."""
        logger.info("Closing MCP connection")
        self.running = False
        
        # Cancel background tasks
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self._drain_task(self.heartbeat_task)
            finally:
                self.heartbeat_task = None
        
        if self.reconnect_task:
            self.reconnect_task.cancel()
            try:
                await self._drain_task(self.reconnect_task)
            finally:
                self.reconnect_task = None
        
        # Close client if it has cleanup methods
        if self.client:
            # MultiServerMCPClient doesn't have explicit close method
            # but we clear the reference
            self.client = None
        
        # Clear state
        self.server_config = None
        self.tools_cache = []
        self.prompts_cache = {}
        self.last_heartbeat_ok = False
        self._loop = None

    async def _drain_task(self, task: asyncio.Task) -> None:
        """Await a cancelled task on its owning loop to prevent pending destruction warnings."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        try:
            task_loop = task.get_loop()
        except Exception:
            task_loop = None

        # If we are on the same loop, await directly
        if current_loop is not None and task_loop is current_loop:
            try:
                await task
            except asyncio.CancelledError:
                pass
            return

        # If the task has a loop, schedule a drain coroutine on that loop
        if task_loop is not None and not task_loop.is_closed():
            try:
                fut = asyncio.run_coroutine_threadsafe(self._drain_task_on_loop(task), task_loop)
                try:
                    fut.result(timeout=2)
                except Exception:
                    pass
            except Exception:
                pass

    async def _drain_task_on_loop(self, task: asyncio.Task) -> None:
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    async def _heartbeat_loop(self, interval_sec: int = 45) -> None:
        """
        Heartbeat loop to keep SSE connection alive.
        
        Args:
            interval_sec: Interval between heartbeats in seconds
        """
        logger.info(f"Starting heartbeat loop with {interval_sec}s interval")
        
        while self.running:
            try:
                await asyncio.sleep(interval_sec)
                
                if not self.running:
                    break
                
                # Test connection by checking if we have active connections
                # We don't call get_tools() as some servers may not support tools
                if self.client and self.client.connections:
                    # Just check that we have active connections
                    self.last_heartbeat_ok = True
                    logger.debug("Heartbeat successful")
                else:
                    raise Exception("No active MCP connections")
                
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                self.last_heartbeat_ok = False
                if not self.running:
                    break
                self._ensure_reconnect()
    
    def _ensure_reconnect(self) -> None:
        """Ensure reconnection task is started if not already running."""
        if self.reconnect_task is None or self.reconnect_task.done():
            logger.info("Starting reconnection task")
            self.reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    async def _reconnect_loop(self, base_delay: float = 0.5, max_delay: float = 30.0) -> None:
        """
        Reconnection loop with exponential backoff and jitter.
        
        Args:
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between attempts
        """
        attempt = 0
        
        while self.running and not self.last_heartbeat_ok:
            try:
                # Calculate delay with exponential backoff and full jitter
                delay = min(max_delay, base_delay * (2 ** attempt))
                jittered_delay = random.uniform(0, delay)
                
                logger.info(f"Reconnection attempt {attempt + 1} in {jittered_delay:.2f}s")
                await asyncio.sleep(jittered_delay)
                
                if not self.running:
                    break
                
                # Try to recreate client and validate
                new_client = MultiServerMCPClient(self.server_config)
                # Just check that client was created with connections
                if not new_client.connections:
                    raise Exception("No connections in new client")
                
                # Success - swap in new client
                async with self.lock:
                    self.client = new_client
                    self.tools_cache = []  # Clear cache to force refresh
                    self.prompts_cache = {}  # Clear prompts cache to force refresh
                    self.last_heartbeat_ok = True
                
                logger.info("Reconnection successful")
                break
                
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt + 1} failed: {e}")
                attempt += 1
        
        # Clear reconnect task reference
        self.reconnect_task = None
    
    @property
    def is_connected(self) -> bool:
        """Check if the manager is connected and healthy."""
        return self.running and self.client is not None and self.last_heartbeat_ok