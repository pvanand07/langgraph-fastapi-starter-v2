"""Chat server implementation with LangChain agent."""

from typing import Optional, AsyncIterator, Dict, Any
from datetime import datetime
import json
import logging
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from prompt import CONTEXT_INTELLIGENCE_PROMPT

from config import DEFAULT_MODEL_CONFIG
from memory_store import get_memory
from tools import get_tools, ChatContext

# Configure logger
# Get logger - will use root logger configuration from basicConfig in main.py
logger = logging.getLogger(__name__)

def create_model(model_id: Optional[str] = None) -> ChatOpenAI:
    """Create a ChatOpenAI model with optional model_id override (matching IResearcher-v5 pattern)."""
    config = DEFAULT_MODEL_CONFIG.copy()
    if model_id:
        config["model"] = model_id
    return ChatOpenAI(**config)


def create_system_prompt(context: str = "") -> str:
    """Create system prompt for the agent."""
    today_date = datetime.now().strftime("%B %d, %Y")
    
    base_prompt = f"""You are a helpful AI assistant. Today's date is {today_date}.

You have access to the following tools:
- calculator: Evaluate mathematical expressions
- get_current_time: Get the current date and time
- search_documents: Search through user's uploaded documents
- query_duckdb: Execute SQL queries against the user's data tables in DuckDB
- create_view: Create persistent views in DuckDB

You also have access to data tables stored in DuckDB. 

When users ask questions about their data:
1. First, check the context to see what tables and schemas are available
2. Plan your approach to finding out the requested information using the available tables
3. In case you run into errors perform more eda to refine your assumptions, refine your original approach and then try again
3. In case of missing information, ask the user for more information or suggest alternative approaches to get the information
4. Use the query_duckdb tool to execute SQL queries to get specific data
5. Reference the table names and schemas provided in the context to construct accurate queries

Be helpful, and accurate in your responses.
If you are not sure about the answer, say so and ask the user for more information.
{CONTEXT_INTELLIGENCE_PROMPT}"""
    
    if context:
        base_prompt += f"\n\nCONTEXT:\n{context}"
    
    return base_prompt


class ChatServer:
    """Chat server managing LangChain agent lifecycle."""
    
    def __init__(self):
        """Initialize the chat server."""
        self.tools = get_tools()
        self.checkpointer = get_memory()
    
    def get_agent(self, model_id: Optional[str] = None, context: str = "") -> Any:
        """Get or create a LangChain agent instance."""
        model = create_model(model_id)
        system_prompt = create_system_prompt(context)
        
        agent = create_agent(
            model,
            tools=self.tools,
            checkpointer=self.checkpointer,
            system_prompt=system_prompt,
            context_schema=ChatContext
        )
        
        return agent
    
    async def process_message(
        self,
        message: str,
        thread_id: str,
        user_id: str,
        model_id: Optional[str] = None,
        context: str = ""
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a user message and stream the response.
        
        Args:
            message: User message text
            thread_id: Conversation thread ID
            user_id: User ID
            model_id: Optional model override
            context: Optional context string (e.g., document context)
        
        Yields:
            Dictionary with response chunks
        """
        # Create agent with context (context is included in system prompt)
        agent = self.get_agent(model_id, context)
        
        # Prepare config with thread_id and user_id, and context
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }
        
        # Create context for tools
        context_obj = ChatContext(user_id=user_id, thread_id=thread_id)
        
        # Stream agent response
        logger.info(f"Starting to stream response for thread_id: {thread_id}")
        async for event in agent.astream(
            {"messages": [HumanMessage(content=message)]},
            config,
            context=context_obj,
            stream_mode=["messages", "updates"],
            subgraphs=False,
        ):
            logger.debug(f"Event received: {event[0]}")
            # Handle message events
            if event[0] == "messages" and event[1][0].content:
                msg = event[1][0]
                metadata = event[1][1]
                if isinstance(msg, AIMessage):
                    # Only yield chunks not from tools node
                    if metadata.get("langgraph_node") != "tools":
                        yield {
                            "type": "chunk",
                            "content": msg.content
                        }
                elif isinstance(msg, ToolMessage):
                    yield {
                        "type": "tool_end",
                        "name": msg.name,
                        "output": msg.content,
                        "artifacts_data": getattr(msg, 'artifact', None)
                    }
            
            # Handle updates for tool calls and final responses
            elif event[0] == "updates" and event[1]:
                logger.debug(f"Updates event received: {type(event[1])}")
                if isinstance(event[1], dict):
                    logger.debug(f"Updates keys: {list(event[1].keys())}")
                    
                    # Iterate over update sources (e.g., "model", "tools")
                    for source, update in event[1].items():
                        if source == "model" and isinstance(update, dict):
                            messages = update.get("messages", [])
                            if messages:
                                msg = messages[0]
                                logger.debug(f"Model update message type: {type(msg)}")
                                if isinstance(msg, AIMessage):
                                    # Check for tool_calls in the message
                                    # Tool calls can be in tool_calls attribute or additional_kwargs
                                    tool_calls = getattr(msg, 'tool_calls', [])
                                    if not tool_calls:
                                        tool_calls = msg.additional_kwargs.get('tool_calls', [])
                                    
                                    if tool_calls:
                                        # Tool start event - emit for each tool call
                                        for call in tool_calls:
                                            tool_name = None
                                            tool_input = None
                                            
                                            # Handle ToolCall objects (from langchain_core.messages)
                                            if hasattr(call, 'name'):
                                                tool_name = call.name
                                                tool_input = getattr(call, 'args', None) or getattr(call, 'arguments', None)
                                            # Handle dict format
                                            elif isinstance(call, dict):
                                                if 'function' in call:
                                                    # Format: {'function': {'name': '...', 'arguments': '...'}, ...}
                                                    func_dict = call.get('function', {})
                                                    tool_name = func_dict.get('name')
                                                    tool_input = func_dict.get('arguments')
                                                    # Parse JSON string if needed
                                                    if isinstance(tool_input, str):
                                                        try:
                                                            tool_input = json.loads(tool_input)
                                                        except (json.JSONDecodeError, ValueError):
                                                            pass
                                                else:
                                                    # Format: {'name': '...', 'args': {...}, ...}
                                                    tool_name = call.get('name')
                                                    tool_input = call.get('args') or call.get('arguments')
                                            
                                            if tool_name:
                                                yield {
                                                    "type": "tool_start",
                                                    "name": tool_name,
                                                    "input": tool_input
                                                }
                                    
                                    # Full response event (only if there's content or it's the final message)
                                    if msg.content or not tool_calls:
                                        usage_metadata = getattr(msg, 'usage_metadata', {})
                                        full_response_data = {
                                            "type": "full_response",
                                            "content": msg.content,
                                            "usage_metadata": usage_metadata
                                        }
                                        yield full_response_data
                        
                        elif source == "tools" and isinstance(update, dict):
                            # Tool execution results are already handled in messages stream
                            # But we can log them here for debugging
                            messages = update.get("messages", [])
                            if messages:
                                logger.debug(f"Tools update: {len(messages)} message(s)")
