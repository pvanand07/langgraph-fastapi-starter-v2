"""Chat server implementation with LangChain agent."""

from typing import Optional, AsyncIterator, Dict, Any
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from config import DEFAULT_MODEL_CONFIG
from memory_store import get_memory
from tools import get_tools


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

Be concise, helpful, and accurate in your responses."""
    
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
            system_prompt=system_prompt
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
        
        # Prepare config with thread_id and user_id
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": user_id
            }
        }
        
        # Stream agent response
        full_response = ""
        async for event in agent.astream(
            {"messages": [HumanMessage(content=message)]},
            config,
            stream_mode=["messages", "updates"],
            subgraphs=False,
        ):
            # Handle message events
            if event[0] == "messages" and event[1]:
                msg = event[1][0]
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content
                    yield {
                        "type": "chunk",
                        "content": content
                    }
                    full_response += content
                elif isinstance(msg, ToolMessage):
                    yield {
                        "type": "tool_end",
                        "name": getattr(msg, 'name', 'unknown'),
                        "output": str(msg.content)
                    }
            
            # Handle updates for tool calls and final responses
            elif event[0] == "updates" and event[1]:
                update = event[1]
                if isinstance(update, dict):
                    # Check for agent updates with tool calls
                    if "agent" in update:
                        agent_update = update["agent"]
                        if isinstance(agent_update, dict):
                            messages = agent_update.get("messages", [])
                            if messages:
                                for msg in messages:
                                    if isinstance(msg, AIMessage):
                                        # Check for tool calls
                                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                            for tool_call in msg.tool_calls:
                                                yield {
                                                    "type": "tool_start",
                                                    "name": tool_call.get("name", "unknown"),
                                                    "input": tool_call.get("args", {})
                                                }
                                        # Final response
                                        if msg.content:
                                            yield {
                                                "type": "full_response",
                                                "content": msg.content
                                            }
                                            full_response = msg.content

