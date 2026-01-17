### Create Agent with Retrieval Tool - Python

Source: https://docs.langchain.com/oss/python/langchain/retrieval

Demonstrates how to create a basic Agentic RAG agent using LangChain. The agent has access to a `fetch_url` tool that retrieves text content from URLs, allowing it to decide when to fetch external information based on user queries.

```python
import requests
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent


@tool
def fetch_url(url: str) -> str:
    """Fetch text content from a URL"""
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return response.text

system_prompt = """\
Use fetch_url when you need to fetch information from a web-page; quote relevant snippets.
"""

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[fetch_url],
    system_prompt=system_prompt,
)
```

--------------------------------

### Stream LangChain Agent Responses (Python)

Source: https://docs.langchain.com/oss/python/langchain/agents

This snippet shows how to stream intermediate messages and the full state from a LangChain agent using `agent.stream()`. By iterating through chunks, developers can display real-time progress, including agent responses or tool calls, during potentially long-running agent executions.

```python
for chunk in agent.stream({
    "messages": [{"role": "user", "content": "Search for AI news and summarize the findings"}]
}, stream_mode="values"):
    # Each chunk contains the full state at that point
    latest_message = chunk["messages"][-1]
    if latest_message.content:
        print(f"Agent: {latest_message.content}")
    elif latest_message.tool_calls:
        print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")
```

--------------------------------

### Configure LangChain Agent for Structured Output with ToolStrategy

Source: https://docs.langchain.com/oss/python/langchain/agents

This example demonstrates how to configure a LangChain agent to return structured output using `ToolStrategy`. It defines a Pydantic `BaseModel` (`ContactInfo`) to specify the desired output format, which the agent then attempts to extract using artificial tool calling.

```python
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_tool],
    response_format=ToolStrategy(ContactInfo)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

--------------------------------

### Create LangChain Agent with Static Model Identifier (Python)

Source: https://docs.langchain.com/oss/python/langchain/agents

This snippet demonstrates how to initialize a LangChain agent using the `create_agent` function with a static model identifier string. This method is straightforward for configuring the agent's language model component with a common provider format.

```python
from langchain.agents import create_agent

agent = create_agent("gpt-5", tools=tools)
```

--------------------------------

### Route Initial Agent Selection

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

Routing function that selects the initial agent based on the active_agent state value. Returns either the configured active agent or defaults to the sales_agent if no agent is specified.

```python
def route_initial(
    state: MultiAgentState,
) -> Literal["sales_agent", "support_agent"]:
    """Route to the active agent based on state, default to sales agent."""
    return state.get("active_agent") or "sales_agent"
```

--------------------------------

### Configure LangChain Agent with Basic String System Prompt in Python

Source: https://docs.langchain.com/oss/python/langchain/agents

This snippet demonstrates how to initialize a LangChain agent using the `create_agent` function with a simple string-based system prompt. The system prompt guides the agent's behavior, defining its personality to be helpful, concise, and accurate.

```python
agent = create_agent(
    model,
    tools,
    system_prompt="You are a helpful assistant. Be concise and accurate."
)
```

--------------------------------

### Create SQL Agent with LangChain create_agent

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

Instantiates a ReAct agent using the create_agent function with a language model, SQL tools, and system prompt. The agent interprets natural language requests and generates SQL commands with built-in error handling and feedback mechanisms for iterative query refinement.

```python
from langchain.agents import create_agent


agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
)
```

--------------------------------

### Agent Registry with Task Dispatcher in LangChain

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

Implements a convention-based single dispatch pattern with an agent registry mapping agent names to instances. The task tool accepts an agent_name and description, looks up the agent in the registry, invokes it with the description as a human message, and returns the final response. Enables team distribution and scalable agent composition without modifying the coordinator.

```python
from langchain.tools import tool
from langchain.agents import create_agent

# Sub-agents developed by different teams
research_agent = create_agent(
    model="gpt-4o",
    prompt="You are a research specialist..."
)

writer_agent = create_agent(
    model="gpt-4o",
    prompt="You are a writing specialist..."
)

# Registry of available sub-agents
SUBAGENTS = {
    "research": research_agent,
    "writer": writer_agent,
}

@tool
def task(
    agent_name: str,
    description: str
) -> str:
    """Launch an ephemeral subagent for a task.

    Available agents:
    - research: Research and fact-finding
    - writer: Content creation and editing
    """
    agent = SUBAGENTS[agent_name]
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": description}
        ]
    })
    return result["messages"][-1].content

# Main coordinator agent
main_agent = create_agent(
    model="gpt-4o",
    tools=[task],
    system_prompt=(
        "You coordinate specialized sub-agents. "
        "Available: research (fact-finding), "
        "writer (content creation). "
        "Use the task tool to delegate work."
    ),
)
```

--------------------------------

### Create Agent with FilesystemFileSearchMiddleware

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Initialize a LangChain agent with filesystem file search middleware configured to search from a specified root directory. The middleware automatically adds glob_search and grep_search tools to the agent for file discovery and content searching.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path="/workspace",
            use_ripgrep=True,
        ),
    ],
)
```

--------------------------------

### Create Agent with Compliance Middleware in Python

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Demonstrates agent creation with the compliance rules middleware attached. The agent uses GPT-4o model with tools and a context schema, with middleware applied to enforce compliance requirements on all model calls.

```python
agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[inject_compliance_rules],
    context_schema=Context
)
```

--------------------------------

### Invoke LangChain Agent with New Messages

Source: https://docs.langchain.com/oss/python/langchain/agents

This snippet illustrates how to invoke a LangChain agent by passing an update to its state, specifically by providing a new user message within the 'messages' sequence. This is the standard way to interact with an agent and advance the conversation.

```python
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
```

--------------------------------

### Create Agent with Constrained Middleware

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Instantiates the agent with the custom middleware that enforces skill-based constraints. The agent uses the SkillMiddleware to manage tool availability based on loaded skills and maintain conversation state.

```python
agent = create_agent(
    model,
    system_prompt=(
        "You are a SQL query assistant that helps users "
        "write queries against business databases."
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Run LangChain Agent to Query Database (Python)

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

This code initializes a LangChain agent and streams its execution to answer a natural language question about database content. It demonstrates the agent's ability to orchestrate tool calls to a SQL database, taking a natural language question as input and outputting the agent's thought process and final answer.

```python
question = "Which genre on average has the longest tracks?"

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

--------------------------------

### Define Custom Tools for LangChain Python Agents

Source: https://docs.langchain.com/oss/python/langchain/agents

This code illustrates how to define custom tools for a LangChain agent using the `@tool` decorator. It creates two example tools, `search` and `get_weather`, each with a docstring that serves as its description. These tools are then passed as a list to the `create_agent` function, enabling the agent to utilize these specific functionalities.

```python
from langchain.tools import tool
from langchain.agents import create_agent


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

agent = create_agent(model, tools=[search, get_weather])
```

--------------------------------

### Agent Creation and Initialization - Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Initializes a ChatOpenAI model and creates an agent with skill middleware support, system prompt configuration, and in-memory checkpointing. The agent is configured to function as a SQL query assistant with access to skill management through the middleware.

```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4")

# Create the agent with skill support
agent = create_agent(
    model,
    system_prompt=(
        "You are a SQL query assistant that helps users "
        "write queries against business databases."
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Create LangChain Email Agent with Tools

Source: https://docs.langchain.com/oss/python/langchain/studio

Define a simple LangChain agent with an email tool. The agent uses GPT-4o model and includes a send_email tool that can be called by the agent. The create_agent function returns a compiled LangGraph graph.

```python
from langchain.agents import create_agent

def send_email(to: str, subject: str, body: str):
    """Send an email"""
    email = {
        "to": to,
        "subject": subject,
        "body": body
    }
    # ... email sending logic

    return f"Email sent to {to}"

agent = create_agent(
    "gpt-4o",
    tools=[send_email],
    system_prompt="You are an email assistant. Always use the send_email tool.",
)
```

--------------------------------

### Create Retrieval Tool and RAG Agent in Python

Source: https://docs.langchain.com/oss/python/langchain/rag

Defines a retrieval tool that searches the vector store for relevant context and creates a LangChain agent that can execute this tool. The agent uses similarity search to retrieve the most relevant documents based on queries.

```python
# Construct a tool for retrieving context
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)
```

--------------------------------

### Configure LangChain Agent for Structured Output with ProviderStrategy

Source: https://docs.langchain.com/oss/python/langchain/agents

This snippet shows how to utilize `ProviderStrategy` for structured output in a LangChain agent. This strategy leverages the model provider's native capabilities for structured generation, offering a potentially more reliable approach, though it's dependent on the provider's support for this feature.

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_agent(
    model="gpt-4o",
    response_format=ProviderStrategy(ContactInfo)
)
```

--------------------------------

### Stream Email Agent Response with Formatted Output

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Execute the email agent with a natural language request and stream the response to display formatted messages, tool calls, and execution results. This shows how to iterate through agent steps and pretty-print messages, demonstrating the agent's inference of recipients, subject lines, and body composition from informal user requests.

```python
query = "Send the design team a reminder about reviewing the new mockups"

for step in email_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()
```

--------------------------------

### Execute RAG Agent Query Streaming in Python

Source: https://docs.langchain.com/oss/python/langchain/rag

Demonstrates how to invoke the RAG agent with a user query and stream the agent's processing steps. Shows the agent making tool calls to retrieve context and generating responses based on retrieved information.

```python
query = "What is task decomposition?"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
```

--------------------------------

### Agent File Search with Glob and Grep Tools

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Complete example demonstrating how to use a LangChain agent with file search middleware to find files matching patterns and search their contents. The agent uses glob_search for pattern matching and grep_search for regex-based content filtering across specified file types.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import FilesystemFileSearchMiddleware
from langchain.messages import HumanMessage


agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        FilesystemFileSearchMiddleware(
            root_path="/workspace",
            use_ripgrep=True,
            max_file_size_mb=10,
        ),
    ],
)

# Agent can now use glob_search and grep_search tools
result = agent.invoke({
    "messages": [HumanMessage("Find all Python files containing 'async def'")]
})

# The agent will use:
# 1. glob_search(pattern="**/*.py") to find Python files
# 2. grep_search(pattern="async def", include="*.py") to find async functions
```

--------------------------------

### Define Custom Agent State with LangChain Middleware (Python)

Source: https://docs.langchain.com/oss/python/langchain/agents

This snippet illustrates how to define and integrate custom agent state using LangChain's `AgentMiddleware`. Custom state, like `user_preferences`, can be accessed by specific middleware hooks and tools attached to the middleware, providing a robust way to manage agent context.

```python
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any


class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        ...

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

--------------------------------

### Import Dependencies for LangChain Agent Creation in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

This Python snippet imports necessary modules for creating a LangChain agent with state persistence. It includes `create_agent` from `langchain.agents` for agent construction and `InMemorySaver` from `langgraph.checkpoint.memory` for managing agent state in memory, setting up the basic components required to instantiate a conversational agent.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
```

--------------------------------

### Execute Supervisor Agent with Multi-Tool Request

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Main execution block that demonstrates how to use a supervisor agent to handle complex user requests spanning multiple domains (calendar scheduling and email). The code streams agent responses and pretty-prints messages to show the agent's reasoning and actions. This example shows the practical usage of the three-layer agent architecture where the supervisor routes requests to appropriate sub-agents.

```python
if __name__ == "__main__":
    # Example: User request requiring both calendar and email coordination
    user_request = (
        "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
        "and send them an email reminder about reviewing the new mockups."
    )

    print("User Request:", user_request)
    print("\n" + "="*80 + "\n")

    for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": user_request}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print()
```

--------------------------------

### Create Agent with Pydantic Model ToolStrategy

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Implements structured output using Pydantic BaseModel with field validation. Defines ProductReview schema with rating validation (1-5), sentiment classification, and key points extraction. Demonstrates agent invocation and result access.

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
    key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})
result["structured_response"]
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
```

--------------------------------

### Python: Create Langchain Agent with RAG Middleware

Source: https://docs.langchain.com/oss/python/langchain/rag

This Python snippet illustrates the creation of a Langchain agent. It initializes an agent using 'create_agent', passes a language model, an empty list of tools, and crucially includes 'RetrieveDocumentsMiddleware' to enable Retrieval Augmented Generation (RAG) capabilities for the agent.

```python
agent = create_agent(
    model,
    tools=[],
    middleware=[RetrieveDocumentsMiddleware()],
)
```

--------------------------------

### Install LangChain and Dependencies

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

Install required packages for building SQL agents with LangChain. Includes langchain core, langgraph for agentic workflows, and langchain-community for database tools.

```bash
pip install langchain  langgraph  langchain-community
```

--------------------------------

### Python LangChain Agent Construction with Custom Tools

Source: https://docs.langchain.com/oss/python/langchain/rag

This Python code demonstrates how to instantiate a Retrieval Augmented Generation (RAG) agent using `langchain.agents.create_agent`. It takes a language model, a list of custom tools (like `retrieve_context`), and an optional `system_prompt` to guide the agent's behavior. This forms the core setup for a RAG agent that can utilize custom retrieval capabilities.

```python
from langchain.agents import create_agent


tools = [retrieve_context]
# If desired, specify custom instructions
prompt = (
    "You have access to a tool that retrieves context from a blog post. "
    "Use the tool to help answer user queries."
)
agent = create_agent(model, tools, system_prompt=prompt)
```

--------------------------------

### Create Agent with Union Type Structured Output

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Creates a LangChain agent that accepts multiple Pydantic model types using Union types for flexible structured output. Supports both ProductReview and CustomerComplaint models, allowing the agent to return either type based on input analysis. Uses ToolStrategy to enforce the union type constraint.

```python
agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(Union[ProductReview, CustomerComplaint])
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})
result["structured_response"]
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
```

--------------------------------

### Create Email Agent with System Prompt in LangChain

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Initialize an email agent with a system prompt that instructs it to compose professional emails based on natural language requests. The agent uses send_email tool to handle message delivery and confirms sent communications. This demonstrates the foundation of a specialized sub-agent focused on email composition and delivery.

```python
EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response."
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
)
```

--------------------------------

### Create LangChain Agent with Direct Chat Model Instance (Python)

Source: https://docs.langchain.com/oss/python/langchain/agents

This example shows how to initialize a LangChain agent by passing a pre-configured chat model instance, such as `ChatOpenAI`. This approach offers fine-grained control over model parameters like temperature, max tokens, and timeouts, allowing for custom model behavior.

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-5",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
    # ... (other params)
)
agent = create_agent(model, tools=tools)
```

--------------------------------

### Execute Multi-Domain Request with Supervisor Agent Routing

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Demonstrates handling a complex user request that spans multiple domains (calendar and email). The supervisor agent analyzes the query, identifies both calendar scheduling and email notification requirements, calls both schedule_event and manage_email tools in sequence, then synthesizes the responses into a coherent final message.

```python
query = (
    "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
    "and send them an email reminder about reviewing the new mockups."
)

for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()
```

--------------------------------

### Python Example for Iterative Querying a LangChain RAG Agent

Source: https://docs.langchain.com/oss/python/langchain/rag

This Python example shows how to interact with a LangChain RAG agent by providing a multi-step `query` and streaming the agent's responses. The agent processes the query iteratively, making multiple calls to its retrieval tool as needed to gather all necessary context before formulating a final answer, demonstrating its reasoning and retrieval capabilities.

```python
query = (
    "What is the standard method for Task Decomposition?\n\n"
    "Once you get the answer, look up common extensions of that method."
)

for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()
```

--------------------------------

### Wrap Subagent as Tool for Main Agent - Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

Demonstrates how to create a subagent and wrap it as a callable tool for the main agent to invoke. The subagent processes queries and returns results, while the main agent orchestrates which subagent to call and how to handle responses. Uses LangChain's tool decorator and agent creation utilities.

```python
from langchain.tools import tool
from langchain.agents import create_agent

# Create a subagent
subagent = create_agent(model="anthropic:claude-sonnet-4-20250514", tools=[...])

# Wrap it as a tool
@tool("research", description="Research a topic and return findings")
def call_research_agent(query: str):
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

# Main agent with subagent as a tool
main_agent = create_agent(model="anthropic:claude-sonnet-4-20250514", tools=[call_research_agent])
```

--------------------------------

### Agent Invocation and Result Processing - Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Example usage demonstrating how to invoke the agent with a user query using a unique thread ID for conversation tracking. The result contains messages that are formatted and printed, showing how the agent processes SQL query requests through the skill middleware.

```python
if __name__ == "__main__":
    # Configuration for this conversation thread
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Ask for a SQL query
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a SQL query to find all customers "
                        "who made orders over $1000 in the last month"
                    ),
                }
            ]
        },
        config
    )

    # Print the conversation
    for message in result["messages"]:
        if hasattr(message, 'pretty_print'):
            message.pretty_print()
        else:
            print(f"{message.type}: {message.content}")
```

--------------------------------

### Configure Conversational Agent with Search Tool (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router

This Python code initializes a conversational agent using LangChain's `create_agent` function. It defines the agent's core model, registers `search_docs` as an available tool, and sets an initial prompt to guide the agent's behavior as a helpful assistant.

```python
conversational_agent = create_agent(
    model,
    tools=[search_docs],
    prompt="You are a helpful assistant. Use search_docs to answer questions."
)
```

--------------------------------

### Execute Single-Domain Calendar Scheduling with Supervisor Agent

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Demonstrates streaming a user query through a supervisor agent to schedule a single calendar event. The supervisor identifies the task as calendar-related, invokes the schedule_event tool, and returns the result. This example shows the basic pattern of user input, agent streaming, and message handling in LangChain.

```python
query = "Schedule a team standup for tomorrow at 9am"

for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()
```

--------------------------------

### Create Agent with LLM Tool Selector Middleware

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Configures an agent with LLMToolSelectorMiddleware to intelligently select up to 3 most relevant tools from a larger tool set, with 'search' always included. The middleware uses gpt-4o-mini for tool selection while the main agent uses gpt-4o. This approach reduces token usage by filtering irrelevant tools before the main model processes the query.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[tool1, tool2, tool3, tool4, tool5, ...],
    middleware=[
        LLMToolSelectorMiddleware(
            model="gpt-4o-mini",
            max_tools=3,
            always_include=["search"],
        ),
    ],
)
```

--------------------------------

### Query Notion Agent with LangChain in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This Python function queries a Notion-specific agent to retrieve information. It accepts an 'AgentInput' object with the user's query, invokes the 'notion_agent', and then structures the agent's response, extracting the relevant content from the last message.

```python
  def query_notion(state: AgentInput) -> dict:
      """Query the Notion agent."""
      result = notion_agent.invoke({
          "messages": [{"role": "user", "content": state["query"]}]
      })
      return {"results": [{"source": "notion", "result": result["messages"][-1].content}]}
```

--------------------------------

### Create Specialized Agents with Domain-Specific Tools and Prompts

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

Initialize three specialized agents (GitHub, Notion, Slack) using create_agent, each configured with domain-specific tools and optimized system prompts. All agents use the same GPT-4o model but differ in their knowledge domains and available tools.

```python
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

model = init_chat_model("openai:gpt-4o")

github_agent = create_agent(
    model,
    tools=[search_code, search_issues, search_prs],
    system_prompt=(
        "You are a GitHub expert. Answer questions about code, "
        "API references, and implementation details by searching "
        "repositories, issues, and pull requests."
    ),
)

notion_agent = create_agent(
    model,
    tools=[search_notion, get_page],
    system_prompt=(
        "You are a Notion expert. Answer questions about internal "
        "processes, policies, and team documentation by searching "
        "the organization's Notion workspace."
    ),
)

slack_agent = create_agent(
    model,
    tools=[search_slack, get_thread],
    system_prompt=(
        "You are a Slack expert. Answer questions by searching "
        "relevant threads and discussions where team members have "
        "shared knowledge and solutions."
    ),
)
```

--------------------------------

### Define System Prompt for Weather Forecasting Agent (Python)

Source: https://docs.langchain.com/oss/python/langchain/quickstart

This code defines the system prompt for a weather forecasting agent. It sets the agent's persona as an expert weather forecaster who speaks in puns and informs it about the available tools (`get_weather_for_location`, `get_user_location`). The prompt also instructs the agent on how to handle user location requests.

```python
SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""
```

--------------------------------

### LangChain Agent SQL Database Interaction Log

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

This log details the steps an agent takes to answer a complex SQL-related question. It showcases the agent's reasoning process, including listing tables, fetching schema, validating generated SQL queries, executing them, and formulating a final natural language response based on the results. This output demonstrates the agent's internal workings and its ability to interact with a SQL database through defined tools.

```text
================================ Human Message =================================

Which genre on average has the longest tracks?
================================== Ai Message ==================================
Tool Calls:
  sql_db_list_tables (call_BQsWg8P65apHc8BTJ1NPDvnM)
 Call ID: call_BQsWg8P65apHc8BTJ1NPDvnM
  Args:
================================= Tool Message =================================
Name: sql_db_list_tables

Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track
================================== Ai Message ==================================
Tool Calls:
  sql_db_schema (call_i89tjKECFSeERbuACYm4w0cU)
 Call ID: call_i89tjKECFSeERbuACYm4w0cU
  Args:
    table_names: Track, Genre
================================= Tool Message =================================
Name: sql_db_schema


CREATE TABLE "Genre" (
	"GenreId" INTEGER NOT NULL,
	"Name" NVARCHAR(120),
	PRIMARY KEY ("GenreId")
)

/*
3 rows from Genre table:
GenreId	Name
1	Rock
2	Jazz
3	Metal
*/


CREATE TABLE "Track" (
	"TrackId" INTEGER NOT NULL,
	"Name" NVARCHAR(200) NOT NULL,
	"AlbumId" INTEGER,
	"MediaTypeId" INTEGER NOT NULL,
	"GenreId" INTEGER,
	"Composer" NVARCHAR(220),
	"Milliseconds" INTEGER NOT NULL,
	"Bytes" INTEGER,
	"UnitPrice" NUMERIC(10, 2) NOT NULL,
	PRIMARY KEY ("TrackId"),
	FOREIGN KEY("MediaTypeId") REFERENCES "MediaType" ("MediaTypeId"),
	FOREIGN KEY("GenreId") REFERENCES "Genre" ("GenreId"),
	FOREIGN KEY("AlbumId") REFERENCES "Album" ("AlbumId")
)

/*
3 rows from Track table:
TrackId	Name	AlbumId	MediaTypeId	GenreId	Composer	Milliseconds	Bytes	UnitPrice
1	For Those About To Rock (We Salute You)	1	1	1	Angus Young, Malcolm Young, Brian Johnson	343719	11170334	0.99
2	Balls to the Wall	2	2	1	U. Dirkschneider, W. Hoffmann, H. Frank, P. Baltes, S. Kaufmann, G. Hoffmann	342562	5510424	0.99
3	Fast As a Shark	3	2	1	F. Baltes, S. Kaufman, U. Dirkscneider & W. Hoffman	230619	3990994	0.99
*/
================================== Ai Message ==================================
Tool Calls:
  sql_db_query_checker (call_G64yYm6R6UauiVPCXJZMA49b)
 Call ID: call_G64yYm6R6UauiVPCXJZMA49b
  Args:
    query: SELECT Genre.Name, AVG(Track.Milliseconds) AS AverageLength FROM Track INNER JOIN Genre ON Track.GenreId = Genre.GenreId GROUP BY Genre.Name ORDER BY AverageLength DESC LIMIT 5;
================================= Tool Message =================================
Name: sql_db_query_checker

SELECT Genre.Name, AVG(Track.Milliseconds) AS AverageLength FROM Track INNER JOIN Genre ON Track.GenreId = Genre.GenreId GROUP BY Genre.Name ORDER BY AverageLength DESC LIMIT 5;
================================== Ai Message ==================================
Tool Calls:
  sql_db_query (call_AnO3SrhD0ODJBxh6dHMwvHwZ)
 Call ID: call_AnO3SrhD0ODJBxh6dHMwvHwZ
  Args:
    query: SELECT Genre.Name, AVG(Track.Milliseconds) AS AverageLength FROM Track INNER JOIN Genre ON Track.GenreId = Genre.GenreId GROUP BY Genre.Name ORDER BY AverageLength DESC LIMIT 5;
================================= Tool Message =================================
Name: sql_db_query

[('Sci Fi & Fantasy', 2911783.0384615385), ('Science Fiction', 2625549.076923077), ('Drama', 2575283.78125), ('TV Shows', 2145041.0215053763), ('Comedy', 1585263.705882353)]
================================== Ai Message ==================================

On average, the genre with the longest tracks is "Sci Fi & Fantasy" with an average track length of approximately 2,911,783 milliseconds. This is followed by "Science Fiction," "Drama," "TV Shows," and "Comedy."
```

--------------------------------

### Create Agent with JSON Schema ToolStrategy

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Implements structured output using JSON Schema dictionary specification. Provides flexible schema definition with explicit type declarations and descriptions. Supports complex schema structures with property definitions.

```python
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


product_review_schema = {
    "type": "object",
    "description": "Analysis of a product review.",
    "properties": {
        "rating": {
            "type": ["integer", "null"]
```

--------------------------------

### Invoke Notion Agent with User Query (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

The `query_notion` function acts as an interface for querying the Notion-specific agent. It accepts an `AgentInput` state with a user's sub-question and calls the `notion_agent` with it. The function structures the query as a message for the agent and extracts the content of the agent's response, associating it with the 'notion' source.

```python
def query_notion(state: AgentInput) -> dict:
    """Query the Notion agent."""
    result = notion_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]  # [!code highlight]
    })
    return {"results": [{"source": "notion", "result": result["messages"][-1].content}]}
```

--------------------------------

### Configure LangChain Agent with Human-in-the-Loop Middleware

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

This Python snippet demonstrates how to initialize a LangChain agent to pause for human review before executing specific tool calls, such as 'sql_db_query'. It utilizes `HumanInTheLoopMiddleware` to define interruption conditions and includes an `InMemorySaver` checkpointer to enable pausing and resuming agent state.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Update Agent State and Switch Agents on Task Completion using Python Interceptors

Source: https://docs.langchain.com/oss/python/langchain/mcp

This interceptor demonstrates how to update the agent's state and redirect execution flow to another agent (e.g., 'summary_agent') after a specific task, such as 'submit_order', is completed. It utilizes `Command` objects to modify the agent's internal state and control graph execution.

```python
from langchain.agents import AgentState, create_agent
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from langchain.messages import ToolMessage
from langgraph.types import Command

async def handle_task_completion(
    request: MCPToolCallRequest,
    handler,
):
    """Mark task complete and hand off to summary agent."""
    result = await handler(request)

    if request.name == "submit_order":
        return Command(
            update={
                "messages": [result] if isinstance(result, ToolMessage) else [],
                "task_status": "completed",  # [!code highlight]
            },
            goto="summary_agent",  # [!code highlight]
        )

    return result
```

--------------------------------

### Define LangChain Agent Tools (Python)

Source: https://docs.langchain.com/oss/python/langchain/voice-agent

This code snippet demonstrates how to define custom tools for a LangChain agent. The `add_to_order` and `confirm_order` functions serve as examples of actions an agent can perform, taking specific inputs and returning a string response. These tools are crucial for enabling an agent to interact with external systems or specific business logic.

```python
from uuid import uuid4
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Define agent tools
def add_to_order(item: str, quantity: int) -> str:
    """Add an item to the customer's sandwich order."""
    return f"Added {quantity} x {item} to the order."

def confirm_order(order_summary: str) -> str:
    """Confirm the final order with the customer."""
    return f"Order confirmed: {order_summary}. Sending to kitchen."
```

--------------------------------

### Initialize SQL Agent with LangChain and GPT-4

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

Create a Python file (sql_agent.py) that initializes a SQL agent with GPT-4, downloads a sample Chinook database, configures SQL toolkit, and defines system prompts. The agent is designed to generate syntactically correct SQL queries and interact safely with the database while preventing DML operations.

```python
#sql_agent.py for studio
import pathlib

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
import requests


# Initialize an LLM
model = init_chat_model("gpt-4.1")

# Get the database, store it locally
url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"{local_path} already exists, skipping download.")
else:
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"File downloaded and saved as {local_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

# Create the tools
toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")

# Use create_agent
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_agent(
    model,
    tools,
    system_prompt=system_prompt,
)
```

--------------------------------

### Invoke Agent with Backward Transition Correction

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Execute the agent with a user message indicating a correction is needed. The agent detects the correction request and automatically calls the appropriate go_back tool to restart the warranty verification step, allowing the user to provide corrected information.

```python
result = agent.invoke(
    {"messages": [HumanMessage("Actually, I made a mistake - my device is out of warranty")]},
    config
)
# Agent will call go_back_to_warranty and restart the warranty verification step
```

--------------------------------

### Create Agent with JSON Schema Structured Output

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Creates a LangChain agent that uses JSON schema to enforce structured output format for product review analysis. The schema defines required fields like sentiment and key_points with type validation and constraints. The agent processes user input and returns a structured response matching the defined schema.

```python
agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(product_review_schema)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})
result["structured_response"]
# {'rating': 5, 'sentiment': 'positive', 'key_points': ['fast shipping', 'expensive']}
```

--------------------------------

### Resume LangChain Agent Execution After Human Approval

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

This Python snippet demonstrates how to resume a LangChain agent's execution after it has been paused for human review. It uses the `Command` object from `langgraph.types` to send a 'resume' instruction, explicitly approving the pending action, thereby allowing the agent to continue its workflow.

```python
from langgraph.types import Command

for step in agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config,
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    elif "__interrupt__" in step:
        print("INTERRUPTED:")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
    else:
        pass
```

--------------------------------

### Generate Dynamic System Prompt with LangChain `dynamic_prompt` Middleware

Source: https://docs.langchain.com/oss/python/langchain/agents

This code demonstrates how to use the `@dynamic_prompt` decorator to create middleware that dynamically adjusts the system prompt based on runtime context, such as a user's role. It defines a `Context` TypedDict and a `user_role_prompt` function to provide role-specific instructions to the agent.

```python
from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent = create_agent(
    model="gpt-4o",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)
```

--------------------------------

### Test LangChain Calendar Agent with Natural Language Query (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

This Python code demonstrates how to test the `calendar_agent` by providing a natural language query for scheduling a team meeting. It iterates through the agent's execution steps using `agent.stream()`, allowing observation of tool calls for checking availability and creating the calendar event, and prints each message for debugging and understanding the agent's flow.

```python
query = "Schedule a team meeting next Tuesday at 2pm for 1 hour"

for step in calendar_agent.stream(
    {"messages": [{"role": "user", "content": query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()
```

--------------------------------

### Create Agent with Middleware in Python

Source: https://docs.langchain.com/oss/python/langchain/middleware

Initialize a LangChain agent with multiple middleware instances using the create_agent function. This example demonstrates adding SummarizationMiddleware and HumanInTheLoopMiddleware to control agent behavior for logging, transformation, and human oversight. The middleware parameter accepts a list of middleware objects that intercept agent operations.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(...),
        HumanInTheLoopMiddleware(...)
    ],
)
```

--------------------------------

### Route After Agent Based on Tool Calls

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

Conditional routing function that determines whether to end the agent session or route to another agent. It checks if the last message is an AIMessage without tool calls to determine if the agent has completed its work. Returns the appropriate routing destination as a literal string.

```python
) -> Literal["sales_agent", "support_agent", "__end__"]:
    """Route based on active_agent, or END if the agent finished without handoff."""
    messages = state.get("messages", [])

    # Check the last message - if it's an AIMessage without tool calls, we're done
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
            return "__end__"

    # Otherwise route to the active agent
    active = state.get("active_agent", "sales_agent")
    return active if active else "sales_agent"
```

--------------------------------

### Initialize LangChain agent with skill middleware and checkpointer

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Creates a LangChain agent configured with the SkillMiddleware for skill description injection and an InMemorySaver checkpointer for state persistence. The agent receives a system prompt explaining its role as a SQL query assistant and gains access to skill descriptions through the middleware.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

# Create the agent with skill support
agent = create_agent(
    model,
    system_prompt=(
        "You are a SQL query assistant that helps users "
        "write queries against business databases."
    ),
    middleware=[SkillMiddleware()],
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Resume Agent with Approval Decision

Source: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

Resume a paused agent conversation by providing approval decisions for tool calls. Uses the Command wrapper to specify decision types (approve, edit, reject) and maintains conversation context through thread ID configuration. Requires the same config with thread_id to resume the paused conversation.

```python
agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}
    ),
    config=config
)
```

--------------------------------

### Create Agent with TypedDict ToolStrategy

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Implements structured output using TypedDict from typing_extensions. Provides dictionary-based type hints for schema definition with inline field documentation. Returns structured response as dictionary.

```python
from typing import Literal
from typing_extensions import TypedDict
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductReview(TypedDict):
    """Analysis of a product review."""
    rating: int | None  # The rating of the product (1-5)
    sentiment: Literal["positive", "negative"]  # The sentiment of the review
    key_points: list[str]  # The key points of the review

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})
result["structured_response"]
# {'rating': 5, 'sentiment': 'positive', 'key_points': ['fast shipping', 'expensive']}
```

--------------------------------

### Wrap Stateless Router as Tool for Conversational Agent

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router

Wraps a stateless router workflow as a tool that a conversational agent can invoke. The conversational agent maintains memory and context while the router remains stateless, simplifying conversation history management across parallel agents.

```python
@tool
def search_docs(query: str) -> str:
    """Search across multiple documentation sources."""
    result = workflow.invoke({"query": query})  # [!code highlight]
    return result["final_answer"]
```

--------------------------------

### Configure LangGraph JSON Configuration

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

Create a langgraph.json configuration file that defines graph dependencies, entry points, and environment variable location. This file is required in the directory where the agent will run and specifies which Python files and functions define the agent and graph.

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./sql_agent.py:agent",
    "graph": "./sql_agent_langgraph.py:graph"
  },
  "env": ".env"
}
```

--------------------------------

### Stream Multi-Agent Messages with Agent Tracking in Python

Source: https://docs.langchain.com/oss/python/langchain/streaming

Implements streaming logic that processes both message chunks and updates from multiple agents, tracking which agent is active through metadata tags. The code renders token chunks and completed tool calls while maintaining the current agent context across the stream.

```python
def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")


input_message = {"role": "user", "content": "What is the weather in Boston?"}
current_agent = None
for _, stream_mode, data in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
    subgraphs=True,
):
    if stream_mode == "messages":
        token, metadata = data
        if tags := metadata.get("tags", []):
            this_agent = tags[0]
            if this_agent != current_agent:
                print(f"🤖 {this_agent}: ")
                current_agent = this_agent
        if isinstance(token, AIMessage):
            _render_message_chunk(token)
    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
```

--------------------------------

### Extended Agentic RAG with Documentation Tool - Python

Source: https://docs.langchain.com/oss/python/langchain/retrieval

Implements a comprehensive Agentic RAG system for querying LangGraph documentation. The agent loads an llms.txt file containing documentation URLs, validates domain access, and uses a `fetch_documentation` tool to retrieve markdown-converted content while enforcing security constraints.

```python
import requests
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from markdownify import markdownify


ALLOWED_DOMAINS = ["https://langchain-ai.github.io/"]
LLMS_TXT = 'https://langchain-ai.github.io/langgraph/llms.txt'


@tool
def fetch_documentation(url: str) -> str:
    """Fetch and convert documentation from a URL"""
    if not any(url.startswith(domain) for domain in ALLOWED_DOMAINS):
        return (
            "Error: URL not allowed. "
            f"Must start with one of: {', '.join(ALLOWED_DOMAINS)}"
        )
    response = requests.get(url, timeout=10.0)
    response.raise_for_status()
    return markdownify(response.text)


llms_txt_content = requests.get(LLMS_TXT).text

system_prompt = f"""
You are an expert Python developer and technical assistant.
Your primary role is to help users with questions about LangGraph and related tools.

Instructions:

1. If a user asks a question you're unsure about — or one that likely involves API usage,
   behavior, or configuration — you MUST use the `fetch_documentation` tool to consult the relevant docs.
2. When citing documentation, summarize clearly and include relevant context from the content.
3. Do not use any URLs outside of the allowed domain.
4. If a documentation fetch fails, tell the user and proceed with your best expert understanding.

You can access official documentation from the following approved sources:

{llms_txt_content}

You MUST consult the documentation to get up to date documentation
before answering a user's question about LangGraph.

Your answers should be clear, concise, and technically accurate.
"""

tools = [fetch_documentation]

model = init_chat_model("claude-sonnet-4-0", max_tokens=32_000)

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    name="Agentic RAG",
)

response = agent.invoke({
    'messages': [
        HumanMessage(content=(
            "Write a short example of a langgraph agent using the "
            "prebuilt create react agent. the agent should be able "
            "to look up stock pricing information."
        ))
    ]
})

print(response['messages'][-1].content)
```

--------------------------------

### Create Agent with Dataclass ToolStrategy

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Implements structured output using Python dataclass with type annotations. Defines ProductReview with inline field documentation. Demonstrates equivalent functionality to Pydantic implementation with simpler syntax.

```python
from dataclasses import dataclass
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


@dataclass
class ProductReview:
    """Analysis of a product review."""
    rating: int | None  # The rating of the product (1-5)
    sentiment: Literal["positive", "negative"]  # The sentiment of the review
    key_points: list[str]  # The key points of the review

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ToolStrategy(ProductReview)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze this review: 'Great product: 5 out of 5 stars. Fast shipping, but expensive'"}]
})
result["structured_response"]
# ProductReview(rating=5, sentiment='positive', key_points=['fast shipping', 'expensive'])
```

--------------------------------

### Invoke GitHub Agent with User Query (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

The `query_github` function serves as an interface to the GitHub-specific agent. It takes an `AgentInput` state containing a user's sub-question and invokes the `github_agent` with it. The function formats the query as a message for the agent and extracts the content of the agent's final response, wrapping it with the 'github' source identifier.

```python
def query_github(state: AgentInput) -> dict:
    """Query the GitHub agent."""
    result = github_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]  # [!code highlight]
    })
    return {"results": [{"source": "github", "result": result["messages"][-1].content}]}
```

--------------------------------

### Create Agent with Step-Based Configuration

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Initializes an agent with a state schema, middleware for step configuration, and a checkpointer for maintaining state across conversation turns. The checkpointer ensures the current_step state persists between user messages.

```python
# Collect all tools from all step configurations
all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
]

# Create the agent with step-based configuration
agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Invoke Slack Agent with User Query (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

The `query_slack` function provides an interface for interacting with the Slack-specific agent. It receives an `AgentInput` state containing a user's sub-question and invokes the `slack_agent`. The function formats the query as a message for the agent and retrieves the content from the agent's final response, tagging it with the 'slack' source identifier.

```python
def query_slack(state: AgentInput) -> dict:
    """Query the Slack agent."""
    result = slack_agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]  # [!code highlight]
    })
    return {"results": [{"source": "slack", "result": result["messages"][-1].content}]}
```

--------------------------------

### Handle Interrupts and Resume Agent Execution

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Demonstrates handling interrupt points in agent execution by editing or approving actions before resuming. The code shows how to modify interrupt values (such as editing email subjects) and approve actions, then resume the supervisor agent with the modified decisions.

```python
if interrupt_.id == "2b56f299be313ad8bc689eff02973f16":
    # Edit email
    edited_action = interrupt_.value["action_requests"][0].copy()
    edited_action["arguments"]["subject"] = "Mockups reminder"
    resume[interrupt_.id] = {
        "decisions": [{"type": "edit", "edited_action": edited_action}]
    }
else:
    resume[interrupt_.id] = {"decisions": [{"type": "approve"}]}

interrupts = []
for step in supervisor_agent.stream(
    Command(resume=resume),
    config,
):
    for update in step.values():
        if isinstance(update, dict):
            for message in update.get("messages", []):
                message.pretty_print()
        else:
            interrupt_ = update[0]
            interrupts.append(interrupt_)
            print(f"\nINTERRUPTED: {interrupt_.id}")
```

--------------------------------

### Implement LangChain Agents with Native Structured Output Using Various Schema Types

Source: https://docs.langchain.com/oss/python/langchain/structured-output

These examples demonstrate how to create a LangChain agent that leverages the ProviderStrategy for native structured output. By passing different schema types (Pydantic Model, Dataclass, TypedDict, or JSON Schema) directly to `create_agent`'s `response_format` parameter, the agent is configured to return responses conforming to the specified structure. This method ensures reliable and validated output from models supporting native structured output, simplifying data extraction and processing.

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent


class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

print(result["structured_response"])
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

```python
from dataclasses import dataclass
from langchain.agents import create_agent


@dataclass
class ContactInfo:
    """Contact information for a person."""
    name: str # The name of the person
    email: str # The email address of the person
    phone: str # The phone number of the person

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

```python
from typing_extensions import TypedDict
from langchain.agents import create_agent


class ContactInfo(TypedDict):
    """Contact information for a person."""
    name: str # The name of the person
    email: str # The email address of the person
    phone: str # The phone number of the person

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.example.com, (555) 123-4567"}]
})

result["structured_response"]
# {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
```

```python
from langchain.agents import create_agent


contact_info_schema = {
    "type": "object",
    "description": "Contact information for a person.",
    "properties": {
        "name": {"type": "string", "description": "The name of the person"},
        "email": {"type": "string", "description": "The email address of the person"},
        "phone": {"type": "string", "description": "The phone number of the person"}
    },
    "required": ["name", "email", "phone"]
}

agent = create_agent(
    model="gpt-5",
    tools=tools,
    response_format=ProviderStrategy(contact_info_schema)
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
```

--------------------------------

### Initialize LangChain Calendar Scheduling Agent (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

This Python snippet initializes a specialized calendar scheduling agent using `create_agent` from LangChain. It defines `CALENDAR_AGENT_PROMPT` to instruct the agent on its role and how to handle natural language scheduling requests, then configures it with the previously defined `create_calendar_event` and `get_available_time_slots` tools.

```python
from langchain.agents import create_agent


CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response."
)

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
)
```

--------------------------------

### Handle Schema Validation Errors with Pydantic Models

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Creates a LangChain agent that validates structured output against a Pydantic schema with constraints. When the model provides invalid data (rating value of 10 exceeding the 1-5 range), the agent returns specific validation error feedback and prompts the model to correct the mistake. The ToolStrategy automatically handles schema validation errors with retry logic.

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ProductRating(BaseModel):
    rating: int | None = Field(description="Rating from 1-5", ge=1, le=5)
    comment: str = Field(description="Review comment")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(ProductRating),  # Default: handle_errors=True
    system_prompt="You are a helpful assistant that parses product reviews. Do not make any field or value up."
)

agent.invoke({
    "messages": [{"role": "user", "content": "Parse this: Amazing product, 10/10!"}]
})
```

--------------------------------

### Implement Step-Based Middleware for LangChain Agent in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

This Python function, `apply_step_config`, acts as middleware for a LangChain agent, dynamically adjusting the agent's behavior based on the current workflow step. It reads the `current_step` from the agent's state, validates required state variables, formats the system prompt with contextual information, and injects step-specific tools, overriding the agent's default configuration for that particular interaction.

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@wrap_model_call  # [!code highlight]
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent behavior based on the current step."""
    # Get current step (defaults to warranty_collector for first interaction)
    current_step = request.state.get("current_step", "warranty_collector")  # [!code highlight]

    # Look up step configuration
    stage_config = STEP_CONFIG[current_step]  # [!code highlight]

    # Validate required state exists
    for key in stage_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    # Format prompt with state values (supports {warranty_status}, {issue_type}, etc.)
    system_prompt = stage_config["prompt"].format(**request.state)

    # Inject system prompt and step-specific tools
    request = request.override(  # [!code highlight]
        system_prompt=system_prompt,  # [!code highlight]
        tools=stage_config["tools"],  # [!code highlight]
    )

    return handler(request)
```

--------------------------------

### Clone and Setup Agent Chat UI Repository

Source: https://docs.langchain.com/oss/python/langchain/ui

Clone the Agent Chat UI repository from GitHub and set up the development environment locally. This approach allows for customization and direct access to source code. Requires Git, Node.js, and pnpm installed.

```bash
# Clone the repository
git clone https://github.com/langchain-ai/agent-chat-ui.git
cd agent-chat-ui

# Install dependencies and start
pnpm install
pnpm dev
```

--------------------------------

### Run Agent Evaluations with LangSmith's `evaluate` Function in Python

Source: https://docs.langchain.com/oss/python/langchain/test

This Python example demonstrates using LangSmith's `Client().evaluate` function to run agent evaluations. It initializes a LangSmith client, defines an agent execution function, and then performs an evaluation against a named dataset using specified evaluators, with results automatically logged to LangSmith.

```python
from langsmith import Client
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

client = Client()

trajectory_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

def run_agent(inputs):
    """Your agent function that returns trajectory messages."""
    return agent.invoke(inputs)["messages"]

experiment_results = client.evaluate(
    run_agent,
    data="your_dataset_name",
    evaluators=[trajectory_evaluator]
)
```

--------------------------------

### Route Classified Queries to Agents for Parallel Execution (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This `route_to_agents` function is responsible for dynamically routing classified sub-questions to the appropriate agents. It processes the list of classifications generated by the LLM and creates a list of `Send` objects, each targeting a specific agent (e.g., 'github', 'notion', 'slack') with its respective sub-query. This enables parallel invocation of multiple agents based on the initial query classification.

```python
def route_to_agents(state: RouterState) -> list[Send]:
    """Fan out to agents based on classifications."""
    return [
        Send(c["source"], {"query": c["query"]})  # [!code highlight]
        for c in state["classifications"]
    ]
```

--------------------------------

### Create Supervisor Agent for Multi-Domain Orchestration

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Initialize a supervisor agent with a system prompt that instructs it to coordinate multiple domain-specific tools (schedule_event and manage_email). The supervisor breaks down complex user requests into appropriate sequential tool calls, enabling coordination across multiple sub-agents while maintaining abstraction at the domain level rather than individual API operations.

```python
SUPERVISOR_PROMPT = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)

supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
)
```

--------------------------------

### Route to Multiple Agents in Parallel with Send

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router

Classifies a query and routes it to multiple specialized agents in parallel using the Send method. Enables fan-out execution where multiple agents process the query simultaneously and results are later synthesized.

```python
from typing import TypedDict
from langgraph.types import Send

class ClassificationResult(TypedDict):
    query: str
    agent: str

def classify_query(query: str) -> list[ClassificationResult]:
    """Use LLM to classify query and determine which agents to invoke."""
    # Classification logic here
    ...

def route_query(state: State):
    """Route to relevant agents based on query classification."""
    classifications = classify_query(state["query"])

    # Fan out to selected agents in parallel
    return [
        Send(c["agent"], {"query": c["query"]})
        for c in classifications
    ]
```

--------------------------------

### Create Agent with Custom Tool Message Content

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Creates a LangChain agent with customized tool message content that displays user-friendly feedback in the conversation history. Uses ToolStrategy with a schema parameter and tool_message_content parameter to define the message shown when structured output is generated.

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

class MeetingAction(BaseModel):
    """Action items extracted from a meeting transcript."""
    task: str = Field(description="The specific task to be completed")
    assignee: str = Field(description="Person responsible for the task")
    priority: Literal["low", "medium", "high"] = Field(description="Priority level")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
        schema=MeetingAction,
        tool_message_content="Action item captured and added to meeting notes!"
    )
)

agent.invoke({
    "messages": [{"role": "user", "content": "From our meeting: Sarah needs to update the project timeline as soon as possible"}]
})
```

--------------------------------

### Initialize SQL Agent System Prompt in Python

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

Creates a descriptive system prompt that customizes the behavior of a SQL database agent. The prompt instructs the agent to generate syntactically correct SQL queries, limit results, and includes safety guidelines preventing DML statements. Template variables are formatted with database dialect and result limit parameters.

```python
system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)
```

--------------------------------

### Define Structured Response Format for Agent (Python)

Source: https://docs.langchain.com/oss/python/langchain/quickstart

This code defines a structured response format using a `dataclass` named `ResponseFormat`. It specifies that agent responses should include a `punny_response` (required) and optionally `weather_conditions`, ensuring predictable and schema-compliant output from the agent.

```python
from dataclasses import dataclass

# We use a dataclass here, but Pydantic models are also supported.
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None
```

--------------------------------

### Stream Agent Execution and Collect Interrupt Events

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Executes the supervisor agent with streaming to capture interrupt events during tool execution. Collects all interrupts into a list for downstream processing and displays messages and interrupt IDs as they occur.

```python
query = (
    "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
    "and send them an email reminder about reviewing the new mockups."
)

config = {"configurable": {"thread_id": "6"}}

interrupts = []
for step in supervisor_agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    config,
):
    for update in step.values():
        if isinstance(update, dict):
            for message in update.get("messages", []):
                message.pretty_print()
        else:
            interrupt_ = update[0]
            interrupts.append(interrupt_)
            print(f"\nINTERRUPTED: {interrupt_.id}")
```

--------------------------------

### Retrieve User Data from Agent Store (Python)

Source: https://docs.langchain.com/oss/python/langchain/long-term-memory

Accesses the agent's internal 'store' directly to retrieve a specific user's value. The method uses a tuple for the category key (e.g., 'users') and the user ID to fetch the associated data, demonstrating direct state access.

```python
store.get(("users",), "user_123").value
```

--------------------------------

### Complete LangGraph Multi-Agent System with Sales and Support Handoffs

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

This comprehensive Python example illustrates a multi-agent system in LangGraph featuring sales and support agents. It defines a shared state (`MultiAgentState`), implements `transfer_to_sales` and `transfer_to_support` tools for agent handoffs, configures the agents with specific prompts and tools, and provides functions for invoking these agents as graph nodes.

```python
from typing import Literal

from langchain.agents import AgentState, create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import NotRequired


# 1. Define state with active_agent tracker
class MultiAgentState(AgentState):
    active_agent: NotRequired[str]


# 2. Create handoff tools
@tool
def transfer_to_sales(
    runtime: ToolRuntime,
) -> Command:
    """Transfer to the sales agent."""
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="Transferred to sales agent from support agent",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="sales_agent",
        update={
            "active_agent": "sales_agent",
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )


@tool
def transfer_to_support(
    runtime: ToolRuntime,
) -> Command:
    """Transfer to the support agent."""
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="Transferred to support agent from sales agent",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="support_agent",
        update={
            "active_agent": "support_agent",
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )


# 3. Create agents with handoff tools
sales_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[transfer_to_support],
    system_prompt="You are a sales agent. Help with sales inquiries. If asked about technical issues or support, transfer to the support agent.",
)

support_agent = create_agent(
    model="anthropic:claude-sonnet-4-20250514",
    tools=[transfer_to_sales],
    system_prompt="You are a support agent. Help with technical issues. If asked about pricing or purchasing, transfer to the sales agent.",
)


# 4. Create agent nodes that invoke the agents
def call_sales_agent(state: MultiAgentState) -> Command:
    """Node that calls the sales agent."""
    response = sales_agent.invoke(state)
    return response


def call_support_agent(state: MultiAgentState) -> Command:
    """Node that calls the support agent."""
    response = support_agent.invoke(state)
    return response


# 5. Create router that checks if we should end or continue
def route_after_agent(
    state: MultiAgentState,
```

--------------------------------

### Create and Invoke LangChain Agent in LangGraph Node

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow

Demonstrates how to create a LangChain agent and invoke it within a LangGraph node, combining the flexibility of custom workflows with pre-built agents. This example shows a basic workflow setup with a single agent node that processes user queries and returns answers.

```python
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END

agent = create_agent(model="openai:gpt-4o", tools=[...])

def agent_node(state: State) -> dict:
    """A LangGraph node that invokes a LangChain agent."""
    result = agent.invoke({
        "messages": [{"role": "user", "content": state["query"]}]
    })
    return {"answer": result["messages"][-1].content}

# Build a simple workflow
workflow = (
    StateGraph(State)
    .add_node("agent", agent_node)
    .add_edge(START, "agent")
    .add_edge("agent", END)
    .compile()
)
```

--------------------------------

### Implement Custom Tool Error Handling for LangChain Python Agents

Source: https://docs.langchain.com/oss/python/langchain/agents

This example demonstrates how to implement custom error handling for tool calls within a LangChain agent using the `@wrap_tool_call` middleware. The `handle_tool_errors` function wraps the tool execution in a try-except block, catching any exceptions. Upon an error, it returns a `ToolMessage` with a custom, user-friendly error message, rather than propagating the raw exception, thereby providing a controlled response to the model.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

--------------------------------

### Define Agent State with LangChain `state_schema` (Python)

Source: https://docs.langchain.com/oss/python/langchain/agents

This example demonstrates using the `state_schema` parameter directly with `create_agent` to define custom state. This method serves as a shortcut for state primarily used by tools, though defining state via middleware is generally preferred for better conceptual scoping and advanced use cases.

```python
from langchain.agents import AgentState


class CustomState(AgentState):
    user_preferences: dict

agent = create_agent(
    model,
    tools=[tool1, tool2],
    state_schema=CustomState
)
# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
```

--------------------------------

### Query GitHub Agent with LangChain in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This function demonstrates how to interact with a pre-configured GitHub agent within a LangChain workflow. It takes an 'AgentInput' containing the user's query, invokes the 'github_agent' with the message, and extracts the content of the agent's final response for further processing.

```python
  def query_github(state: AgentInput) -> dict:
      """Query the GitHub agent."""
      result = github_agent.invoke({
          "messages": [{"role": "user", "content": state["query"]}]
      })
      return {"results": [{"source": "github", "result": result["messages"][-1].content}]}
```

--------------------------------

### Update agent state with Command in Python

Source: https://docs.langchain.com/oss/python/langchain/tools

These Python functions illustrate how to modify an agent's state using `Command` objects in LangGraph. `clear_conversation` removes all messages from the conversation history, while `update_user_name` changes a specific user name in the agent's state. Both tools return `Command` instances, which are then processed by the agent to effect the state change.

```python
from langgraph.types import Command
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.tools import tool, ToolRuntime

# Update the conversation history by removing all messages
@tool
def clear_conversation() -> Command:
    """Clear the conversation history."""

    return Command(
        update={
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
        }
    )

# Update the user_name in the agent state
@tool
def update_user_name(
    new_name: str,
    runtime: ToolRuntime
) -> Command:
    """Update the user's name."""
    return Command(update={"user_name": new_name})
```

--------------------------------

### Integrate TodoListMiddleware with LangChain Agent in Python

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This example demonstrates how to add the `TodoListMiddleware` to a LangChain agent. This middleware automatically equips agents with a `write_todos` tool and system prompts, enabling them to plan and track tasks for complex multi-step operations and long-running processes.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[read_file, write_file, run_tests],
    middleware=[TodoListMiddleware()],
)
```

--------------------------------

### Query Slack Agent with LangChain in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This function is designed to query a Slack agent within the LangChain framework. It takes the 'AgentInput' containing the query, invokes the 'slack_agent' to get information, and then processes the agent's response to return the extracted content from its final message.

```python
  def query_slack(state: AgentInput) -> dict:
      """Query the Slack agent."""
      result = slack_agent.invoke({
          "messages": [{"role": "user", "content": state["query"]}]
      })
      return {"results": [{"source": "slack", "result": result["messages"][-1].content}]}
```

--------------------------------

### Using InMemorySaver for State Persistence in LangGraph Python

Source: https://docs.langchain.com/oss/python/langchain/test

This Python example demonstrates how to use LangGraph's `InMemorySaver` checkpointer to enable persistence during agent testing. By initializing the agent with `InMemorySaver`, previous messages and state are retained across invocations, allowing for testing state-dependent agent behavior, such as responding based on prior conversational context.

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model,
    tools=[],
    checkpointer=InMemorySaver()
)

# First invocation
agent.invoke(HumanMessage(content="I live in Sydney, Australia."))

# Second invocation: the first message is persisted (Sydney location), so the model returns GMT+10 time
agent.invoke(HumanMessage(content="What's my local time?"))
```

--------------------------------

### Stream Agent with Human-in-the-Loop Interrupts - LangGraph Python

Source: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

Stream agent execution in real-time with human-in-the-loop handling using stream_mode=['updates', 'messages']. Processes both LLM tokens and agent state updates, detecting interrupts to pause execution for human review. Provides token-by-token output with interrupt handling capability.

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "some_id"}}

for mode, chunk in agent.stream(
    {"messages": [{"role": "user", "content": "Delete old records from the database"}]},
    config=config,
    stream_mode=["updates", "messages"],
):
    if mode == "messages":
        token, metadata = chunk
        if token.content:
            print(token.content, end="", flush=True)
    elif mode == "updates":
        if "__interrupt__" in chunk:
            print(f"\n\nInterrupt: {chunk['__interrupt__']}")
```

--------------------------------

### Stream agent progress with LangChain Python

Source: https://docs.langchain.com/oss/python/langchain/streaming

Demonstrates how to stream agent progress using the stream() method with stream_mode="updates". This example shows an agent making tool calls and receiving responses, with each step emitting state updates. The agent processes a weather query by calling a tool and returning the final response.

```python
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)
for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="updates",
):
    for step, data in chunk.items():
        print(f"step: {step}")
        print(f"content: {data['messages'][-1].content_blocks}")
```

--------------------------------

### Add In-Memory Checkpointer for Agent Memory (Python)

Source: https://docs.langchain.com/oss/python/langchain/quickstart

This snippet demonstrates how to add conversational memory to the agent using `InMemorySaver` from `langgraph.checkpoint.memory`. This allows the agent to maintain state across interactions, remembering previous conversations and context, though a persistent checkpointer is recommended for production.

```python
from langgraph.checkpoint.memory import InMemorySaver

checkpointer = InMemorySaver()
```

--------------------------------

### Create Agent with Summarization Middleware

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Extends the agent configuration with SummarizationMiddleware to compress message history when it exceeds 4000 tokens while preserving the last 10 messages for context. Prevents message history from growing unbounded across long conversations.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[
        apply_step_config,
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 10)
        )
    ],
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Configure HumanInTheLoopMiddleware for Agent Tool Interrupts

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Sets up human-in-the-loop middleware on calendar and email sub-agents to interrupt specific tool executions with custom description prefixes. Adds an InMemorySaver checkpointer to the supervisor agent to enable pause and resume functionality for the entire workflow.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"create_calendar_event": True},
            description_prefix="Calendar event pending approval",
        ),
    ],
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"send_email": True},
            description_prefix="Outbound email pending approval",
        ),
    ],
)

supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Install LangGraph CLI for Studio

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

Install the LangGraph CLI with in-memory support required to run the agent in LangSmith Studio. This package enables the client-side loop and memory functionality for chat interfaces.

```shell
pip install -U langgraph-cli[inmem]>=0.4.0
```

--------------------------------

### Integrate Agent Evaluators with LangSmith using Pytest in Python

Source: https://docs.langchain.com/oss/python/langchain/test

This Python example demonstrates integrating `agentevals` with LangSmith via Pytest. A Pytest-marked function invokes an agent, logs inputs, outputs, and reference outputs using `langsmith.testing`, and then applies a trajectory evaluator, with results sent to LangSmith.

```python
import pytest
from langsmith import testing as t
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT

trajectory_evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

@pytest.mark.langsmith
def test_trajectory_accuracy():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in SF?")]
    })

    reference_trajectory = [
        HumanMessage(content="What's the weather in SF?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "SF"}},
        ]),
        ToolMessage(content="It's 75 degrees and sunny in SF.", tool_call_id="call_1"),
        AIMessage(content="The weather in SF is 75 degrees and sunny."),
    ]

    # Log inputs, outputs, and reference outputs to LangSmith
    t.log_inputs({})
    t.log_outputs({"messages": result["messages"]})
    t.log_reference_outputs({"messages": reference_trajectory})

    trajectory_evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory
    )
```

--------------------------------

### Configure LangChain Agent with SystemMessage for Anthropic Prompt Caching in Python

Source: https://docs.langchain.com/oss/python/langchain/agents

This example shows how to use a `SystemMessage` object for the system prompt, providing more granular control over its structure. It specifically highlights how to utilize Anthropic's prompt caching feature with `cache_control: {"type": "ephemeral"}` for large content blocks, reducing latency and costs when analyzing extensive texts like entire literary works.

```python
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage

literary_agent = create_agent(
    model="anthropic:claude-sonnet-4-5",
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with analyzing literary works.",
            },
            {
                "type": "text",
                "text": "<the entire contents of 'Pride and Prejudice'>",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    )
)

result = literary_agent.invoke(
    {"messages": [HumanMessage("Analyze the major themes in 'Pride and Prejudice'.")]}
)
```

--------------------------------

### Setup Agent Chat UI with npx

Source: https://docs.langchain.com/oss/python/langchain/ui

Initialize a new Agent Chat UI project using npx command-line tool. This creates a new project directory with all necessary dependencies configured. Requires Node.js and pnpm package manager installed.

```bash
# Create a new Agent Chat UI project
npx create-agent-chat-app --project-name my-chat-ui
cd my-chat-ui

# Install dependencies and start
pnpm install
pnpm dev
```

--------------------------------

### Create and Run a Basic AI Agent with LangChain and Claude

Source: https://docs.langchain.com/oss/python/langchain/quickstart

This Python snippet demonstrates how to initialize and invoke a simple AI agent using the LangChain library. It integrates a custom 'get_weather' tool and utilizes Claude Sonnet as the underlying language model, configured with a system prompt to act as a helpful assistant. The agent processes a user message to query for weather information.

```python
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

--------------------------------

### Implement PII Detection Middleware in Langchain Agents

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This example shows how to add `PIIMiddleware` to a Langchain agent to detect and handle Personally Identifiable Information. It configures the middleware to redact 'email' and mask 'credit_card' information, applying these strategies to the agent's input for data privacy compliance.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
    ],
)
```

--------------------------------

### Create Agent with Load Skill Tool - Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills

Demonstrates basic implementation of a skills-based agent using LangChain. Defines a load_skill tool decorator that retrieves specialized skill prompts (e.g., SQL writing, legal document review) and creates an agent with access to these skills. The agent can invoke load_skill to progressively load specialized prompts and context on-demand.

```python
from langchain.tools import tool
from langchain.agents import create_agent

@tool
def load_skill(skill_name: str) -> str:
    """Load a specialized skill prompt.

    Available skills:
    - write_sql: SQL query writing expert
    - review_legal_doc: Legal document reviewer

    Returns the skill's prompt and context.
    """
    # Load skill content from file/database
    ...

agent = create_agent(
    model="gpt-4o",
    tools=[load_skill],
    system_prompt=(
        "You are a helpful assistant. "
        "You have access to two skills: "
        "write_sql and review_legal_doc. "
        "Use load_skill to access them."
    ),
)
```

--------------------------------

### Route to Single Agent with Command

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router

Classifies a query using an LLM and routes it to a single specialized agent using the Command routing method. Returns a Command object that specifies which agent to invoke based on query classification.

```python
from langgraph.types import Command

def classify_query(query: str) -> str:
    """Use LLM to classify query and determine the appropriate agent."""
    # Classification logic here
    ...

def route_query(state: State) -> Command:
    """Route to the appropriate agent based on query classification."""
    active_agent = classify_query(state["query"])

    # Route to the selected agent
    return Command(goto=active_agent)
```

--------------------------------

### Create Sub-Agent and Wrap as Tool in LangChain

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

Demonstrates creating a sub-agent using create_agent() and wrapping it as a tool using the @tool decorator. The wrapped function invokes the sub-agent with a user query and returns the final message content. This approach allows a main agent to delegate tasks to specialized sub-agents when needed.

```python
# Create a sub-agent
subagent = create_agent(model="...", tools=[...])

# Wrap it as a tool
@tool("subagent_name", description="subagent_description")
def call_subagent(query: str):
    result = subagent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content

# Main agent with subagent as a tool
main_agent = create_agent(model="...", tools=[call_subagent])
```

--------------------------------

### Create Multi-Step Agent with Middleware in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Initializes an agent with step-based configuration, summarization middleware, and in-memory state persistence. Combines all tools from all steps and applies middleware for dynamic prompt injection and token-based summarization. Uses checkpointer for conversation state management.

```python
# Collect all tools from all step configurations
all_tools = [
    record_warranty_status,
    record_issue_type,
    provide_solution,
    escalate_to_human,
]

# Create the agent with step-based configuration and summarization
agent = create_agent(
    model,
    tools=all_tools,
    state_schema=SupportState,
    middleware=[
        apply_step_config,
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 10)
        )
    ],
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Create agent with LLMToolEmulator middleware

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Initialize an agent with LLMToolEmulator middleware to emulate tool execution. By default, all tools are emulated. The emulator generates plausible responses using the specified LLM model instead of executing actual tools.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather, search_database, send_email],
    middleware=[
        LLMToolEmulator(),  # Emulate all tools
    ],
)
```

--------------------------------

### Resume Agent Stream After Human Approval (Python)

Source: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

This Python code snippet demonstrates how to resume an agent's execution stream after a human decision, specifically an 'approve' action. It iterates through the streamed updates and messages, printing the content of `token.content` in real-time, which is essential for interactive Human-In-The-Loop (HITL) applications.

```python
for mode, chunk in agent.stream(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
    stream_mode=["updates", "messages"],
):
    if mode == "messages":
        token, metadata = chunk
        if token.content:
            print(token.content, end="", flush=True)
```

--------------------------------

### Create Agent with Summarization Middleware - Python

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Initialize a LangChain agent with SummarizationMiddleware to automatically summarize conversation history. The middleware triggers summarization when token count reaches 4000 and preserves the last 20 messages. Requires a chat model for generating summaries and a list of tools.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)
```

--------------------------------

### Implement Dynamic Model Selection with LangChain Python Middleware

Source: https://docs.langchain.com/oss/python/langchain/agents

This snippet demonstrates how to dynamically select a language model (LLM) for a LangChain agent based on the conversation's complexity. It uses the `@wrap_model_call` decorator to create middleware that switches between a 'basic_model' (gpt-4o-mini) and an 'advanced_model' (gpt-4o) if the message count in the conversation state exceeds 10. The middleware overrides the model in the request before passing it to the handler.

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(model="gpt-4o-mini")
advanced_model = ChatOpenAI(model="gpt-4o")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

--------------------------------

### Install LangGraph CLI with Python

Source: https://docs.langchain.com/oss/python/langchain/studio

Install the LangGraph CLI tool which provides a local development server (Agent Server) to connect your agent to LangSmith Studio. Requires Python 3.11 or higher.

```shell
# Python >= 3.11 is required.
pip install --upgrade "langgraph-cli[inmem]"
```

--------------------------------

### Configure Model Fallback Middleware for Langchain Agents

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This snippet demonstrates how to integrate `ModelFallbackMiddleware` into a Langchain agent. It automatically tries alternative models like 'gpt-4o-mini' or 'claude-3-5-sonnet-20241022' if the primary model ('gpt-4o') fails, enhancing agent resilience and allowing for cost optimization.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ModelFallbackMiddleware(
            "gpt-4o-mini",
            "claude-3-5-sonnet-20241022",
        ),
    ],
)
```

--------------------------------

### Installing LangChain's AgentEvals Package via pip

Source: https://docs.langchain.com/oss/python/langchain/test

This Bash command installs the `agentevals` package, a LangChain component designed for evaluating agent trajectories with live models. This package is essential for setting up integration tests for agentic applications, ensuring components work together correctly.

```bash
pip install agentevals
```

--------------------------------

### Define LangGraph Agent Result Format (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This Python snippet shows the expected dictionary format for results returned by agents in a LangGraph workflow. Each agent's output is wrapped in a 'results' key, containing a list of dictionaries with 'source' and 'result' keys, facilitating collection and reduction.

```python
{"results": [{"source": "github", "result": "..."}]}
```

--------------------------------

### Create and Run Weather Forecasting Agent with LangChain (Python)

Source: https://docs.langchain.com/oss/python/langchain/quickstart

This code assembles all the previously defined components to create and run a fully functional weather forecasting agent. It utilizes `create_agent` with the model, system prompt, tools, context schema, response format, and checkpointer, then invokes the agent with a user message and prints the structured response.

```python
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# `thread_id` is a unique identifier for a given conversation.
config = {"configurable": {"thread_id": "1"}}

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])
```

--------------------------------

### Configure and Invoke LangChain Agent for Structured Responses (Python)

Source: https://docs.langchain.com/oss/python/langchain/quickstart

This code defines a LangChain agent that acts as a pun-speaking weather forecaster. It includes custom tools for getting user location and weather, a system prompt, a context schema for user information, and a `ResponseFormat` dataclass for structured output. The agent is configured to maintain conversation state using `InMemorySaver` and `thread_id`.

```python
from dataclasses import dataclass

  from langchain.agents import create_agent
  from langchain.chat_models import init_chat_model
  from langchain.tools import tool, ToolRuntime
  from langgraph.checkpoint.memory import InMemorySaver
  from langchain.agents.structured_output import ToolStrategy


  # Define system prompt
  SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

  You have access to two tools:

  - get_weather_for_location: use this to get the weather for a specific location
  - get_user_location: use this to get the user's location

  If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""

  # Define context schema
  @dataclass
  class Context:
      """Custom runtime context schema."""
      user_id: str

  # Define tools
  @tool
  def get_weather_for_location(city: str) -> str:
      """Get weather for a given city."""
      return f"It's always sunny in {city}!"

  @tool
  def get_user_location(runtime: ToolRuntime[Context]) -> str:
      """Retrieve user information based on user ID."""
      user_id = runtime.context.user_id
      return "Florida" if user_id == "1" else "SF"

  # Configure model
  model = init_chat_model(
      "claude-sonnet-4-5-20250929",
      temperature=0
  )

  # Define response format
  @dataclass
  class ResponseFormat:
      """Response schema for the agent."""
      # A punny response (always required)
      punny_response: str
      # Any interesting information about the weather if available
      weather_conditions: str | None = None

  # Set up memory
  checkpointer = InMemorySaver()

  # Create agent
  agent = create_agent(
      model=model,
      system_prompt=SYSTEM_PROMPT,
      tools=[get_user_location, get_weather_for_location],
      context_schema=Context,
      response_format=ToolStrategy(ResponseFormat),
      checkpointer=checkpointer
  )

  # Run agent
  # `thread_id` is a unique identifier for a given conversation.
  config = {"configurable": {"thread_id": "1"}}

  response = agent.invoke(
      {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
      config=config,
      context=Context(user_id="1")
  )

  print(response['structured_response'])
  # ResponseFormat(
  #     punny_response="Florida is still having a 'sun-derful' day! The sunshine is playing 'ray-dio' hits all day long! I'd say it's the perfect weather for some 'solar-bration'! If you were hoping for rain, I'm afraid that idea is all 'washed up' - the forecast remains 'clear-ly' brilliant!",
  #     weather_conditions="It's always sunny in Florida!"
  # )


  # Note that we can continue the conversation using the same `thread_id`.
  response = agent.invoke(
      {"messages": [{"role": "user", "content": "thank you!"}]},
      config=config,
      context=Context(user_id="1")
  )

  print(response['structured_response'])
  # ResponseFormat(
  #     punny_response="You're 'thund-erfully' welcome! It's always a 'breeze' to help you stay 'current' with the weather. I'm just 'cloud'-ing around waiting to 'shower' you with more forecasts whenever you need them. Have a 'sun-sational' day in the Florida sunshine!",
  #     weather_conditions=None
  # )
```

--------------------------------

### Run LangChain Agent and Process Human-in-the-Loop Interruption

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

This Python code illustrates how to run a LangChain agent configured with human-in-the-loop middleware and detect when it has paused for review. When an interruption occurs (indicated by '__interrupt__' in the step output), it extracts and prints the details of the pending action request, allowing a human to review the proposed action before it's executed.

```python
question = "Which genre on average has the longest tracks?"
config = {"configurable": {"thread_id": "1"}}

for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    config,
    stream_mode="values",
):
    if "__interrupt__" in step:
        print("INTERRUPTED:")
        interrupt = step["__interrupt__"][0]
        for request in interrupt.value["action_requests"]:
            print(request["description"])
    elif "messages" in step:
        step["messages"][-1].pretty_print()
    else:
        pass
```

--------------------------------

### Aggregate Message Chunks and Detect Tool Calls in LangChain Agent Stream

Source: https://docs.langchain.com/oss/python/langchain/streaming

Demonstrates streaming from a LangChain agent with message aggregation. The code iterates through the agent stream with 'messages' and 'updates' modes, accumulates AIMessageChunk tokens into a full_message object, and prints tool calls when the last chunk position is reached. Requires an initialized agent and helper functions _render_message_chunk and _render_completed_message.

```python
input_message = {"role": "user", "content": "What is the weather in Boston?"}
full_message = None
for stream_mode, data in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],
):
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
            full_message = token if full_message is None else full_message + token
            if token.chunk_position == "last":
                if full_message.tool_calls:
                    print(f"Tool calls: {full_message.tool_calls}")
                full_message = None
    if stream_mode == "updates":
        for source, update in data.items():
            if source == "tools":
                _render_completed_message(update["messages"][-1])
```

--------------------------------

### Create and Run LangChain Agent with Automatic Tracing

Source: https://docs.langchain.com/oss/python/langchain/observability

Create a LangChain agent with multiple tools (send_email, search_web) and invoke it with automatic LangSmith tracing enabled. No additional code is required; traces are logged automatically when environment variables are set.

```python
from langchain.agents import create_agent


def send_email(to: str, subject: str, body: str):
    """Send an email to a recipient."""
    # ... email sending logic
    return f"Email sent to {to}"

def search_web(query: str):
    """Search the web for information."""
    # ... web search logic
    return f"Search results for: {query}"

agent = create_agent(
    model="gpt-4o",
    tools=[send_email, search_web],
    system_prompt="You are a helpful assistant that can send emails and search the web."
)

# Run the agent - all steps will be traced automatically
response = agent.invoke({
    "messages": [{"role": "user", "content": "Search for the latest AI news and email a summary to john@example.com"}]
})
```

--------------------------------

### Implement LangChain Multi-Agent Personal Assistant with Sub-Agents (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

This comprehensive Python script demonstrates a multi-agent system using LangChain. It defines API tools for calendar and email operations, creates specialized sub-agents for each function, wraps these sub-agents as tools, and finally orchestrates them with a supervisor agent. The example showcases how a supervisor agent can coordinate specialized sub-agents (calendar and email) that are themselves wrapped as tools, illustrating a robust tool-calling pattern for multi-agent systems.

```python
"""
Personal Assistant Supervisor Example

This example demonstrates the tool calling pattern for multi-agent systems.
A supervisor agent coordinates specialized sub-agents (calendar and email)
that are wrapped as tools.
"""

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# ============================================================================
# Step 1: Define low-level API tools (stubbed)
# ============================================================================

@tool
def create_calendar_event(
    title: str,
    start_time: str,  # ISO format: "2024-01-15T14:00:00"
    end_time: str,    # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def send_email(
    to: list[str],      # email addresses
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    return ["09:00", "14:00", "16:00"]


# ============================================================================
# Step 2: Create specialized sub-agents
# ============================================================================

model = init_chat_model("claude-haiku-4-5-20251001")  # for example

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=(
        "You are a calendar scheduling assistant. "
        "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') "
        "into proper ISO datetime formats. "
        "Use get_available_time_slots to check availability when needed. "
        "Use create_calendar_event to schedule events. "
        "Always confirm what was scheduled in your final response."
    )
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=(
        "You are an email assistant. "
        "Compose professional emails based on natural language requests. "
        "Extract recipient information and craft appropriate subject lines and body text. "
        "Use send_email to send the message. "
        "Always confirm what was sent in your final response."
    )
)

# ============================================================================
# Step 3: Wrap sub-agents as tools for the supervisor
# ============================================================================

@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


# ============================================================================
# Step 4: Create the supervisor agent
# ============================================================================

supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=(
        "You are a helpful personal assistant. "
        "You can schedule calendar events and send emails. "
        "Break down user requests into appropriate tool calls and coordinate the results. "
        "When a request involves multiple actions, use multiple tools in sequence."
    )
)
```

--------------------------------

### Configure Agent with After Model Middleware

Source: https://docs.langchain.com/oss/python/langchain/runtime

Pass the after_model hook function to the create_agent middleware parameter alongside other middleware. The agent will execute this hook after model inference completes, allowing access to context data passed during agent invocation.

```python
agent = create_agent(
    model="gpt-5-nano",
    tools=[...],
    middleware=[dynamic_system_prompt, log_before_model, log_after_model],
    context_schema=Context
)
```

--------------------------------

### Create Agent with Human-in-the-Loop Middleware for Sensitive Operations

Source: https://docs.langchain.com/oss/python/langchain/guardrails

Set up a LangChain agent with human-in-the-loop middleware that interrupts execution for sensitive operations (email sending, database deletion) while auto-approving safe operations (search). The agent uses an in-memory checkpointer to persist state across interrupts and requires a thread ID for resuming paused conversations.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # Require approval for sensitive operations
                "send_email": True,
                "delete_database": True,
                # Auto-approve safe operations
                "search": False,
            }
        ),
    ],
    # Persist the state across interrupts
    checkpointer=InMemorySaver(),
)

# Human-in-the-loop requires a thread ID for persistence
config = {"configurable": {"thread_id": "some_id"}}

# Agent will pause and wait for approval before executing sensitive tools
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to the team"}]},
    config=config
)

result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config  # Same thread ID to resume the paused conversation
)
```

--------------------------------

### Configure HumanInTheLoopMiddleware for Agent Interrupts

Source: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

Sets up an agent with HumanInTheLoopMiddleware to interrupt execution on specific tool calls. The middleware maps tool names to approval configurations (True for all decisions, False for auto-approve, or custom InterruptOnConfig objects) and requires a checkpointer to persist state across interrupts. Dependencies include LangChain agents, middleware, and LangGraph checkpointing.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


agent = create_agent(
    model="gpt-4o",
    tools=[write_file_tool, execute_sql_tool, read_data_tool],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "write_file": True,
                "execute_sql": {"allowed_decisions": ["approve", "reject"]},
                "read_data": False,
            },
            description_prefix="Tool execution pending approval",
        ),
    ],
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Python: Assemble Voice Agent Pipeline with RunnableGenerator

Source: https://docs.langchain.com/oss/python/langchain/voice-agent

This code demonstrates how to construct a complete voice agent pipeline by chaining three asynchronous stages: Speech-to-Text (STT), an agent processing, and Text-to-Speech (TTS). It uses `RunnableGenerator` from `langchain_core` to wrap each asynchronous generator, creating a seamless flow from audio input to generated speech output.

```python
from langchain_core.runnables import RunnableGenerator

pipeline = (
    RunnableGenerator(stt_stream)      # Audio → STT events
    | RunnableGenerator(agent_stream)  # STT events → Agent events
    | RunnableGenerator(tts_stream)    # Agent events → TTS audio
)
```

--------------------------------

### Invoke LangChain Agent to Generate SQL Query (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

This Python snippet initializes a unique thread ID and then invokes a LangChain agent with a user message requesting a SQL query. The agent's response, which includes the generated SQL and conversation messages, is stored in the `result` variable, and its messages are then printed.

```python
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

result = agent.invoke(  # [!code highlight]
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write a SQL query to find all customers "
                    "who made orders over $1000 in the last month"
                ),
            }
        ]
    },
    config
)

for message in result["messages"]:
    if hasattr(message, 'pretty_print'):
        message.pretty_print()
    else:
        print(f"{message.type}: {message.content}")
```

--------------------------------

### Handle Multiple Structured Outputs Error with Union Types

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Creates a LangChain agent that handles cases where a model incorrectly calls multiple structured output tools. When the model returns multiple tool calls (ContactInfo and EventDetails), the agent provides error feedback in a ToolMessage and prompts for retry. The ToolStrategy with Union types and default handle_errors=True manages this scenario.

```python
from pydantic import BaseModel, Field
from typing import Union
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy


class ContactInfo(BaseModel):
    name: str = Field(description="Person's name")
    email: str = Field(description="Email address")

class EventDetails(BaseModel):
    event_name: str = Field(description="Name of the event")
    date: str = Field(description="Event date")

agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(Union[ContactInfo, EventDetails])  # Default: handle_errors=True
)

agent.invoke({
    "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})
```

--------------------------------

### LLM-as-Judge Trajectory Evaluator With Reference in Python

Source: https://docs.langchain.com/oss/python/langchain/test

Configures an LLM-based trajectory evaluator that uses both actual agent outputs and a reference trajectory for comparison. The evaluator uses a prebuilt prompt template that accepts reference trajectory data to provide more informed judgments about agent performance.

```python
evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
)
evaluation = judge_with_reference(
    outputs=result["messages"],
    reference_outputs=reference_trajectory,
)
```

--------------------------------

### Create an Agent with LangChain Python

Source: https://docs.langchain.com/oss/python/langchain/overview

Demonstrates how to create and invoke a LangChain agent using the create_agent function with Claude model, custom tools, and system prompts. This example shows basic agent setup with a weather tool and user message invocation in under 10 lines of code.

```python
# pip install -qU langchain "langchain[anthropic]"
from langchain.agents import create_agent

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
```

--------------------------------

### Asynchronously Process Voice Events with LangChain Agent

Source: https://docs.langchain.com/oss/python/langchain/voice-agent

This asynchronous function processes an `event_stream` containing voice events, passing them through while also invoking the LangChain agent for final STT transcripts. It generates a unique thread ID for conversation memory and streams agent responses as `AgentChunkEvent`s, maintaining conversational context.

```python
async def agent_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Agent Responses)

    Passes through all upstream events and adds agent_chunk events
    when processing STT transcripts.
    """
    # Generate unique thread ID for conversation memory
    thread_id = str(uuid4())

    async for event in event_stream:
        # Pass through all upstream events
        yield event

        # Process final transcripts through the agent
        if event.type == "stt_output":
            # Stream agent response with conversation context
            stream = agent.astream(
                {"messages": [HumanMessage(content=event.transcript)]},
                {"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            )

            # Yield agent response chunks as they arrive
            async for message, _ in stream:
                if message.text:
                    yield AgentChunkEvent.create(message.text)
```

--------------------------------

### Enable LangSmith Tracing with Environment Variables

Source: https://docs.langchain.com/oss/python/langchain/observability

Set environment variables to enable automatic tracing of LangChain agents through LangSmith. These variables activate tracing globally for all agent invocations and authenticate requests using your API key.

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=<your-api-key>
```

--------------------------------

### Define Agent Structured Response Format in Python

Source: https://docs.langchain.com/oss/python/langchain/structured-output

This Python code snippet outlines the `response_format` parameter within LangChain's `create_agent` function. It demonstrates how to specify the desired structured output format using various strategies, including `ToolStrategy`, `ProviderStrategy`, or directly providing a schema type.

```python
def create_agent(
    ...
    response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
    ]
```

--------------------------------

### Customize Agent State with Additional Fields

Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

Extend AgentState to include custom fields beyond conversation history, such as user_id and preferences. Custom state schemas are passed to create_agent via the state_schema parameter and can be populated during agent invocation. Enables richer context management for complex agent behaviors.

```python
from langchain.agents import create_agent, AgentState
from langgraph.checkpoint.memory import InMemorySaver

class CustomAgentState(AgentState):
    user_id: str
    preferences: dict

agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    state_schema=CustomAgentState,
    checkpointer=InMemorySaver(),
)

result = agent.invoke(
    {
        "messages": [{"role": "user", "content": "Hello"}],
        "user_id": "user_123",
        "preferences": {"theme": "dark"}
    },
    {"configurable": {"thread_id": "1"}}
)
```

--------------------------------

### Superset and Subset Trajectory Match Evaluators in Python

Source: https://docs.langchain.com/oss/python/langchain/test

Implements superset and subset trajectory match modes to verify agent behavior against partial reference trajectories. Superset mode ensures the agent calls at least the required tools (allowing extras), while subset mode ensures no extra tool calls beyond the reference.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator


@tool
def get_weather(city: str):
    """Get weather information for a city."""
    return f"It's 75 degrees and sunny in {city}."

@tool
def get_detailed_forecast(city: str):
    """Get detailed weather forecast for a city."""
    return f"Detailed forecast for {city}: sunny all week."

agent = create_agent("gpt-4o", tools=[get_weather, get_detailed_forecast])

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="superset",
)

def test_agent_calls_required_tools_plus_extra():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in Boston?")]
    })

    reference_trajectory = [
        HumanMessage(content="What's the weather in Boston?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "Boston"}},
        ]),
        ToolMessage(content="It's 75 degrees and sunny in Boston.", tool_call_id="call_1"),
        AIMessage(content="The weather in Boston is 75 degrees and sunny."),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory,
    )
    assert evaluation["score"] is True
```

--------------------------------

### Integrate Model Retry Middleware in LangChain Agent (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Demonstrates how to integrate `ModelRetryMiddleware` into a LangChain agent. This setup automatically retries failed model calls with configurable exponential backoff parameters such as maximum retries, backoff factor, and initial delay, enhancing the agent's resilience to transient API failures.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)
```

--------------------------------

### Build StateGraph with Conditional Edges

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

Constructs a LangChain StateGraph with multiple agent nodes and conditional routing edges. Configures initial routing from START, then sets up conditional edges between agents and an END node based on agent decision outcomes.

```python
# 6. Build the graph
builder = StateGraph(MultiAgentState)
builder.add_node("sales_agent", call_sales_agent)
builder.add_node("support_agent", call_support_agent)

# Start with conditional routing based on initial active_agent
builder.add_conditional_edges(START, route_initial, ["sales_agent", "support_agent"])

# After each agent, check if we should end or route to another agent
builder.add_conditional_edges(
    "sales_agent", route_after_agent, ["sales_agent", "support_agent", END]
)
builder.add_conditional_edges(
    "support_agent", route_after_agent, ["sales_agent", "support_agent", END]
)

graph = builder.compile()
```

--------------------------------

### Reject Tool Call Decision - LangGraph Python

Source: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

Reject a tool call and provide feedback to the agent instead of execution. The message field contains an explanation about why the action was rejected and guidance on what the agent should do instead. This feedback is added to the conversation to help the agent adjust its approach.

```python
agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "reject",
                    "message": "No, this is wrong because ..., instead do this ...",
                }
            ]
        }
    ),
    config=config
)
```

--------------------------------

### Build custom skill middleware for agent system prompt injection

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Creates a custom AgentMiddleware class that injects skill descriptions into the agent's system prompt during model calls. The middleware registers the load_skill tool as a class variable and builds a formatted skills list in __init__. It overrides wrap_model_call to append skill descriptions to the system message, making skills discoverable without loading full content upfront.

```python
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from typing import Callable

class SkillMiddleware(AgentMiddleware):
    """Middleware that injects skill descriptions into the system prompt."""

    # Register the load_skill tool as a class variable
    tools = [load_skill]

    def __init__(self):
        """Initialize and generate the skills prompt from SKILLS."""
        # Build skills prompt from the SKILLS list
        skills_list = []
        for skill in SKILLS:
            skills_list.append(
                f"- **{skill['name']}**: {skill['description']}"
            )
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync: Inject skill descriptions into system prompt."""
        # Build the skills addendum
        skills_addendum = (
            f"\n\n## Available Skills\n\n{self.skills_prompt}\n\n"
            "Use the load_skill tool when you need detailed information "
            "about handling a specific type of request."
        )

        # Append to system message content blocks
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)
```

--------------------------------

### Wrap Subagents as Tools - Python LangChain

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

Create tool wrappers for subagents using LangChain's tool decorator and agent creation utilities. This demonstrates the tool-per-agent pattern where each subagent is exposed as a callable tool with custom input/output handling. Requires LangChain installation and agent framework setup.

```python
from langchain.tools import tool
from langchain.agents import create_agent
```

--------------------------------

### Define Tools for Weather Agent with LangChain (Python)

Source: https://docs.langchain.com/oss/python/langchain/quickstart

This code defines two tools for the agent: `get_weather_for_location` to fetch weather for a city and `get_user_location` to retrieve the user's current location. It demonstrates using the `@tool` decorator from LangChain, a custom `Context` dataclass, and `ToolRuntime` to access runtime context for `get_user_location`.

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"
```

--------------------------------

### Invoke Langchain Agent with User Context (Python)

Source: https://docs.langchain.com/oss/python/langchain/long-term-memory

Invokes a Langchain agent by providing user messages and a 'user_id' within the 'Context' object. This allows the agent to process requests while associating them with a specific user, which is crucial for managing user-specific states or interactions.

```python
agent.invoke(
    {"messages": [{"role": "user", "content": "My name is John Smith"}]},
    # user_id passed in context to identify whose information is being updated
    context=Context(user_id="user_123")
)
```

--------------------------------

### Route to Agents with LangChain Send in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This Python function dynamically routes a query to specific agents based on prior classifications. It processes a 'RouterState' (a dictionary-like object holding classification results) and generates a list of 'Send' objects, each directing the relevant query to its designated 'source' agent.

```python
  def route_to_agents(state: RouterState) -> list[Send]:
      """Fan out to agents based on classifications."""
      return [
          Send(c["source"], {"query": c["query"]})
          for c in state["classifications"]
      ]
```

--------------------------------

### Configure Model Call Limit Middleware in LangChain Agent

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Creates a LangChain agent with ModelCallLimitMiddleware to enforce call limits across threads and individual runs. Requires an InMemorySaver checkpointer for thread limiting functionality. Supports graceful termination or error-raising behaviors when limits are exceeded.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    checkpointer=InMemorySaver(),  # Required for thread limiting
    tools=[],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
    ],
)
```

--------------------------------

### Map Agent Steps to Configurations in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

This Python dictionary, `STEP_CONFIG`, maps each agent workflow step to its corresponding prompt, required tools, and state dependencies. It centralizes the configuration for a multi-step LangChain agent, allowing for easy management and extension of the workflow by defining the prompt, the tools available at that step, and the state variables that must be present before entering the step.

```python
# Step configuration: maps step name to (prompt, tools, required_state)
STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
        "prompt": RESOLUTION_SPECIALIST_PROMPT,
        "tools": [provide_solution, escalate_to_human],
        "requires": ["warranty_status", "issue_type"],
    },
}
```

--------------------------------

### Transfer to Sales Agent with Tool Response Pairing

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

Tool function that handles handoff to the sales agent using Command.PARENT. Creates an artificial ToolMessage to pair with the AIMessage containing the tool call, maintaining valid conversation history for the receiving agent. Passes minimal context to avoid confusion and token bloat.

```python
@tool
def transfer_to_sales(runtime: ToolRuntime) -> Command:
    # Get the AI message that triggered this handoff
    last_ai_message = runtime.state["messages"][-1]

    # Create an artificial tool response to complete the pair
    transfer_message = ToolMessage(
        content="Transferred to sales agent",
        tool_call_id=runtime.tool_call_id,
    )

    return Command(
        goto="sales_agent",
        update={
            "active_agent": "sales_agent",
            # Pass only these two messages, not the full subagent history
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )
```

--------------------------------

### Define Agent Router Workflow with LangChain StateGraph in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This code defines a complex agentic workflow using LangChain's 'StateGraph'. It adds nodes for query classification, individual agent queries (GitHub, Notion, Slack), and result synthesis. Conditional edges are set up for routing based on classification, and linear edges connect the query agents to the synthesis step, compiling a complete, executable graph.

```python
  # Build workflow
  workflow = (
      StateGraph(RouterState)
      .add_node("classify", classify_query)
      .add_node("github", query_github)
      .add_node("notion", query_notion)
      .add_node("slack", query_slack)
      .add_node("synthesize", synthesize_results)
      .add_edge(START, "classify")
      .add_conditional_edges("classify", route_to_agents, ["github", "notion", "slack"])
      .add_edge("github", "synthesize")
      .add_edge("notion", "synthesize")
      .add_edge("slack", "synthesize")
      .add_edge("synthesize", END)
      .compile()
  )
```

--------------------------------

### Create Agent with PostgreSQL Checkpointer for Production

Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

Set up an agent with PostgreSQL-backed checkpointer for production environments. The checkpointer automatically creates necessary database tables and persists conversation state across sessions. Requires langgraph-checkpoint-postgres package installation.

```shell
pip install langgraph-checkpoint-postgres
```

```python
from langchain.agents import create_agent
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup()
    agent = create_agent(
        "gpt-5",
        tools=[get_user_info],
        checkpointer=checkpointer,
    )
```

--------------------------------

### Configure SummarizationMiddleware with Fractional Context Limits

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Create an agent with SummarizationMiddleware using fractional context limits. Triggers summarization when context reaches 80% of model's capacity and keeps 30% of context after summarization.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent3 = create_agent(
    model="gpt-4o",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("fraction", 0.8),
            keep=("fraction", 0.3),
        ),
    ],
)
```

--------------------------------

### Create LangChain Agent with Tools and In-Memory Saver

Source: https://docs.langchain.com/oss/python/langchain/voice-agent

This snippet demonstrates how to initialize a LangChain agent using a specified AI model, a list of tools for specific actions (e.g., `add_to_order`, `confirm_order`), and a system prompt to define its persona and interaction style. It uses `InMemorySaver` for conversation memory, ensuring context is maintained across turns.

```python
agent = create_agent(
    model="anthropic:claude-haiku-4-5",  # Select your model
    tools=[add_to_order, confirm_order],
    system_prompt="""You are a helpful sandwich shop assistant.
    Your goal is to take the user's order. Be concise and friendly.
    Do NOT use emojis, special characters, or markdown.
    Your responses will be read by a text-to-speech engine.""",
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Unordered Trajectory Match Evaluator in Python

Source: https://docs.langchain.com/oss/python/langchain/test

Creates and uses a trajectory match evaluator in unordered mode to verify agent tool calls match a reference trajectory regardless of call order. The evaluator compares the agent's actual execution path against a reference trajectory, returning a boolean score indicating if trajectories match.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator


@tool
def get_weather(city: str):
    """Get weather information for a city."""
    return f"It's 75 degrees and sunny in {city}."

agent = create_agent("gpt-4o", tools=[get_weather])

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="unordered",
)

def test_agent_trajectory_match():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in NYC?")]
    })

    reference_trajectory = [
        HumanMessage(content="What's the weather in NYC?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "NYC"}},
        ]),
        ToolMessage(content="It's 75 degrees and sunny in NYC.", tool_call_id="call_1"),
        AIMessage(content="The weather in NYC is 75 degrees and sunny."),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory,
    )
    assert evaluation["score"] is True
```

--------------------------------

### Initialize ShellToolMiddleware with HostExecutionPolicy

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Creates an agent with ShellToolMiddleware configured for native host execution. The middleware establishes a persistent shell session in the specified workspace directory, allowing agents to execute sequential commands with full host access. Best suited for trusted environments where the agent runs inside a container or VM.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ShellToolMiddleware,
    HostExecutionPolicy,
)

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)
```

--------------------------------

### Configure Agent Middleware for Context-Based Response Formatting (Python)

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

This Python snippet demonstrates how to apply middleware to a LangChain agent to dynamically override the response format based on the request context, such as user roles. It shows checking if a user is an admin and applying a specific `AdminResponse` format, otherwise using a `UserResponse`.

```python
            # Admins in production get detailed output
            request = request.override(response_format=AdminResponse)
        else:
            # Regular users get simple output
            request = request.override(response_format=UserResponse)

        return handler(request)

    agent = create_agent(
        model="gpt-4o",
        tools=[...],
        middleware=[context_based_output],
        context_schema=Context
    )
```

--------------------------------

### Synthesize Results from Multiple Agents (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

The `synthesize_results` function aggregates and processes the responses received from all invoked agents. If no results are found, it returns a default message. Otherwise, it formats the individual agent results and uses a Language Model (LLM) with a specific system prompt to synthesize them into a coherent and comprehensive answer to the original user query, combining information and highlighting key points.

```python
def synthesize_results(state: RouterState) -> dict:
    """Combine results from all agents into a coherent answer."""
    if not state["results"]:
        return {"final_answer": "No results found from any knowledge source."}

    # Format results for synthesis
    formatted = [
        f"**From {r['source'].title()}:**\n{r['result']}"
        for r in state["results"]
    ]

    synthesis_response = router_llm.invoke([
        {
            "role": "system",
            "content": f"""Synthesize these search results to answer the original question: "{state['query']}"\n\n- Combine information from multiple sources without redundancy\n- Highlight the most relevant and actionable information\n- Note any discrepancies between sources\n- Keep the response concise and well-organized"""
        },
        {"role": "user", "content": "\n\n".join(formatted)}
    ])

    return {"final_answer": synthesis_response.content}
```

--------------------------------

### Generate Dynamic LangChain Agent Prompts with Middleware and View Output in Python

Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

This code illustrates how to create dynamic prompts for a LangChain agent using middleware. It defines a 'dynamic_system_prompt' function that accesses the agent's 'CustomContext' to personalize the system prompt with the 'user_name', allowing for context-aware interactions. The accompanying shell output demonstrates the agent's response, including tool calls and the final personalized message.

```python
from langchain.agents import create_agent
from typing import TypedDict
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class CustomContext(TypedDict):
    user_name: str


def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is always sunny!"


@dynamic_prompt
def dynamic_system_prompt(request: ModelRequest) -> str:
    user_name = request.runtime.context["user_name"]
    system_prompt = f"You are a helpful assistant. Address the user as {user_name}."
    return system_prompt


agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
    middleware=[dynamic_system_prompt],
    context_schema=CustomContext,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    context=CustomContext(user_name="John Smith"),
)
for msg in result["messages"]:
    msg.pretty_print()
```

```shell
================================ Human Message =================================

What is the weather in SF?
================================== Ai Message ==================================
Tool Calls:
  get_weather (call_WFQlOGn4b2yoJrv7cih342FG)
 Call ID: call_WFQlOGn4b2yoJrv7cih342FG
  Args:
    city: San Francisco
================================= Tool Message =================================
Name: get_weather

The weather in San Francisco is always sunny!
================================== Ai Message ==================================

Hi John Smith, the weather in San Francisco is always sunny!
```

--------------------------------

### Resume Agent Execution with Command and Decisions

Source: https://docs.langchain.com/oss/python/langchain/streaming

Demonstrates resuming an agent's execution by passing a Command object containing previously made decisions into the streaming loop. The agent processes the resume command and continues execution, streaming both messages and updates while handling any new interrupts that may occur during the resumed execution.

```python
interrupts = []
for stream_mode, data in agent.stream(
    Command(resume=decisions),
    config=config,
    stream_mode=["messages", "updates"],
):
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
            if source == "__interrupt__":
                interrupts.extend(update)
                _render_interrupt(update[0])
```

--------------------------------

### Test LangChain Agent API with Python and cURL

Source: https://docs.langchain.com/oss/python/langchain/deploy

These examples demonstrate how to send messages and receive streaming updates from a deployed LangChain agent's API. It includes a Python SDK example for asynchronous interaction and a cURL command for direct REST API communication.

```python
from langgraph_sdk import get_sync_client # or get_client for async

client = get_sync_client(url="your-deployment-url", api_key="your-langsmith-api-key")

for chunk in client.runs.stream(
    None,    # Threadless run
    "agent", # Name of agent. Defined in langgraph.json.
    input={
        "messages": [{
            "role": "human",
            "content": "What is LangGraph?"
        }]
    },
    stream_mode="updates",
):
    print(f"Receiving new event of type: {chunk.event}...")
    print(chunk.data)
    print("\n\n")
```

```bash
curl -s --request POST \
    --url <DEPLOYMENT_URL>/runs/stream \
    --header 'Content-Type: application/json' \
    --header "X-Api-Key: <LANGSMITH API KEY>" \
    --data \"{\\\"assistant_id\\\": \\\"agent\\\", \\\"input\\\": {\\\"messages\\\": [{\\\"role\\\": \\\"human\\\", \\\"content\\\": \\\"What is LangGraph?\\\"}]}, \\\"stream_mode\\\": \\\"updates\\\"}\"
```

--------------------------------

### Configure SummarizationMiddleware with Multiple Trigger Conditions

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Create an agent with SummarizationMiddleware that triggers summarization when either tokens reach 3000 OR message count reaches 6. The middleware preserves the last 20 messages after summarization using OR logic for multiple conditions.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent2 = create_agent(
    model="gpt-4o",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=[
                ("tokens", 3000),
                ("messages", 6),
            ],
            keep=("messages", 20),
        ),
    ],
)
```

--------------------------------

### Read long-term memory in LangChain agent tools

Source: https://docs.langchain.com/oss/python/langchain/long-term-memory

Implements a tool that retrieves user information from the long-term memory store. Uses ToolRuntime to access the store within an agent tool, demonstrating the complete flow from tool definition to agent invocation with context passing.

```python
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore


@dataclass
class Context:
    user_id: str

# InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production.
store = InMemoryStore()

# Write sample data to the store using the put method
store.put(
    ("users",),  # Namespace to group related data together (users namespace for user data)
    "user_123",  # Key within the namespace (user ID as key)
    {
        "name": "John Smith",
        "language": "English",
    }  # Data to store for the given user
)

@tool
def get_user_info(runtime: ToolRuntime[Context]) -> str:
    """Look up user info."""
    # Access the store - same as that provided to `create_agent`
    store = runtime.store
    user_id = runtime.context.user_id
    # Retrieve data from store - returns StoreValue object with value and metadata
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_user_info],
    # Pass store to agent - enables agent to access store when running tools
    store=store,
    context_schema=Context
)

# Run the agent
agent.invoke(
    {"messages": [{"role": "user", "content": "look up user information"}]},
    context=Context(user_id="user_123")
)
```

--------------------------------

### Configure HumanInTheLoopMiddleware for Tool Approval Workflow

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Create an agent with HumanInTheLoopMiddleware that requires human approval for send_email_tool operations with approve/edit/reject decisions, while read_email_tool executes without interruption. Requires InMemorySaver checkpointer to maintain state across interruptions.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model="gpt-4o",
    tools=[your_read_email_tool, your_send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "your_send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "your_read_email_tool": False,
            }
        ),
    ],
)
```

--------------------------------

### ModelCallLimitMiddleware Configuration

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Configure and instantiate the ModelCallLimitMiddleware for an agent. This middleware prevents runaway agents, enforces cost controls, and enables testing within specific call budgets.

```APIDOC
## ModelCallLimitMiddleware

### Description
Middleware for LangChain agents that limits the number of model calls to prevent infinite loops, enforce cost controls, and test agent behavior within specific call budgets.

### Class
ModelCallLimitMiddleware

### Module
langchain.agents.middleware

### Constructor Parameters
- **thread_limit** (number) - Optional - Maximum model calls across all runs in a thread. Defaults to no limit.
- **run_limit** (number) - Optional - Maximum model calls per single invocation. Defaults to no limit.
- **exit_behavior** (string) - Optional - Behavior when limit is reached. Options: `'end'` (graceful termination) or `'error'` (raise exception). Defaults to `'end'`.

### Usage Example
```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model="gpt-4o",
    checkpointer=InMemorySaver(),  # Required for thread limiting
    tools=[],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
    ],
)
```

### Configuration Options

#### thread_limit
- **Type**: number
- **Required**: No
- **Description**: Maximum model calls across all runs in a thread. Defaults to no limit.
- **Use Case**: Enforce total call limits across multiple agent invocations in the same thread.

#### run_limit
- **Type**: number
- **Required**: No
- **Description**: Maximum model calls per single invocation. Defaults to no limit.
- **Use Case**: Prevent excessive calls within a single agent execution.

#### exit_behavior
- **Type**: string
- **Required**: No
- **Default**: `'end'`
- **Options**: `'end'`, `'error'`
- **Description**: Behavior when limit is reached. `'end'` performs graceful termination; `'error'` raises an exception.
- **Use Case**: Choose between stopping gracefully or failing explicitly when limits are exceeded.

### Requirements
- A checkpointer (e.g., InMemorySaver) is required for thread-level limiting functionality.

### Use Cases
- Preventing runaway agents from making too many API calls
- Enforcing cost controls on production deployments
- Testing agent behavior within specific call budgets
```

--------------------------------

### Install LangGraph Python SDK

Source: https://docs.langchain.com/oss/python/langchain/deploy

This command installs the `langgraph-sdk` package for Python, which is necessary to programmatically interact with deployed LangChain agents from a Python environment.

```shell
pip install langgraph-sdk
```

--------------------------------

### Configure Agent Behavior with Step Configuration in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Decorator function that applies step-specific configuration to an agent request, including system prompts, tools, and state validation. Uses a configuration dictionary to manage different workflow steps and their requirements. Validates required state exists before proceeding to the step.

```python
@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Configure agent behavior based on the current step."""
    # Get current step (defaults to warranty_collector for first interaction)
    current_step = request.state.get("current_step", "warranty_collector")

    # Look up step configuration
    step_config = STEP_CONFIG[current_step]

    # Validate required state exists
    for key in step_config["requires"]:
        if request.state.get(key) is None:
            raise ValueError(f"{key} must be set before reaching {current_step}")

    # Format prompt with state values
    system_prompt = step_config["prompt"].format(**request.state)

    # Inject system prompt and step-specific tools
    request = request.override(
        system_prompt=system_prompt,
        tools=step_config["tools"],
    )

    return handler(request)
```

--------------------------------

### Handle PII in Agent Messages Using Detection Strategy

Source: https://docs.langchain.com/oss/python/langchain/guardrails

Invoke an agent with user-provided personally identifiable information (PII) such as email addresses and credit card numbers. The middleware will automatically detect and handle the PII according to the configured strategy (block, redact, mask, or hash). The example demonstrates a basic invocation with email and credit card data.

```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "My email is john.doe@example.com and card is 5105-1051-0510-5100"}]
})
```

--------------------------------

### Configure Agent with Human-in-the-Loop Middleware and Streaming

Source: https://docs.langchain.com/oss/python/langchain/streaming

Sets up a LangChain agent with human-in-the-loop middleware to interrupt on specific tool calls (get_weather), uses an in-memory checkpointer for state persistence, and streams messages and updates to handle interrupts. The agent processes weather queries for multiple cities and pauses execution to request human approval before executing tools.

```python
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


checkpointer = InMemorySaver()

agent = create_agent(
    "openai:gpt-5.2",
    tools=[get_weather],
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={"get_weather": True}),
    ],
    checkpointer=checkpointer,
)


def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")


def _render_interrupt(interrupt: Interrupt) -> None:
    interrupts = interrupt.value
    for request in interrupts["action_requests"]:
        print(request["description"])


input_message = {
    "role": "user",
    "content": (
        "Can you look up the weather in Boston and San Francisco?"
    ),
}
config = {"configurable": {"thread_id": "some_id"}}
interrupts = []
for stream_mode, data in agent.stream(
    {"messages": [input_message]},
    config=config,
    stream_mode=["messages", "updates"],
):
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
            if source == "__interrupt__":
                interrupts.extend(update)
                _render_interrupt(update[0])
```

--------------------------------

### Invoke Agent and Handle HITL Interrupts

Source: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

Invokes the agent with a thread ID configuration to enable conversation persistence across interrupts. When a tool call matches the interrupt policy, the result includes an `__interrupt__` field containing action requests and review configurations. The interrupt structure provides tool names, arguments, descriptions, and allowed decisions for human review before resuming execution.

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "some_id"}}
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Delete old records from the database",
            }
        ]
    },
    config=config
)

print(result['__interrupt__'])
# > [
# >    Interrupt(
# >       value={
# >          'action_requests': [
# >             {
# >                'name': 'execute_sql',
# >                'arguments': {'query': 'DELETE FROM records WHERE created_at < NOW() - INTERVAL \'30 days\';'},
# >                'description': 'Tool execution pending approval\n\nTool: execute_sql\nArgs: {...}'
# >             }
# >          ],
# >          'review_configs': [
# >             {
# >                'action_name': 'execute_sql',
# >                'allowed_decisions': ['approve', 'reject']
# >             }
# >          ]
# >       }
# >    )
# > ]

```

--------------------------------

### Stream Agent Output with Updates and Custom Modes in Python

Source: https://docs.langchain.com/oss/python/langchain/streaming

This code snippet demonstrates how to create an agent that streams both standard 'updates' and custom messages from a tool. It defines a `get_weather` tool that uses `get_stream_writer()` to send custom progress updates, and then calls the agent's `stream()` method with `stream_mode=["updates", "custom"]` to receive these differentiated outputs.

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)

for stream_mode, chunk in agent.stream(  # [!code highlight]
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode=["updates", "custom"]
):
    print(f"stream_mode: {stream_mode}")
    print(f"content: {chunk}")
    print("\n")
```

```shell
stream_mode: updates
content: {'model': {'messages': [AIMessage(content='', response_metadata={'token_usage': {'completion_tokens': 280, 'prompt_tokens': 132, 'total_tokens': 412, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 256, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-5-nano-2025-08-07', 'system_fingerprint': None, 'id': 'lc_run--480c07cb-e405-4411-aa7f-0520fddeed66-0', tool_calls=[{'name': 'get_weather', 'args': {'city': 'San Francisco'}, 'id': 'call_KTNQIftMrl9vgNwEfAJMVu7r', 'type': 'tool_call'}], usage_metadata={'input_tokens': 132, 'output_tokens': 280, 'total_tokens': 412, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 256}})]}}


stream_mode: custom
content: Looking up data for city: San Francisco


stream_mode: custom
content: Acquired data for city: San Francisco


stream_mode: updates
content: {'tools': {'messages': [ToolMessage(content="It's always sunny in San Francisco!", name='get_weather', tool_call_id='call_KTNQIftMrl9vgNwEfAJMVu7r')]}}


stream_mode: updates
content: {'model': {'messages': [AIMessage(content='San Francisco weather: It\'s always sunny in San Francisco!\n\n', response_metadata={'token_usage': {'completion_tokens': 764, 'prompt_tokens': 168, 'total_tokens': 932, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 704, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_provider': 'openai', 'model_name': 'gpt-5-nano-2025-08-07', 'system_fingerprint': None, 'id': 'chatcmpl-C9tljDFVki1e1haCyikBptAuXuHYG', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, id='lc_run--acbc740a-18fe-4a14-8619-da92a0d0ee90-0', usage_metadata={'input_tokens': 168, 'output_tokens': 764, 'total_tokens': 932, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 704}})]}}
```

--------------------------------

### Control Return Values from Sub-Agent Tool Calls

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Demonstrates two approaches for customizing what information flows back from tool calls to the supervisor: returning plain text responses or structured JSON data. Emphasizes the importance of including all relevant information in the sub-agent's final message.

```python
import json

@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language."""
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })

    # Option 1: Return just the confirmation message
    return result["messages"][-1].text

    # Option 2: Return structured data
    # return json.dumps({
    #     "status": "success",
    #     "event_id": "evt_123",
    #     "summary": result["messages"][-1].text
    # })
```

--------------------------------

### Initialize Multi-Agent Architecture with Tags in Python

Source: https://docs.langchain.com/oss/python/langchain/streaming

Creates a supervisor agent and weather sub-agent with distinct tags for streaming disambiguation. The weather sub-agent is initialized with tags and used as a tool within the supervisor agent. Both models are configured using init_chat_model with OpenAI's GPT-5.2 model.

```python
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, AnyMessage


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"


weather_model = init_chat_model(
    "openai:gpt-5.2",
    tags=["weather_sub_agent"],
)

weather_agent = create_agent(model=weather_model, tools=[get_weather])


def call_weather_agent(query: str) -> str:
    """Query the weather agent."""
    result = weather_agent.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return result["messages"][-1].text


supervisor_model = init_chat_model(
    "openai:gpt-5.2",
    tags=["supervisor"],
)

agent = create_agent(model=supervisor_model, tools=[call_weather_agent])
```

--------------------------------

### Handle Multiple Exception Types in ToolStrategy

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Configure ToolStrategy to retry when any of multiple exception types occur. When handle_errors is a tuple of exception types, the agent retries if the raised exception matches any type in the tuple.

```python
ToolStrategy(
    schema=ProductRating,
    handle_errors=(ValueError, TypeError)  # Retry on ValueError and TypeError
)
```

--------------------------------

### Authenticate User and Update Agent State in LangChain

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

This Python code defines a LangChain tool (`authenticate_user`) that updates the agent's `State` based on authentication results. It uses `ToolRuntime` to return a `Command` that modifies the session-specific `State`, marking the user as authenticated or not. This is useful for managing temporary, session-bound context.

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.types import Command

@tool
def authenticate_user(
    password: str,
    runtime: ToolRuntime
) -> Command:
    """Authenticate user and update State."""
    # Perform authentication (simplified)
    if password == "correct":
        # Write to State: mark as authenticated using Command
        return Command(
            update={"authenticated": True},
        )
    else:
        return Command(update={"authenticated": False})

agent = create_agent(
    model="gpt-4o",
    tools=[authenticate_user]
)
```

--------------------------------

### Define Multi-Stage Prompts for LangChain Agent in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

This Python code defines three distinct prompt templates for a customer support agent's multi-stage workflow. Each prompt is tailored for a specific stage (warranty collection, issue classification, resolution) and includes instructions, current stage context, and placeholders for customer information, enabling dynamic prompt generation based on the agent's state.

```python
# Define prompts as constants for easy reference
WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent helping with device issues.

  CURRENT STAGE: Warranty verification

  At this step, you need to:
  1. Greet the customer warmly
  2. Ask if their device is under warranty
  3. Use record_warranty_status to record their response and move to the next step

  Be conversational and friendly. Don't ask multiple questions at once."""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent helping with device issues.

  CURRENT STAGE: Issue classification
  CUSTOMER INFO: Warranty status is {warranty_status}

  At this step, you need to:
  1. Ask the customer to describe their issue
  2. Determine if it's a hardware issue (physical damage, broken parts) or software issue (app crashes, performance)
  3. Use record_issue_type to record the classification and move to the next step

  If unclear, ask clarifying questions before classifying."""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping with device issues.

  CURRENT STAGE: Resolution
  CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

  At this step, you need to:
  1. For SOFTWARE issues: provide troubleshooting steps using provide_solution
  2. For HARDWARE issues:
     - If IN WARRANTY: explain warranty repair process using provide_solution
     - If OUT OF WARRANTY: escalate_to_human for paid repair options

  Be specific and helpful in your solutions."""
```

--------------------------------

### Implement Safety Guardrail Middleware with Class Syntax

Source: https://docs.langchain.com/oss/python/langchain/guardrails

Creates a SafetyGuardrailMiddleware class that extends AgentMiddleware to validate agent responses using an LLM before returning to users. The after_agent hook intercepts the final AI message, evaluates it with a safety model, and replaces unsafe content with a rejection message. Dependencies include LangChain agent middleware, LanGraph runtime, and chat models.

```python
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from typing import Any

class SafetyGuardrailMiddleware(AgentMiddleware):
    """Model-based guardrail: Use an LLM to evaluate response safety."""

    def __init__(self):
        super().__init__()
        self.safety_model = init_chat_model("gpt-4o-mini")

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Get the final AI response
        if not state["messages"]:
            return None

        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            return None

        # Use a model to evaluate safety
        safety_prompt = f"""Evaluate if this response is safe and appropriate.
        Respond with only 'SAFE' or 'UNSAFE'.

        Response: {last_message.content}"""

        result = self.safety_model.invoke([{"role": "user", "content": safety_prompt}])

        if "UNSAFE" in result.content:
            last_message.content = "I cannot provide that response. Please rephrase your request."

        return None

# Use the safety guardrail
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[SafetyGuardrailMiddleware()],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I make explosives?"}]
})
```

--------------------------------

### LLMToolEmulator - Basic Usage

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Initialize an agent with the LLMToolEmulator middleware to emulate all tool executions using an LLM. This is useful for testing agent behavior without executing real tools.

```APIDOC
## LLMToolEmulator - Basic Usage

### Description
Emulate all tool calls using an LLM instead of executing the actual tools. The LLM generates plausible responses based on tool schemas and descriptions.

### Module
langchain.agents.middleware

### Class
LLMToolEmulator

### Basic Example
```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator

agent = create_agent(
    model="gpt-4o",
    tools=[get_weather, search_database, send_email],
    middleware=[
        LLMToolEmulator(),  # Emulate all tools
    ],
)
```

### Constructor Parameters
- **tools** (list[str | BaseTool]) - Optional - List of tool names or BaseTool instances to emulate. If None (default), ALL tools will be emulated. If empty list [], no tools will be emulated. If array with tool names/instances, only those tools will be emulated.
- **model** (string | BaseChatModel) - Optional - Model to use for generating emulated tool responses. Can be a model identifier string (e.g., 'anthropic:claude-sonnet-4-5-20250929') or a BaseChatModel instance. Defaults to the agent's model if not specified.

### Use Cases
- Testing agent behavior without executing real tools
- Developing agents when external tools are unavailable or expensive
- Prototyping agent workflows before implementing actual tools
```

--------------------------------

### Create agent with context schema in LangChain

Source: https://docs.langchain.com/oss/python/langchain/runtime

Demonstrates how to define a context schema using a dataclass and pass it to create_agent. The context is then provided during agent invocation to inject runtime dependencies like user information.

```python
from dataclasses import dataclass

from langchain.agents import create_agent


@dataclass
class Context:
    user_name: str

agent = create_agent(
    model="gpt-5-nano",
    tools=[...],
    context_schema=Context
)

agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="John Smith")
)
```

--------------------------------

### Configure LangChain Agent with Advanced Tool Retry Middleware Options (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This Python example demonstrates a more comprehensive configuration of `ToolRetryMiddleware` for a LangChain agent. It showcases advanced options including setting a maximum delay, enabling jitter, specifying particular tools for retry logic, defining specific exception types to retry on, and customizing the behavior upon final failure.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware


agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool, api_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=True,
            tools=["api_tool"],
            retry_on=(ConnectionError, TimeoutError),
            on_failure="continue",
        ),
    ],
)
```

--------------------------------

### Content Filter Middleware - Decorator Syntax

Source: https://docs.langchain.com/oss/python/langchain/guardrails

Implements the same content filtering guardrail using decorator syntax with the @before_agent decorator. This approach provides a simpler alternative to class-based middleware while maintaining the same functionality of blocking requests before agent processing.

```python
from typing import Any

from langchain.agents.middleware import before_agent, AgentState, hook_config
from langgraph.runtime import Runtime

banned_keywords = ["hack", "exploit", "malware"]

@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Deterministic guardrail: Block requests containing banned keywords."""
    # Get the first user message
    if not state["messages"]:
        return None

    first_message = state["messages"][0]
    if first_message.type != "human":
        return None

    content = first_message.content.lower()

    # Check for banned keywords
    for keyword in banned_keywords:
        if keyword in content:
            # Block execution before any processing
            return {
                "messages": [{
                    "role": "assistant",
                    "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                }],
                "jump_to": "end"
            }

    return None

# Use the custom guardrail
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[content_filter],
)

# This request will be blocked before any processing
result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I hack into a database?"}]
})
```

--------------------------------

### Define a Tool-Wrapped Knowledge Base Search for Conversational Agents in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This Python snippet demonstrates how to wrap a knowledge base search workflow as a tool. This tool can then be integrated into a conversational agent, allowing the agent to utilize the knowledge base for answering user queries while managing conversational memory with `InMemorySaver`.

```python
from langgraph.checkpoint.memory import InMemorySaver


@tool
def search_knowledge_base(query: str) -> str:
    """Search across multiple knowledge sources (GitHub, Notion, Slack).

    Use this to find information about code, documentation, or team discussions.
    """
    result = workflow.invoke({"query": query})
    return result["final_answer"]


conversational_agent = create_agent(
    model,
    tools=[search_knowledge_base],
    system_prompt=(
        "You are a helpful assistant that answers questions about our organization. "
        "Use the search_knowledge_base tool to find information across our code, "
        "documentation, and team discussions."
    ),
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Create Trajectory Match Evaluator - Strict Mode

Source: https://docs.langchain.com/oss/python/langchain/test

Demonstrates creating a trajectory match evaluator in strict mode to ensure agent messages and tool calls occur in the exact same order as reference trajectory. Useful for enforcing specific operation sequences like policy lookup before authorization.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator


@tool
def get_weather(city: str):
    """Get weather information for a city."""
    return f"It's 75 degrees and sunny in {city}."

agent = create_agent("gpt-4o", tools=[get_weather])

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="strict",
)

def test_weather_tool_called_strict():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in San Francisco?")]
    })

    reference_trajectory = [
        HumanMessage(content="What's the weather in San Francisco?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_weather", "args": {"city": "San Francisco"}}
        ]),
        ToolMessage(content="It's 75 degrees and sunny in San Francisco.", tool_call_id="call_1"),
        AIMessage(content="The weather in San Francisco is 75 degrees and sunny."),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory
    )
    assert evaluation["score"] is True
```

--------------------------------

### Initialize LangChain Agent with Basic Context Editing (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This Python snippet demonstrates how to set up a LangChain agent with `ContextEditingMiddleware` to manage conversation context. It uses `ClearToolUsesEdit` to automatically clear older tool outputs when the token count exceeds 100,000, preserving the 3 most recent results. This helps in long conversations to prevent exceeding token limits and reduce costs by removing less relevant historical data.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=100000,
                    keep=3,
                ),
            ],
        ),
    ],
)
```

--------------------------------

### Add Metadata and Tags to Agent Invoke Configuration (Python)

Source: https://docs.langchain.com/oss/python/langchain/observability

This snippet demonstrates how to attach custom metadata and tags directly to a LangChain agent's invocation using the 'config' parameter. These annotations are applied specifically to the trace generated by this execution, aiding in focused debugging and monitoring within LangSmith.

```python
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Send a welcome email"}]},
    config={
        "tags": ["production", "email-assistant", "v1.0"],
        "metadata": {
            "user_id": "user_123",
            "session_id": "session_456",
            "environment": "production"
        }
    }
)
```

--------------------------------

### Create LangChain SQL Database Toolkit and List Tools (Python)

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

This Python snippet demonstrates how to create a `SQLDatabaseToolkit` from `langchain_community.agent_toolkits`, essential for integrating database interactions into LangChain agents. It requires an initialized `SQLDatabase` instance and an LLM, then proceeds to retrieve and print the names and descriptions of the available tools, such as `sql_db_query` and `sql_db_schema`.

```python
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")
```

--------------------------------

### Create LangGraph Configuration File

Source: https://docs.langchain.com/oss/python/langchain/studio

Create a langgraph.json configuration file that specifies project dependencies, defines graph locations, and references the environment file. The CLI uses this to locate your agent and manage dependencies.

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/agent.py:agent"
  },
  "env": ".env"
}
```

--------------------------------

### Generated SQL Query for High-Value Customers

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

This SQL query, generated by the LangChain agent, identifies distinct customers who made completed orders over $1000 in the last month. It joins customer and order tables, filters by amount, status, and date, then orders the results by customer ID.

```sql
SELECT DISTINCT
    c.customer_id,
    c.name,
    c.email,
    c.customer_tier
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.total_amount > 1000
  AND o.status = 'completed'
  AND o.order_date >= CURRENT_DATE - INTERVAL '1 month'
ORDER BY c.customer_id;
```

--------------------------------

### Terminate Agent Execution Early with Python Interceptors

Source: https://docs.langchain.com/oss/python/langchain/mcp

This interceptor shows how to use a `Command` object with `goto="__end__"` to gracefully stop the agent's execution when a specific condition is met, such as a 'mark_complete' action. This is useful for ending a run early once a primary goal is achieved.

```python
async def end_on_success(
    request: MCPToolCallRequest,
    handler,
):
    """End agent run when task is marked complete."""
    result = await handler(request)

    if request.name == "mark_complete":
        return Command(
            update={"messages": [result], "status": "done"},
            goto="__end__",  # [!code highlight]
        )

    return result
```

--------------------------------

### Implement Agent Jump with Decorator Pattern

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Returns a dictionary with 'jump_to' key to exit early from middleware execution. The decorator approach uses @after_model and @hook_config to define jump targets. This example checks for blocked content and jumps to the end of agent execution if detected.

```python
from langchain.agents.middleware import after_model, hook_config, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any


@after_model
@hook_config(can_jump_to=["end"])
def check_for_blocked(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last_message = state["messages"][-1]
    if "BLOCKED" in last_message.content:
        return {
            "messages": [AIMessage("I cannot respond to that request.")],
            "jump_to": "end"
        }
    return None
```

--------------------------------

### Create Python Tools for Agent Workflow State Management

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

This snippet defines several Python tools using the `@tool` decorator. `record_warranty_status` and `record_issue_type` are key, as they return `Command` objects to update the agent's state, including transitioning the `current_step`. `escalate_to_human` and `provide_solution` are example tools for problem resolution. These tools enable dynamic state transitions and information recording within the agent's workflow.

```python
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command

@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:  # [!code highlight]
    """Record the customer's warranty status and transition to issue classification."""
    return Command(  # [!code highlight]
        update={  # [!code highlight]
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded as: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",  # [!code highlight]
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:  # [!code highlight]
    """Record the type of issue and transition to resolution specialist."""
    return Command(  # [!code highlight]
        update={  # [!code highlight]
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",  # [!code highlight]
        }
    )


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the case to a human support specialist."""
    # In a real system, this would create a ticket, notify staff, etc.
    return f"Escalating to human support. Reason: {reason}"


@tool
def provide_solution(solution: str) -> str:
    """Provide a solution to the customer's issue."""
    return f"Solution provided: {solution}"
```

--------------------------------

### Invoke a Conversational Agent with Multi-Turn Memory in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This Python code illustrates how to invoke a configured conversational agent for a multi-turn conversation. It sets up a `thread_id` for consistent memory and demonstrates two consecutive user queries, showcasing the agent's ability to retain context across turns thanks to the `InMemorySaver` checkpointer.

```python
config = {"configurable": {"thread_id": "user-123"}}

result = conversational_agent.invoke(
    {"messages": [{"role": "user", "content": "How do I authenticate API requests?"}]},
    config
)
print(result["messages"][-1].content)

result = conversational_agent.invoke(
    {"messages": [{"role": "user", "content": "What about rate limiting for those endpoints?"}]},
    config
)
print(result["messages"][-1].content)
```

--------------------------------

### Create Agent with In-Memory Checkpointer

Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

Initialize an agent with in-memory short-term memory persistence using InMemorySaver. This approach stores conversation state within the application's memory and associates it with a thread_id for conversation management. Suitable for development and testing environments.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    "gpt-5",
    tools=[get_user_info],
    checkpointer=InMemorySaver(),
)

agent.invoke(
    {"messages": [{"role": "user", "content": "Hi! My name is Bob."}]},
    {"configurable": {"thread_id": "1"}},
)
```

--------------------------------

### Configure LangChain Agent with Detailed Context Editing for Token Management (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This Python example provides a detailed configuration for a LangChain agent using `ContextEditingMiddleware` to proactively manage conversation tokens. It configures `ClearToolUsesEdit` to trigger context clearing when the conversation reaches 2000 tokens, ensuring the 3 most recent tool results are always preserved. Additional options like `clear_tool_inputs=False` and a custom `placeholder` are demonstrated for fine-grained control over context editing, enabling efficient token usage.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit


agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, your_calculator_tool, database_tool],
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=2000,
                    keep=3,
                    clear_tool_inputs=False,
                    exclude_tools=[],
                    placeholder="[cleared]",
                ),
            ],
        ),
    ],
)
```

--------------------------------

### Import LangChain and LangGraph Dependencies in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Imports core modules from LangChain and LangGraph libraries including agent creation functions, middleware classes, message types, checkpointing utilities, and type hints. These imports provide the foundation for building an agent with middleware support and in-memory persistence.

```python
import uuid
from typing import TypedDict, NotRequired, Callable
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, AgentMiddleware
from langchain.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

```

--------------------------------

### LLMToolSelectorMiddleware Configuration

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Initialize the LLM Tool Selector middleware with an agent. This middleware uses structured output to ask an LLM which tools are most relevant for the current query, filtering irrelevant tools and improving model focus.

```APIDOC
## LLMToolSelectorMiddleware

### Description
Middleware component that intelligently selects relevant tools using an LLM before calling the main model. Useful for agents with many tools (10+) where most aren't relevant per query.

### Usage
Import and configure the middleware when creating an agent.

### Parameters

#### Configuration Options
- **model** (string | BaseChatModel) - Optional - Model for tool selection. Can be a model identifier string (e.g., 'openai:gpt-4o-mini') or a BaseChatModel instance. Defaults to the agent's main model.
- **system_prompt** (string) - Optional - Instructions for the selection model. Uses built-in prompt if not specified.
- **max_tools** (number) - Optional - Maximum number of tools to select. If the model selects more, only the first max_tools will be used. No limit if not specified.
- **always_include** (list[string]) - Optional - Tool names to always include regardless of selection. These do not count against the max_tools limit.

### Request Example
```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolSelectorMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[tool1, tool2, tool3, tool4, tool5],
    middleware=[
        LLMToolSelectorMiddleware(
            model="gpt-4o-mini",
            max_tools=3,
            always_include=["search"]
        )
    ]
)
```

### Key Features
- Reduces token usage by filtering irrelevant tools
- Improves model focus and accuracy
- Ideal for agents with 10+ tools
- Uses structured output for tool selection
- Supports always-included tools that bypass filtering

### Reference
API Reference: [`LLMToolSelectorMiddleware`](https://reference.langchain.com/python/langchain/middleware/#langchain.agents.middleware.LLMToolSelectorMiddleware)
```

--------------------------------

### Stream LLM tokens with agent.stream() - Python

Source: https://docs.langchain.com/oss/python/langchain/streaming

Creates a LangChain agent with a weather tool and streams individual tokens as they are produced by the LLM using stream_mode='messages'. The agent processes user queries and outputs token chunks with metadata about which LangGraph node produced them, enabling real-time token-level visibility into model execution.

```python
from langchain.agents import create_agent


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_weather],
)
for token, metadata in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
):
    print(f"node: {metadata['langgraph_node']}")
    print(f"content: {token.content_blocks}")
    print("\n")
```

--------------------------------

### Define Middleware to Return Source Documents in RAG (Langchain, Python)

Source: https://docs.langchain.com/oss/python/langchain/rag

This Python code defines a custom `AgentMiddleware` called `RetrieveDocumentsMiddleware` to enhance a two-step RAG chain. It extends the agent's state with a `context` key to store retrieved documents. The `before_model` method performs a similarity search based on the last user message, populates the `context` with `Document` objects, and then incorporates their content into a system message for the LLM. This allows access to document metadata in the application state alongside context injection.

```python
from typing import Any
from langchain_core.documents import Document
from langchain.agents.middleware import AgentMiddleware, AgentState


class State(AgentState):
    context: list[Document]


class RetrieveDocumentsMiddleware(AgentMiddleware[State]):
    state_schema = State

    def before_model(self, state: AgentState) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        retrieved_docs = vector_store.similarity_search(last_message.text)

        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
```

--------------------------------

### Configure LangChain Agent with Basic Tool Retry Middleware (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This Python code snippet initializes a LangChain agent and applies `ToolRetryMiddleware` to automatically retry failed tool calls. It sets basic retry parameters such as the maximum number of retries, the exponential backoff factor, and the initial delay, which are essential for handling transient failures in external API calls.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)
```

--------------------------------

### Test Multi-Turn Agent Workflow

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Tests the complete workflow across multiple conversation turns, demonstrating state transitions from warranty collection to issue classification to resolution. Uses a unique thread_id to maintain conversation state and shows how the agent progresses through configured steps.

```python
from langchain.messages import HumanMessage
import uuid

# Configuration for this conversation thread
thread_id = str(uuid.uuid4())
config = {"configurable": {"thread_id": thread_id}}

# Turn 1: Initial message - starts with warranty_collector step
print("=== Turn 1: Warranty Collection ===")
result = agent.invoke(
    {"messages": [HumanMessage("Hi, my phone screen is cracked")]},
    config
)
for msg in result['messages']:
    msg.pretty_print()

# Turn 2: User responds about warranty
print("\n=== Turn 2: Warranty Response ===")
result = agent.invoke(
    {"messages": [HumanMessage("Yes, it's still under warranty")]},
    config
)
for msg in result['messages']:
    msg.pretty_print()
print(f"Current step: {result.get('current_step')}")

# Turn 3: User describes the issue
print("\n=== Turn 3: Issue Description ===")
result = agent.invoke(
    {"messages": [HumanMessage("The screen is physically cracked from dropping it")]},
    config
)
for msg in result['messages']:
    msg.pretty_print()
print(f"Current step: {result.get('current_step')}")

# Turn 4: Resolution
print("\n=== Turn 4: Resolution ===")
result = agent.invoke(
    {"messages": [HumanMessage("What should I do?")]},
    config
)
for msg in result['messages']:
    msg.pretty_print()
```

--------------------------------

### Implement State-Driven LangChain Agent with Middleware for Customer Support in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

This comprehensive Python example demonstrates a customer support agent using LangChain with middleware. It defines a `SupportState` to track the current step, a `record_warranty_status` tool to update the state, and `apply_step_config` middleware to dynamically modify the system prompt and tools based on the `current_step`. The agent is then created with this middleware and a state schema, allowing for adaptive behavior.

```python
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Callable

# 1. Define state with current_step tracker
class SupportState(AgentState):
    """Track which step is currently active."""
    current_step: str = "triage"
    warranty_status: str | None = None

# 2. Tools update current_step via Command
@tool
def record_warranty_status(
    status: str,
    runtime: ToolRuntime[None, SupportState]
) -> Command:
    """Record warranty status and transition to next step."""
    return Command(update={
        "messages": [
            ToolMessage(
                content=f"Warranty status recorded: {status}",
                tool_call_id=runtime.tool_call_id
            )
        ],
        "warranty_status": status,
        # Transition to next step
        "current_step": "specialist"
    })

# 3. Middleware applies dynamic configuration based on current_step
@wrap_model_call
def apply_step_config(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Configure agent behavior based on current_step."""
    step = request.state.get("current_step", "triage")

    # Map steps to their configurations
    configs = {
        "triage": {
            "prompt": "Collect warranty information...",
            "tools": [record_warranty_status]
        },
        "specialist": {
            "prompt": "Provide solutions based on warranty: {warranty_status}",
            "tools": [provide_solution, escalate]
        }
    }

    config = configs[step]
    request = request.override(
        system_prompt=config["prompt"].format(**request.state),
        tools=config["tools"]
    )
    return handler(request)

# 4. Create agent with middleware
agent = create_agent(
    model,
    tools=[record_warranty_status, provide_solution, escalate],
    state_schema=SupportState,
    middleware=[apply_step_config],
    checkpointer=InMemorySaver()
)
```

--------------------------------

### Create and Use Asynchronous Trajectory Evaluators in Python

Source: https://docs.langchain.com/oss/python/langchain/test

This Python example demonstrates creating and using asynchronous LLM-based trajectory judges and match evaluators from `agentevals`. It shows how to instantiate async evaluators and invoke them within an `async` function to evaluate agent outputs asynchronously.

```python
from agentevals.trajectory.llm import create_async_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT
from agentevals.trajectory.match import create_async_trajectory_match_evaluator

async_judge = create_async_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

async_evaluator = create_async_trajectory_match_evaluator(
    trajectory_match_mode="strict",
)

async def test_async_evaluation():
    result = await agent.ainvoke({
        "messages": [HumanMessage(content="What's the weather?")]
    })

    evaluation = await async_judge(outputs=result["messages"])
    assert evaluation["score"] is True
```

--------------------------------

### Configure LangChain Agent with Global and Tool-Specific Call Limits in Python

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This Python code demonstrates how to initialize a LangChain agent with `ToolCallLimitMiddleware`. It sets both a global limit for all tools and a specific limit for the 'search' tool, controlling the maximum number of invocations per thread and per run. This prevents excessive tool usage and manages resource consumption.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        # Global limit
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
        # Tool-specific limit
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=5,
            run_limit=3,
        ),
    ],
)
```

--------------------------------

### Define Pydantic Models for Product Review

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Defines two Pydantic BaseModel classes for structured data validation: ProductReview for analyzing reviews with rating, sentiment, and key points; and CustomerComplaint for categorizing issues with type, severity, and description. Uses Literal types and Field constraints for input validation.

```python
from pydantic import BaseModel, Field
from typing import Literal, Union

class ProductReview(BaseModel):
    """Analysis of a product review."""
    rating: int | None = Field(description="The rating of the product", ge=1, le=5)
    sentiment: Literal["positive", "negative"] = Field(description="The sentiment of the review")
    key_points: list[str] = Field(description="The key points of the review. Lowercase, 1-3 words each.")

class CustomerComplaint(BaseModel):
    """A customer complaint about a product or service."""
    issue_type: Literal["product", "service", "shipping", "billing"] = Field(description="The type of issue")
    severity: Literal["low", "medium", "high"] = Field(description="The severity of the complaint")
    description: str = Field(description="Brief description of the complaint")
```

--------------------------------

### Install and Configure Voyage AI Embeddings

Source: https://docs.langchain.com/oss/python/langchain/rag

Install langchain-voyageai and set up Voyage AI embeddings with the voyage-3 model. Requires VOYAGE_API_KEY environment variable for authentication with Voyage AI services.

```shell
pip install -qU langchain-voyageai
```

```python
import getpass
import os

if not os.environ.get("VOYAGE_API_KEY"):
    os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter API key for Voyage AI: ")

from langchain-voyageai import VoyageAIEmbeddings

embeddings = VoyageAIEmbeddings(model="voyage-3")
```

--------------------------------

### Custom Error Message with ToolStrategy

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Configure ToolStrategy to use a custom error message that will be sent to the model when tool validation fails. When handle_errors is a string, the agent prompts the model to retry with the specified message.

```python
ToolStrategy(
    schema=ProductRating,
    handle_errors="Please provide a valid rating between 1-5 and include a comment."
)
```

--------------------------------

### State-Based Tool Selection in LangChain Agents

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Filter tools dynamically based on conversation state such as authentication status and message count. Uses middleware wrapper to check state properties and override the request with appropriate tools. Enables sensitive tools only after authentication and limits advanced tools early in conversations.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def state_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on conversation State."""
    # Read from State: check if user has authenticated
    state = request.state
    is_authenticated = state.get("authenticated", False)
    message_count = len(state["messages"])

    # Only enable sensitive tools after authentication
    if not is_authenticated:
        tools = [t for t in request.tools if t.name.startswith("public_")]
        request = request.override(tools=tools)
    elif message_count < 5:
        # Limit tools early in conversation
        tools = [t for t in request.tools if t.name != "advanced_search"]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[public_search, private_search, advanced_search],
    middleware=[state_based_tools]
)
```

--------------------------------

### Configure PII Detection Middleware in LangChain Agents (Python)

Source: https://docs.langchain.com/oss/python/langchain/guardrails

This Python code demonstrates how to integrate `PIIMiddleware` into a LangChain agent. It configures the middleware to redact emails, mask credit cards, and block API keys from user input before processing by the model, enhancing security and compliance. The middleware can apply various strategies like `redact`, `mask`, `hash`, or `block` for different PII types.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware


agent = create_agent(
    model="gpt-4o",
    tools=[customer_service_tool, email_tool],
    middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
        ),
    ],
)
```

--------------------------------

### Extract and display loaded document content

Source: https://docs.langchain.com/oss/python/langchain/rag

Display a preview of the loaded document's page content. Shows the first 500 characters of the parsed HTML to verify successful loading and parsing of the web content.

```python
print(docs[0].page_content[:500])
```

--------------------------------

### Configure ShellToolMiddleware with Docker isolation

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Sets up an agent with ShellToolMiddleware using DockerExecutionPolicy to launch isolated Docker containers for each agent run. Includes startup commands for environment setup and a command timeout. This provides stronger isolation than host execution while maintaining flexibility.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ShellToolMiddleware,
    DockerExecutionPolicy,
)

agent_docker = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            startup_commands=["pip install requests", "export PYTHONPATH=/workspace"],
            execution_policy=DockerExecutionPolicy(
                image="python:3.11-slim",
                command_timeout=60.0,
            ),
        ),
    ],
)
```

--------------------------------

### Configure LangChain Agent with Summarization Middleware (Python)

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

This snippet demonstrates how to initialize a LangChain agent and integrate `SummarizationMiddleware`. The middleware is configured to use 'gpt-4o-mini' for summarization, triggered when the conversation exceeds 4000 tokens, and keeps the last 20 messages intact while summarizing older ones. This ensures conversation history is condensed persistently.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger={"tokens": 4000},
            keep={"messages": 20},
        ),
    ],
)
```

--------------------------------

### Implement Two-Step RAG Chain with Dynamic Prompt (Langchain, Python)

Source: https://docs.langchain.com/oss/python/langchain/rag

This Python code defines a `dynamic_prompt` function that retrieves documents based on the last user query from a vector store and injects their content as a system message into the LLM prompt. It then creates an agent without tools, using this dynamic prompt to ensure retrieval always occurs before the model generates a response. This method reduces inference calls compared to agentic RAG by directly incorporating retrieved context.

```python
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    retrieved_docs = vector_store.similarity_search(last_query)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    system_message = (
        "You are a helpful assistant. Use the following context in your response:"
        f"\n\n{docs_content}"
    )

    return system_message


agent = create_agent(model, tools=[], middleware=[prompt_with_context])
```

--------------------------------

### Implement Python Langchain Agent Safety Guardrail with Custom Stream Writer

Source: https://docs.langchain.com/oss/python/langchain/streaming

This Python code defines a `safety_guardrail` middleware for Langchain agents. It intercepts the agent's last message, uses a separate LLM to evaluate its safety based on a `ResponseSafety` Pydantic model, and emits this evaluation as a custom stream update using `get_stream_writer()`. If the response is deemed unsafe, it modifies the original message content to a canned refusal.

```python
from typing import Any, Literal

from langchain.agents.middleware import after_agent, AgentState
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from langgraph.config import get_stream_writer
from pydantic import BaseModel


class ResponseSafety(BaseModel):
    """Evaluate a response as safe or unsafe."""
    evaluation: Literal["safe", "unsafe"]


safety_model = init_chat_model("openai:gpt-5.2")

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Model-based guardrail: Use an LLM to evaluate response safety."""
    stream_writer = get_stream_writer()
    # Get the model response
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None

    # Use another model to evaluate safety
    model_with_tools = safety_model.bind_tools([ResponseSafety], tool_choice="any")
    result = model_with_tools.invoke(
        [
            {
                "role": "system",
                "content": "Evaluate this AI response as generally safe or unsafe.",
            }
        ],
        {"role": "user", "content": f"AI response: {last_message.text}"},
    )
    stream_writer(result)

    tool_call = result.tool_calls[0]
    if tool_call["args"]["evaluation"] == "unsafe":
        last_message.content = "I cannot provide that response. Please rephrase your request."

    return None
```

--------------------------------

### Install Langchain-Postgres for PGVectorStore

Source: https://docs.langchain.com/oss/python/langchain/rag

This snippet provides the shell command to install the `langchain-postgres` package. This package is a prerequisite for utilizing both the `PGVector` and `PGVectorStore` functionalities in LangChain, enabling interaction with PostgreSQL databases.

```shell
pip install -qU langchain-postgres
```

--------------------------------

### View Custom Tool Error Message Output in LangChain Python

Source: https://docs.langchain.com/oss/python/langchain/agents

This snippet illustrates the expected structure of a `ToolMessage` when a tool call fails and is handled by the custom error middleware shown previously. It demonstrates how the `content` field of the `ToolMessage` contains the custom error string, informing the model about the tool's failure with a tailored message.

```python
[
    ...
    ToolMessage(
        content="Tool error: Please check your input and try again. (division by zero)",
        tool_call_id="..."
    ),
    ...
]
```

--------------------------------

### Implement Decorator-Based Middleware for LangChain Agents in Python

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

This snippet demonstrates how to use decorator-based middleware in LangChain agents. It includes an `@before_model` decorator for logging model calls and a `@wrap_model_call` decorator for implementing retry logic. The middleware functions are passed directly to the `create_agent` function.

```python
from langchain.agents.middleware import (
    before_model,
    wrap_model_call,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langchain.agents import create_agent
from langgraph.runtime import Runtime
from typing import Any, Callable


@before_model
def log_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"About to call model with {len(state['messages'])} messages")
    return None

@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")

agent = create_agent(
    model="gpt-4o",
    middleware=[log_before_model, retry_model],
    tools=[...],
)
```

--------------------------------

### TodoListMiddleware Integration

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Integrate `TodoListMiddleware` into an agent to provide task planning and tracking capabilities, especially useful for complex multi-step tasks.

```APIDOC
## TodoListMiddleware Integration

### Description
Integrate `TodoListMiddleware` into an agent to provide task planning and tracking capabilities. This is particularly useful for complex, multi-step tasks or long-running operations where progress visibility is important. The middleware automatically provides agents with a `write_todos` tool and relevant system prompts.

### Method
Middleware Configuration

### Endpoint
`TodoListMiddleware`

### Parameters
#### Configuration Options for `TodoListMiddleware`
- **system_prompt** (string) - Optional - Custom system prompt for guiding todo usage. If not specified, a built-in prompt is used.
- **tool_description** (string) - Optional - Custom description for the `write_todos` tool. If not specified, a built-in description is used.

### Request Example
```python
from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[read_file, write_file, run_tests],
    middleware=[TodoListMiddleware()],
)
```
```

--------------------------------

### Download SQLite Chinook Database (Python)

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

This Python script downloads the `Chinook.db` SQLite sample database from a Google Cloud Storage bucket if it doesn't already exist locally. It uses the `requests` library for HTTP requests and `pathlib` for file system operations, printing status messages upon success or failure.

```python
import requests, pathlib

url = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
local_path = pathlib.Path("Chinook.db")

if local_path.exists():
    print(f"{local_path} already exists, skipping download.")
else:
    response = requests.get(url)
    if response.status_code == 200:
        local_path.write_bytes(response.content)
        print(f"File downloaded and saved as {local_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")
```

--------------------------------

### Define Custom Agent State Schema in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

This snippet defines a custom `AgentState` schema, `SupportState`, in Python using `langchain.agents.AgentState`. It includes `current_step` to track workflow progression and other fields like `warranty_status` and `issue_type` to store context. This state schema is crucial for managing multi-step agent workflows.

```python
from langchain.agents import AgentState
from typing_extensions import NotRequired
from typing import Literal

# Define the possible workflow steps
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]  # [!code highlight]

class SupportState(AgentState):  # [!code highlight]
    """State for customer support workflow."""
    current_step: NotRequired[SupportStep]  # [!code highlight]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]
```

--------------------------------

### Integrate Message Deletion into LangChain Agent Middleware (Python)

Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

This Python example integrates message deletion as a middleware within a LangChain agent to automatically manage conversation length. The `delete_old_messages` function, decorated with `@after_model`, removes the earliest two messages from the `AgentState` after each model interaction. It sets up an `AgentState` with a `system_prompt` and `InMemorySaver`, then demonstrates streaming an agent conversation where old messages are automatically pruned.

```python
from langchain.messages import RemoveMessage
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig


@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None


agent = create_agent(
    "gpt-5-nano",
    tools=[],
    system_prompt="Please be concise and to the point.",
    middleware=[delete_old_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

for event in agent.stream(
    {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])

for event in agent.stream(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    config,
    stream_mode="values",
):
    print([(message.type, message.content) for message in event["messages"]])
```

--------------------------------

### LLM-as-Judge Trajectory Evaluator Without Reference in Python

Source: https://docs.langchain.com/oss/python/langchain/test

Uses an LLM-based evaluator to assess agent execution quality without requiring a reference trajectory. The evaluator employs a language model to judge trajectory accuracy and appropriateness, returning a score and explanatory comment about the trajectory quality.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.llm import create_trajectory_llm_as_judge, TRAJECTORY_ACCURACY_PROMPT


@tool
def get_weather(city: str):
    """Get weather information for a city."""
    return f"It's 75 degrees and sunny in {city}."

agent = create_agent("gpt-4o", tools=[get_weather])

evaluator = create_trajectory_llm_as_judge(
    model="openai:o3-mini",
    prompt=TRAJECTORY_ACCURACY_PROMPT,
)

def test_trajectory_quality():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's the weather in Seattle?")]
    })

    evaluation = evaluator(
        outputs=result["messages"],
    )
    assert evaluation["score"] is True
```

--------------------------------

### Configure LangChain Language Model for Agent (Python)

Source: https://docs.langchain.com/oss/python/langchain/quickstart

This snippet shows how to initialize a language model using `init_chat_model` from LangChain. It configures the model with specific parameters such as the model ID, temperature for response creativity, timeout duration, and maximum token limit for output, ensuring consistent agent responses.

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0.5,
    timeout=10,
    max_tokens=1000
)
```

--------------------------------

### Define LangGraph Handoff Tool for Sales Agent Transfer

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

This Python function creates a `transfer_to_sales` tool for LangGraph. It extracts the last AI message, wraps it in a `ToolMessage`, and returns a `Command` to transfer control to a 'sales_agent' node. The command updates the active agent and message history, using `Command.PARENT` for inter-subgraph communication.

```python
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool, ToolRuntime
from langgraph.types import Command

@tool
def transfer_to_sales(
    runtime: ToolRuntime,
) -> Command:
    """Transfer to the sales agent."""
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    transfer_message = ToolMessage(
        content="Transferred to sales agent",
        tool_call_id=runtime.tool_call_id,
    )
    return Command(
        goto="sales_agent",
        update={
            "active_agent": "sales_agent",
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT
    )
```

--------------------------------

### Update LangChain Agent State with Tools in Python

Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

This snippet demonstrates how to modify an agent's short-term memory (state) by returning state updates directly from tools. It defines a CustomState and CustomContext, then uses an 'update_user_info' tool to update 'user_name' in the state and a 'greet' tool to access it, showcasing how tools can persist and retrieve intermediate results within the agent's execution flow.

```python
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel


class CustomState(AgentState):
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
       return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool it will get and update the user's name.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello {user_name}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[update_user_info, greet],
    state_schema=CustomState,
    context_schema=CustomContext,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
)
```

--------------------------------

### Implement Multi-Source Knowledge Router with LangChain and LangGraph

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This comprehensive Python script demonstrates a multi-source knowledge router for multi-agent systems. It defines a state graph (`RouterState`), uses LangChain agents (GitHub, Notion, Slack) with specific tools, and a classifier to route user queries. The system processes queries by classifying them, invoking relevant agents in parallel, and accumulating results for synthesis into a final answer. Key components include Pydantic models for state and classification, `langgraph` for orchestration, and `langchain` for agents and tools. Dependencies include `langchain`, `langgraph`, and `pydantic`.

```python
"""
Multi-Source Knowledge Router Example

This example demonstrates the router pattern for multi-agent systems.
A router classifies queries, routes them to specialized agents in parallel,
and synthesizes results into a combined response.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from pydantic import BaseModel, Field


# State definitions
class AgentInput(TypedDict):
    """Simple input state for each subagent."""
    query: str


class AgentOutput(TypedDict):
    """Output from each subagent."""
    source: str
    result: str


class Classification(TypedDict):
    """A single routing decision: which agent to call with what query."""
    source: Literal["github", "notion", "slack"]
    query: str


class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str


# Structured output schema for classifier
class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""
    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )


# Tools
@tool
def search_code(query: str, repo: str = "main") -> str:
    """Search code in GitHub repositories."""
    return f"Found code matching '{query}' in {repo}: authentication middleware in src/auth.py"


@tool
def search_issues(query: str) -> str:
    """Search GitHub issues and pull requests."""
    return f"Found 3 issues matching '{query}': #142 (API auth docs), #89 (OAuth flow), #203 (token refresh)"


@tool
def search_prs(query: str) -> str:
    """Search pull requests for implementation details."""
    return f"PR #156 added JWT authentication, PR #178 updated OAuth scopes"


@tool
def search_notion(query: str) -> str:
    """Search Notion workspace for documentation."""
    return f"Found documentation: 'API Authentication Guide' - covers OAuth2 flow, API keys, and JWT tokens"


@tool
def get_page(page_id: str) -> str:
    """Get a specific Notion page by ID."""
    return f"Page content: Step-by-step authentication setup instructions"


@tool
def search_slack(query: str) -> str:
    """Search Slack messages and threads."""
    return f"Found discussion in #engineering: 'Use Bearer tokens for API auth, see docs for refresh flow'"


@tool
def get_thread(thread_id: str) -> str:
    """Get a specific Slack thread."""
    return f"Thread discusses best practices for API key rotation"


# Models and agents
model = init_chat_model("openai:gpt-4o")
router_llm = init_chat_model("openai:gpt-4o-mini")

github_agent = create_agent(
    model,
    tools=[search_code, search_issues, search_prs],
    system_prompt=(
        "You are a GitHub expert. Answer questions about code, "
        "API references, and implementation details by searching "
        "repositories, issues, and pull requests."
    ),
)

notion_agent = create_agent(
    model,
    tools=[search_notion, get_page],
    system_prompt=(
        "You are a Notion expert. Answer questions about internal "
        "processes, policies, and team documentation by searching "
        "the organization's Notion workspace."
    ),
)

slack_agent = create_agent(
    model,
    tools=[search_slack, get_thread],
    system_prompt=(
        "You are a Slack expert. Answer questions by searching "
        "relevant threads and discussions where team members have "
        "shared knowledge and solutions."
    ),
)


# Workflow nodes
def classify_query(state: RouterState) -> dict:
    """Classify query and determine which agents to invoke."""
    structured_llm = router_llm.with_structured_output(ClassificationResult)

    result = structured_llm.invoke([
        {
            "role": "system",
            "content": """Analyze this query and determine which knowledge bases to consult.
For each relevant source, generate a targeted sub-question optimized for that source.

Available sources:
- github: Code, API references, implementation details, issues, pull requests
- notion: Internal documentation, processes, policies, team wikis
- slack: Team discussions, informal knowledge sharing, recent conversations

Return ONLY the sources that are relevant to the query."""
        },
        {"role": "user", "content": state["query"]}
    ])
```

--------------------------------

### Stream Langchain Agent Output with Custom Middleware Events in Python

Source: https://docs.langchain.com/oss/python/langchain/streaming

This Python example shows how to create a Langchain agent, incorporating the previously defined `safety_guardrail` middleware. It then streams the agent's execution, processing different `stream_mode` types including `messages`, `updates`, and crucially, `custom` events. The `custom` events are used to access and print the safety evaluation tool calls emitted by the middleware.

```python
from typing import Any

from langchain.agents import create_agent
from langchain.messages import AIMessageChunk, AIMessage, AnyMessage


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return "It's always sunny in {city}!"


agent = create_agent(
    model="openai:gpt-5.2",
    tools=[get_weather],
    middleware=[safety_guardrail],
)

def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")


input_message = {"role": "user", "content": "What is the weather in Boston?"}
for stream_mode, data in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates", "custom"],
):
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)
    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):
                _render_completed_message(update["messages"][-1])
    if stream_mode == "custom":
        # access completed message in stream
        print(f"Tool calls: {data.tool_calls}")
```

--------------------------------

### Configure LangSmith API Key in .env File

Source: https://docs.langchain.com/oss/python/langchain/studio

Create a .env file in your project root to store the LangSmith API key required for Studio to connect to your local agent. Keep this file out of version control.

```bash
LANGSMITH_API_KEY=lsv2...
```

--------------------------------

### Configure Middleware with Custom State Schema

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Updates the AgentMiddleware to use the custom state schema and registers constrained tools. The middleware injects skill descriptions and enforces state-based constraints through the tool registration.

```python
class SkillMiddleware(AgentMiddleware[CustomState]):
    """Middleware that injects skill descriptions into the system prompt."""

    state_schema = CustomState
    tools = [load_skill, write_sql_query]
```

--------------------------------

### Define Custom State for Skill Tracking

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Extends the base AgentState to include a skills_loaded field that tracks which skills have been loaded into the agent's context. This custom state schema enables state-based constraints on tool availability.

```python
from langchain.agents.middleware import AgentState

class CustomState(AgentState):
    skills_loaded: NotRequired[list[str]]
```

--------------------------------

### Synthesize Agent Results with LangChain LLM in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This Python function synthesizes information gathered from various agents into a single, coherent answer. It formats the individual results, constructs a system prompt with instructions, and uses a 'router_llm' to generate a consolidated response based on the original query. If no results are available, it provides a default message.

```python
  def synthesize_results(state: RouterState) -> dict:
      """Combine results from all agents into a coherent answer."""
      if not state["results"]:
          return {"final_answer": "No results found from any knowledge source."}

      formatted = [
          f"**From {r['source'].title()}:**\n{r['result']}"
          for r in state["results"]
      ]

      synthesis_response = router_llm.invoke([
          {
              "role": "system",
              "content": f"""Synthesize these search results to answer the original question: "{state['query']}"\n\n  - Combine information from multiple sources without redundancy\n  - Highlight the most relevant and actionable information\n  - Note any discrepancies between sources\n  - Keep the response concise and well-organized"""
          },
          {"role": "user", "content": "\n\n".join(formatted)}
      ])

      return {"final_answer": synthesis_response.content}
```

--------------------------------

### Implement Agent Jump with Class-Based Middleware

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Extends AgentMiddleware class to implement jump functionality with after_model hook. Checks for blocked content in messages and returns jump_to directive to end execution. Provides object-oriented approach to middleware implementation.

```python
from langchain.agents.middleware import AgentMiddleware, hook_config, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

class BlockedContentMiddleware(AgentMiddleware):
    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last_message = state["messages"][-1]
        if "BLOCKED" in last_message.content:
            return {
                "messages": [AIMessage("I cannot respond to that request.")],
                "jump_to": "end"
            }
        return None
```

--------------------------------

### Stack Multiple Guardrails for Layered Protection

Source: https://docs.langchain.com/oss/python/langchain/guardrails

Demonstrates combining multiple middleware guardrails in a single agent configuration to implement layered protection. The example stacks four layers: deterministic input filtering, PII redaction on input and output, human-in-the-loop approval for sensitive tools, and model-based safety validation. Guardrails execute in order, enabling comprehensive security coverage across the agent pipeline.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, send_email_tool],
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),

        # Layer 2: PII protection (before and after model)
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # Layer 3: Human approval for sensitive tools
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),

        # Layer 4: Model-based safety check (after agent)
        SafetyGuardrailMiddleware(),
    ],
)
```

--------------------------------

### Illustrate LangChain Agent Middleware Execution Order (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

This Python snippet demonstrates how to initialize a LangChain agent with multiple middleware instances. It highlights the execution sequence of `before_*` hooks (first to last), `after_*` hooks (last to first), and the nested behavior of `wrap_*` hooks, which is essential for understanding how state changes and side effects propagate across different middleware components during an agent's operation.

```python
agent = create_agent(
    model="gpt-4o",
    middleware=[middleware1, middleware2, middleware3],
    tools=[...],
)
```

--------------------------------

### Example Output of Langchain Agent Streaming Tool Call Chunks

Source: https://docs.langchain.com/oss/python/langchain/streaming

This shell output provides an example of the kind of data streamed by a Langchain agent, focusing on tool call chunks. It illustrates how intermediate parsing of tool calls might appear during the streaming process.

```shell
[{'name': 'get_weather', 'args': '', 'id': 'call_je6LWgxYzuZ84mmoDalTYMJC', 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '{"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'city', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '":"', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': 'Boston', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
[{'name': None, 'args': '"}', 'id': None, 'index': 0, 'type': 'tool_call_chunk'}]
```

--------------------------------

### Implement Class-Based Middleware for LangChain Agents in Python

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

This snippet illustrates creating class-based middleware in LangChain agents by extending `AgentMiddleware`. It defines `before_model` and `after_model` methods within the class for comprehensive logging around model calls. An instance of the middleware class is then provided to the `create_agent` function.

```python
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime
from typing import Any, Callable

class LoggingMiddleware(AgentMiddleware):
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"About to call model with {len(state['messages'])} messages")
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None

agent = create_agent(
    model="gpt-4o",
    middleware=[LoggingMiddleware()],
    tools=[...],
)
```

--------------------------------

### Trace Selectively with LangSmith Context Manager

Source: https://docs.langchain.com/oss/python/langchain/observability

Use LangSmith's tracing_context context manager to enable or disable tracing for specific agent invocations. This allows fine-grained control over which operations are traced without modifying global environment settings.

```python
import langsmith as ls

# This WILL be traced
with ls.tracing_context(enabled=True):
    agent.invoke({"messages": [{"role": "user", "content": "Send a test email to alice@example.com"}]})

# This will NOT be traced (if LANGSMITH_TRACING is not set)
agent.invoke({"messages": [{"role": "user", "content": "Send another email"}]})
```

--------------------------------

### Load Skill Tool - Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Tool function that searches through available skills and returns the full skill content or an error message. This function is registered as a tool that the agent can call to retrieve detailed information about specific skills when needed.

```python
if skill["name"] == skill_name:
    return f"Loaded skill: {skill_name}\n\n{skill['content']}"

# Skill not found
available = ", ".join(s["name"] for s in SKILLS)
return f"Skill '{skill_name}' not found. Available skills: {available}"
```

--------------------------------

### Set Dynamic LangSmith Project Name with Context Manager

Source: https://docs.langchain.com/oss/python/langchain/observability

Specify a custom project name programmatically for specific agent invocations using the tracing_context context manager. This allows different operations to log traces to different projects dynamically.

```python
import langsmith as ls

with ls.tracing_context(project_name="email-agent-test", enabled=True):
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Send a welcome email"}]
    })
```

--------------------------------

### Configure LangSmith Tracing for Agent Observability

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Sets up environment variables to enable LangSmith tracing and authenticate with the LangSmith API. This allows for detailed inspection and debugging of agent execution flows. Users can configure this either via bash environment variables or programmatically within Python using the `os` module and `getpass`.

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

```python
import getpass
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
```

--------------------------------

### Customize Subagent Output with LangChain Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

This Python snippet illustrates how to format and enrich the subagent's response before returning it to the main agent. It uses `Command` from `langgraph.types` to update the main agent's state with additional keys (like `example_state_key`) and ensures the final message content, along with the `tool_call_id`, is properly returned, addressing common issues where subagents fail to include results in their final output.

```python
from typing import Annotated
from langchain.agents import AgentState
from langchain.tools import InjectedToolCallId
from langgraph.types import Command


@tool(
    "subagent1_name",
    description="subagent1_description"
)
def call_subagent1(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    result = subagent1.invoke({
        "messages": [{"role": "user", "content": query}]
    })
    return Command(update={
        # Pass back additional state from the subagent
        "example_state_key": result["example_state_key"],
        "messages": [
            ToolMessage(
                content=result["messages"][-1].content,
                tool_call_id=tool_call_id
            )
        ]
    })
```

--------------------------------

### Apply Multiple LangChain Tool Call Limit Middlewares with Diverse Behaviors in Python

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This Python example shows how to configure a LangChain agent with multiple `ToolCallLimitMiddleware` instances, each with different settings. It includes a global limit, tool-specific limits (for 'search' and 'query_database'), and a strict limit for 'scrape_webpage' that raises an error upon exceeding. This allows fine-grained control over tool usage and agent termination behavior.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware


global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)
search_limiter = ToolCallLimitMiddleware(tool_name="search", thread_limit=5, run_limit=3)
database_limiter = ToolCallLimitMiddleware(tool_name="query_database", thread_limit=10)
strict_limiter = ToolCallLimitMiddleware(tool_name="scrape_webpage", run_limit=2, exit_behavior="error")

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool, scraper_tool],
    middleware=[global_limiter, search_limiter, database_limiter, strict_limiter],
)
```

--------------------------------

### Load Tools from MCP Servers with MultiServerMCPClient

Source: https://docs.langchain.com/oss/python/langchain/mcp

Retrieve tools from configured MCP servers and integrate them into a LangChain agent. The get_tools method converts MCP tools into LangChain-compatible tools for use in agents and workflows.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

client = MultiServerMCPClient({...})
tools = await client.get_tools()
agent = create_agent("claude-sonnet-4-5-20250929", tools)
```

--------------------------------

### Initialize Isaacus Embeddings with LangChain

Source: https://docs.langchain.com/oss/python/langchain/rag

This snippet covers installing `langchain-isaacus` and initializing `IsaacusEmbeddings` using an API key and a specific model. It requires setting the `ISAACUS_API_KEY` environment variable, which will be prompted if not already set.

```shell
pip install -qU langchain-isaacus
```

```python
import getpass
import os

if not os.environ.get("ISAACUS_API_KEY"):
    os.environ["ISAACUS_API_KEY"] = getpass.getpass("Enter API key for Isaacus: ")

from langchain_isaacus import IsaacusEmbeddings

embeddings = IsaacusEmbeddings(model="kanon-2-embedder")
```

--------------------------------

### Initialize FAISS Vector Store with LangChain

Source: https://docs.langchain.com/oss/python/langchain/rag

This snippet demonstrates how to install `langchain-community` and `faiss-cpu` to create a `FAISS` vector store. It utilizes an `IndexFlatL2` for efficient similarity search and an `InMemoryDocstore` for managing documents, requiring an embedding function.

```shell
pip install -qU langchain-community faiss-cpu
```

```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
```

--------------------------------

### Custom Error Handler Function with ToolStrategy

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Implement a custom error handler function that returns different messages based on exception type. Demonstrates handling StructuredOutputValidationError and MultipleStructuredOutputsError with specific messages, plus a generic fallback for other errors.

```python
from langchain.agents.structured_output import StructuredOutputValidationError
from langchain.agents.structured_output import MultipleStructuredOutputsError

def custom_error_handler(error: Exception) -> str:
    if isinstance(error, StructuredOutputValidationError):
        return "There was an issue with the format. Try again."
    elif isinstance(error, MultipleStructuredOutputsError):
        return "Multiple structured outputs were returned. Pick the most relevant one."
    else:
        return f"Error: {str(error)}"


agent = create_agent(
    model="gpt-5",
    tools=[],
    response_format=ToolStrategy(
                        schema=Union[ContactInfo, EventDetails],
                        handle_errors=custom_error_handler
                    )  # Default: handle_errors=True
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract info: John Doe (john@email.com) is organizing Tech Conference on March 15th"}]
})

for msg in result['messages']:
    # If message is actually a ToolMessage object (not a dict), check its class name
    if type(msg).__name__ == "ToolMessage":
        print(msg.content)
    # If message is a dictionary or you want a fallback
    elif isinstance(msg, dict) and msg.get('tool_call_id'):
        print(msg['content'])
```

--------------------------------

### Define Custom Agent State and Middleware with Decorators (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

This Python snippet demonstrates how to extend an `AgentState` with custom properties and define middleware using `@before_model` and `@after_model` decorators. It shows how to track `model_call_count` to implement a call limit and increment a counter after each model invocation, enabling persistent state management and conditional logic within a LangChain agent's lifecycle.

```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.agents.middleware import AgentState, before_model, after_model
from typing_extensions import NotRequired
from typing import Any
from langgraph.runtime import Runtime


class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]


@before_model(state_schema=CustomState, can_jump_to=["end"])
def check_call_limit(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
    count = state.get("model_call_count", 0)
    if count > 10:
        return {"jump_to": "end"}
    return None


@after_model(state_schema=CustomState)
def increment_counter(state: CustomState, runtime: Runtime) -> dict[str, Any] | None:
    return {"model_call_count": state.get("model_call_count", 0) + 1}


agent = create_agent(
    model="gpt-4o",
    middleware=[check_call_limit, increment_counter],
    tools=[],
)

# Invoke with custom state
result = agent.invoke({
    "messages": [HumanMessage("Hello")],
    "model_call_count": 0,
    "user_id": "user-123",
})
```

--------------------------------

### Add Documents to Vector Store with Embeddings Python

Source: https://docs.langchain.com/oss/python/langchain/rag

Stores all document splits in a vector store using embeddings in a single command. The function returns a list of document IDs that can be used for tracking and managing stored documents. This enables semantic search capabilities over the indexed documents at runtime.

```python
document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])
```

--------------------------------

### Implement Custom Agent State and Middleware using a Class (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

This Python code illustrates the class-based approach for custom agent state and middleware implementation. It defines a `CustomState` for tracking properties like `model_call_count` and `user_id`, and a `CallCounterMiddleware` class that extends `AgentMiddleware` to manage these state changes with `before_model` and `after_model` methods, providing an alternative structure for complex middleware logic in LangChain.

```python
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langchain.agents.middleware import AgentState, AgentMiddleware
from typing_extensions import NotRequired
from typing import Any


class CustomState(AgentState):
    model_call_count: NotRequired[int]
    user_id: NotRequired[str]


class CallCounterMiddleware(AgentMiddleware[CustomState]):
    state_schema = CustomState

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        count = state.get("model_call_count", 0)
        if count > 10:
            return {"jump_to": "end"}
        return None

    def after_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        return {"model_call_count": state.get("model_call_count", 0) + 1}


agent = create_agent(
    model="gpt-4o",
    middleware=[CallCounterMiddleware()],
    tools=[],
)

# Invoke with custom state
result = agent.invoke({
    "messages": [HumanMessage("Hello")],
    "model_call_count": 0,
    "user_id": "user-123",
})
```

--------------------------------

### Wrap Sub-Agents as Tools in LangChain

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Create tool wrappers for sub-agents (schedule_event and manage_email) that the supervisor can invoke. This architectural pattern abstracts low-level tools into high-level domain-specific tools. Each wrapper includes clear docstrings describing functionality, inputs, and use cases, enabling the supervisor to make informed routing decisions.

```python
@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text
```

--------------------------------

### Load and Index Web Content with RAG Pipeline in Python

Source: https://docs.langchain.com/oss/python/langchain/rag

Creates a complete RAG indexing pipeline that loads web content using WebBaseLoader, splits it into chunks using RecursiveCharacterTextSplitter, and indexes the chunks into a vector store. Demonstrates data ingestion and preprocessing for RAG applications.

```python
import bs4
from langchain.agents import AgentState, create_agent
from langchain_community.document_loaders import WebBaseLoader
from langchain.messages import MessageLikeRepresentation
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)
```

--------------------------------

### Create Trajectory Match Evaluator - Unordered Mode

Source: https://docs.langchain.com/oss/python/langchain/test

Demonstrates creating a trajectory match evaluator in unordered mode to verify tool calls occur regardless of sequence. Suitable for scenarios where multiple information sources must be checked but the order is irrelevant.

```python
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from agentevals.trajectory.match import create_trajectory_match_evaluator


@tool
def get_weather(city: str):
    """Get weather information for a city."""
    return f"It's 75 degrees and sunny in {city}."

@tool
def get_events(city: str):
    """Get events happening in a city."""
    return f"Concert at the park in {city} tonight."

agent = create_agent("gpt-4o", tools=[get_weather, get_events])

evaluator = create_trajectory_match_evaluator(
    trajectory_match_mode="unordered",
)

def test_multiple_tools_any_order():
    result = agent.invoke({
        "messages": [HumanMessage(content="What's happening in SF today?")]
    })

    reference_trajectory = [
        HumanMessage(content="What's happening in SF today?"),
        AIMessage(content="", tool_calls=[
            {"id": "call_1", "name": "get_events", "args": {"city": "SF"}},
            {"id": "call_2", "name": "get_weather", "args": {"city": "SF"}},
        ]),
        ToolMessage(content="Concert at the park in SF tonight.", tool_call_id="call_1"),
        ToolMessage(content="It's 75 degrees and sunny in SF.", tool_call_id="call_2"),
        AIMessage(content="Today in SF: 75 degrees and sunny with a concert at the park tonight."),
    ]

    evaluation = evaluator(
        outputs=result["messages"],
        reference_outputs=reference_trajectory,
    )
```

--------------------------------

### Create Skill Loading Tool in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Defines a LangChain tool decorator function that loads the full content of a skill into the agent's context. The function searches through the SKILLS list to find and return the requested skill's detailed information including schemas and business logic.

```python
from langchain.tools import tool

@tool
def load_skill(skill_name: str) -> str:
    """Load the full content of a skill into the agent's context.

    Use this when you need detailed information about how to handle a specific
    type of request. This will provide you with comprehensive instructions,
    policies, and guidelines for the skill area.

    Args:
        skill_name: The name of the skill to load (e.g., "sales_analytics", "inventory_management")
    """
    # Find and return the requested skill
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return skill["content"]
    return f"Skill '{skill_name}' not found."

```

--------------------------------

### Define Subagent Input Handling in LangChain Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents

This Python example demonstrates how to customize the input context for a subagent using `AgentState` and `ToolRuntime`. It shows how to pull full message history, prior results, or other task metadata from the agent's state, apply transformation logic, and pass specific state keys between the main and subagent's schemas to ensure optimized execution.

```python
from langchain.agents import AgentState
from langchain.tools import tool, ToolRuntime

class CustomState(AgentState):
    example_state_key: str

@tool(
    "subagent1_name",
    description="subagent1_description"
)
def call_subagent1(query: str, runtime: ToolRuntime[None, CustomState]):
    # Apply any logic needed to transform the messages into a suitable input
    subagent_input = some_logic(query, runtime.state["messages"])
    result = subagent1.invoke({
        "messages": subagent_input,
        # You could also pass other state keys here as needed.
        # Make sure to define these in both the main and subagent's
        # state schemas.
        "example_state_key": runtime.state["example_state_key"]
    })
    return result["messages"][-1].content
```

--------------------------------

### Define LangChain Tools for Calendar and Email Actions (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

This Python code defines three LangChain tools: `create_calendar_event`, `send_email`, and `get_available_time_slots`. These functions are decorated with `@tool` and serve as stubs that would, in a real application, interact with external APIs like Google Calendar or SendGrid, providing structured inputs and returning string outputs for agent consumption.

```python
from langchain.tools import tool

@tool
def create_calendar_event(
    title: str,
    start_time: str,       # ISO format: "2024-01-15T14:00:00"
    end_time: str,         # ISO format: "2024-01-15T15:00:00"
    attendees: list[str],  # email addresses
    location: str = ""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    # Stub: In practice, this would call Google Calendar API, Outlook API, etc.
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"


@tool
def send_email(
    to: list[str],  # email addresses
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send an email via email API. Requires properly formatted addresses."""
    # Stub: In practice, this would call SendGrid, Gmail API, etc.
    return f"Email sent to {', '.join(to)} - Subject: {subject}"


@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,  # ISO format: "2024-01-15"
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    # Stub: In practice, this would query calendar APIs
    return ["09:00", "14:00", "16:00"]
```

--------------------------------

### Define Skill TypedDict Structure

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Creates a TypedDict class to define the structure of skills with name, description, and content fields. The description is shown upfront to the agent while content is loaded on-demand for efficiency.

```python
from typing import TypedDict

class Skill(TypedDict):  # [!code highlight]
    """A skill that can be progressively disclosed to the agent."""
    name: str  # Unique identifier for the skill
    description: str  # 1-2 sentence description to show in system prompt
    content: str  # Full skill content with detailed instructions
```

--------------------------------

### Access short-term memory in tool using ToolRuntime - Python

Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

Demonstrates how to access agent state within a tool using the ToolRuntime parameter, which is hidden from the model's view. Creates a custom AgentState with a user_id field and implements a tool that retrieves user information based on state data. The tool_runtime parameter allows direct access to the agent's state dictionary.

```python
from langchain.agents import create_agent, AgentState
from langchain.tools import tool, ToolRuntime


class CustomState(AgentState):
    user_id: str

@tool
def get_user_info(
    runtime: ToolRuntime
) -> str:
    """Look up user info."""
    user_id = runtime.state["user_id"]
    return "User is John Smith" if user_id == "user_123" else "Unknown user"

agent = create_agent(
    model="gpt-5-nano",
    tools=[get_user_info],
    state_schema=CustomState,
)

result = agent.invoke({
    "messages": "look up user information",
    "user_id": "user_123"
})
print(result["messages"][-1].content)
# > User is John Smith.
```

--------------------------------

### Trim LangChain Agent Message History with Middleware (Python)

Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

This Python code demonstrates how to implement a message trimming strategy for a LangChain agent using the `@before_model` middleware decorator. The `trim_messages` function keeps only the first message and the last 3 or 4 messages, effectively truncating the conversation history to fit within an LLM's context window. It utilizes `RemoveMessage` to clear previous messages and then re-adds the selected ones. The example shows how to integrate this middleware when creating an agent and invoke it with a series of messages.

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain_core.runnables import RunnableConfig
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }

agent = create_agent(
    your_model_here,
    tools=your_tools_here,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob. You told me that earlier.
If you'd like me to call you a nickname or use a different name, just say the word.
"""
```

--------------------------------

### Define LangChain Tool to Record Warranty Status and Update State in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

This Python function defines a LangChain tool named `record_warranty_status`. It takes a status string and runtime, records the warranty status, and updates the agent's state to transition to the 'specialist' step, indicating a change in the agent's operational mode. The `ToolMessage` confirms the action.

```python
from langchain.tools import ToolRuntime, tool
from langchain.messages import ToolMessage
from langgraph.types import Command

@tool
def record_warranty_status(
    status: str,
    runtime: ToolRuntime[None, SupportState]
) -> Command:
    """Record warranty status and transition to next step."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded: {status}",
                    tool_call_id=runtime.tool_call_id
                )
            ],
            "warranty_status": status,
            "current_step": "specialist"  # Update state to trigger transition
        }
    )
```

--------------------------------

### Define ToolStrategy Class Structure

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Generic ToolStrategy class definition with schema, tool_message_content, and handle_errors properties. Defines the core configuration for structured output generation supporting multiple error handling strategies and schema types.

```python
class ToolStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    tool_message_content: str | None
    handle_errors: Union[
        bool,
        str,
        type[Exception],
        tuple[type[Exception], ...],
        Callable[[Exception], str],
    ]
```

--------------------------------

### Load web content with WebBaseLoader and BeautifulSoup

Source: https://docs.langchain.com/oss/python/langchain/rag

Load HTML content from a web URL and parse it selectively by filtering for specific HTML tags. Uses WebBaseLoader with BeautifulSoup's SoupStrainer to extract only relevant content (post title, headers, and content). Returns a list of Document objects containing the parsed text.

```python
import bs4
from langchain_community.document_loaders import WebBaseLoader

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")
```

--------------------------------

### Implement Safety Guardrail with Decorator Syntax

Source: https://docs.langchain.com/oss/python/langchain/guardrails

Implements the same safety guardrail functionality using the @after_agent decorator instead of class inheritance. This approach is more concise and functional, initializing the safety model at module level and decorating a standalone function. The decorator approach provides the same can_jump_to configuration for agent state control.

```python
from langchain.agents.middleware import after_agent, AgentState, hook_config
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from langchain.chat_models import init_chat_model
from typing import Any

safety_model = init_chat_model("gpt-4o-mini")

@after_agent(can_jump_to=["end"])
def safety_guardrail(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Model-based guardrail: Use an LLM to evaluate response safety."""
    # Get the final AI response
    if not state["messages"]:
        return None

    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage):
        return None

    # Use a model to evaluate safety
    safety_prompt = f"""Evaluate if this response is safe and appropriate.
    Respond with only 'SAFE' or 'UNSAFE'.

    Response: {last_message.content}"""

    result = safety_model.invoke([{"role": "user", "content": safety_prompt}])

    if "UNSAFE" in result.content:
        last_message.content = "I cannot provide that response. Please rephrase your request."

    return None

# Use the safety guardrail
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[safety_guardrail],
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I make explosives?"}]
})
```

--------------------------------

### Install and Configure Azure OpenAI Embeddings

Source: https://docs.langchain.com/oss/python/langchain/rag

Install langchain-openai and set up Azure OpenAI embeddings with endpoint, deployment, and API version configuration. Requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME, and AZURE_OPENAI_API_VERSION environment variables.

```shell
pip install -U "langchain-openai"
```

```python
import getpass
import os

if not os.environ.get("AZURE_OPENAI_API_KEY"):
    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Enter API key for Azure: ")

from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)
```

--------------------------------

### Invoke Multi-Agent Graph with User Input

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

Invokes the compiled StateGraph with an initial user message and iterates through the result messages to display them. Passes a message list containing a user query about account login issues to initiate the agent workflow.

```python
result = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hi, I'm having trouble with my account login. Can you help?",
            }
        ]
    }
)

for msg in result["messages"]:
    msg.pretty_print()
```

--------------------------------

### Vector Store Document IDs Output Python

Source: https://docs.langchain.com/oss/python/langchain/rag

Shows the expected output format of document IDs returned by the vector_store.add_documents() method. Each ID is a unique identifier (UUID format) for the stored document splits, allowing for document tracking and retrieval management.

```python
['07c18af6-ad58-479a-bfb1-d508033f9c64', '9000bf8e-1993-446f-8d4d-f4e507ba4b8f', 'ba3b5d14-bed9-4f5f-88be-44c88aedc2e6']
```

--------------------------------

### Handoffs workflow sequence diagram

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

Illustrates a multi-step conversational flow using the handoffs pattern, showing user and agent interactions, and state updates. The agent dynamically changes its behavior based on the current step and tools used to progress through a support interaction.

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Workflow State

    User->>Agent: "My phone is broken"
    Note over Agent,Workflow State: Step: Get warranty status<br/>Tools: record_warranty_status
    Agent-->>User: "Is your device under warranty?"

    User->>Agent: "Yes, it's still under warranty"
    Agent->>Workflow State: record_warranty_status("in_warranty")
    Note over Agent,Workflow State: Step: Classify issue<br/>Tools: record_issue_type
    Agent-->>User: "Can you describe the issue?"

    User->>Agent: "The screen is cracked"
    Agent->>Workflow State: record_issue_type("hardware")
    Note over Agent,Workflow State: Step: Provide resolution<br/>Tools: provide_solution, escalate_to_human
    Agent-->>User: "Here's the warranty repair process..."
```

--------------------------------

### Install LangChain dependencies with uv

Source: https://docs.langchain.com/oss/python/langchain/rag

Install LangChain core package along with text-splitters, community integrations, and BeautifulSoup4 using uv package manager. Alternative to pip for faster dependency resolution.

```bash
uv add langchain langchain-text-splitters langchain-community bs4
```

--------------------------------

### Interact with persistent memory (store) using ToolRuntime in Python

Source: https://docs.langchain.com/oss/python/langchain/tools

These Python functions demonstrate how to access and update an agent's persistent memory (store) using `runtime.store`. The `get_user_info` tool retrieves data for a given user ID, and `save_user_info` stores new user data within the store. The example uses an `InMemoryStore` for demonstration purposes, showcasing how an agent can remember and retrieve user-specific information across different interactions.

```python
from typing import Any
from langgraph.store.memory import InMemoryStore
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

# First session: save user info
agent.invoke({
    "messages": [{"role": "user", "content": "Save the following user: userid: abc123, name: Foo, age: 25, email: foo@langchain.dev"}]
})

# Second session: get user info
agent.invoke({
    "messages": [{"role": "user", "content": "Get user info for user with id 'abc123'"}]
})
# Here is the user info for user with ID "abc123":
# - Name: Foo
# - Age: 25
# - Email: foo@langchain.dev
```

--------------------------------

### Initialize In-memory Vector Store with LangChain

Source: https://docs.langchain.com/oss/python/langchain/rag

This snippet demonstrates how to install `langchain-core` and set up an `InMemoryVectorStore`. This vector store is suitable for local development or small-scale applications where data persistence is not required across sessions, using an existing embeddings object.

```shell
pip install -U "langchain-core"
```

```python
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
```

--------------------------------

### Initialize Chroma Vector Store with LangChain

Source: https://docs.langchain.com/oss/python/langchain/rag

This snippet covers installing `langchain-chroma` and setting up a `Chroma` vector store. It allows specifying a collection name, an embedding function, and an optional `persist_directory` for local data storage, enabling persistence across sessions.

```shell
pip install -qU langchain-chroma
```

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
```

--------------------------------

### Inject File Context from State into LLM Prompt (LangChain Python)

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

This middleware function injects context about uploaded files from the agent's `State` into the LLM prompt. It retrieves file metadata, formats it, and prepends it to the existing messages, ensuring the LLM has relevant document information for the current query.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def inject_file_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject context about files user has uploaded this session."""
    # Read from State: get uploaded files metadata
    uploaded_files = request.state.get("uploaded_files", [])

    if uploaded_files:
        # Build context about available files
        file_descriptions = []
        for file in uploaded_files:
            file_descriptions.append(
                f"- {file['name']} ({file['type']}): {file['summary']}"
            )

        file_context = f"""Files you have access to in this conversation:
{chr(10).join(file_descriptions)}

Reference these files when answering questions."""

        # Inject file context before recent messages
        messages = [
            *request.messages,
            {"role": "user", "content": file_context},
        ]
        request = request.override(messages=messages)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[inject_file_context]
)
```

--------------------------------

### Configure pytest VCR Marker in pyproject.toml

Source: https://docs.langchain.com/oss/python/langchain/test

Alternative TOML-based configuration for registering the vcr marker and setting record mode, providing the same functionality as pytest.ini in a modern Python project configuration format.

```toml
[tool.pytest.ini_options]
markers = [
  "vcr: record/replay HTTP via VCR"
]
addopts = "--record-mode=once"
```

--------------------------------

### Initialize PGVector (PostgreSQL) Vector Store with LangChain

Source: https://docs.langchain.com/oss/python/langchain/rag

This snippet covers installing `langchain-postgres` and setting up a `PGVector` store to connect to a PostgreSQL database. It uses an embedding function, specifies a collection name, and requires a valid PostgreSQL connection string for database access.

```shell
pip install -qU langchain-postgres
```

```python
from langchain_postgres import PGVector

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://...",
)
```

--------------------------------

### Configure ModelRetryMiddleware

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Configures the ModelRetryMiddleware for an agent to automatically retry failed model calls based on specified parameters like max retries, backoff strategy, and error handling.

```APIDOC
## CONFIGURE langchain.agents.middleware.ModelRetryMiddleware

### Description
Configures the ModelRetryMiddleware to enhance agent reliability by automatically retrying failed model calls. This middleware is crucial for handling transient errors, improving network-dependent model request reliability, and building resilient agents.

### Method
CONFIGURE

### Endpoint
langchain.agents.middleware.ModelRetryMiddleware

### Parameters
#### Path Parameters
_None_

#### Query Parameters
_None_

#### Request Body
- **max_retries** (number) - Optional - Default: `2` - Maximum number of retry attempts after the initial call (3 total attempts with default).
- **retry_on** (tuple[type[Exception], ...] | callable) - Optional - Default: `(Exception,)` - Either a tuple of exception types to retry on, or a callable that takes an exception and returns `True` if it should be retried.
- **on_failure** (string | callable) - Optional - Default: `'continue'` - Behavior when all retries are exhausted. Options: `'continue'` (default) - Return an `AIMessage` with error details; `'error'` - Re-raise the exception; Custom callable - Function that takes the exception and returns a string for the `AIMessage` content.
- **backoff_factor** (number) - Optional - Default: `2.0` - Multiplier for exponential backoff. Each retry waits `initial_delay * (backoff_factor ** retry_number)` seconds. Set to `0.0` for constant delay.
- **initial_delay** (number) - Optional - Default: `1.0` - Initial delay in seconds before first retry.
- **max_delay** (number) - Optional - Default: `60.0` - Maximum delay in seconds between retries (caps exponential backoff growth).
- **jitter** (boolean) - Optional - Default: `true` - Whether to add random jitter (±25%) to delay to avoid thundering herd.

### Request Example
```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        ),
    ],
)
```

### Response
#### Success Response (200)
Returns an instance of `ModelRetryMiddleware` configured with the specified parameters, ready to be used within a LangChain agent.

#### Response Example
```python
# An instance of ModelRetryMiddleware is created and integrated
# into the agent's middleware stack.
# The 'response' is the successfully configured agent.
agent = create_agent(
    model="gpt-4o",
    tools=[search_tool],
    middleware=[
        ModelRetryMiddleware(
            max_retries=3,
            backoff_factor=2.0,
            initial_delay=1.0,
        )
    ]
)
```
```

--------------------------------

### Python LangChain Tool with Additional Search Parameters

Source: https://docs.langchain.com/oss/python/langchain/rag

This Python snippet illustrates how to add specific, type-hinted arguments to a LangChain tool function, such as a `section` parameter with `Literal` types. This allows developers to guide the LLM to specify additional search parameters beyond a simple query, enabling more granular control over tool execution.

```python
from typing import Literal

def retrieve_context(query: str, section: Literal["beginning", "middle", "end"]):
```

--------------------------------

### Install LangChain dependencies with pip

Source: https://docs.langchain.com/oss/python/langchain/rag

Install LangChain core package along with text-splitters, community integrations, and BeautifulSoup4 using pip package manager. Required for basic LangChain functionality and web scraping capabilities.

```bash
pip install langchain langchain-text-splitters langchain-community bs4
```

--------------------------------

### Install LangChain Package

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Install the langchain package using either pip or conda package managers. This is the core dependency required for building the multi-agent supervisor system.

```bash
pip install langchain
```

```bash
conda install langchain -c conda-forge
```

--------------------------------

### ShellToolMiddleware Constructor

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Initializes the ShellToolMiddleware, configuring the persistent shell session for an agent, including its workspace, execution policy, and command handling.

```APIDOC
## Class: ShellToolMiddleware

### Description
Exposes a persistent shell session to agents for command execution, facilitating system commands, automation, and file operations. It allows configuration of the shell environment, execution policies, and command sanitization.

### Class Reference
`langchain.agents.middleware.ShellToolMiddleware`

### Constructor Parameters
- **workspace_root** (str | Path | None) - Optional - Base directory for the shell session. If omitted, a temporary directory is created when the agent starts and removed when it ends.
- **startup_commands** (tuple[str, ...] | list[str] | str | None) - Optional - Optional commands executed sequentially after the session starts.
- **shutdown_commands** (tuple[str, ...] | list[str] | str | None) - Optional - Optional commands executed before the session shuts down.
- **execution_policy** (BaseExecutionPolicy | None) - Optional - Execution policy controlling timeouts, output limits, and resource configuration. Options include `HostExecutionPolicy`, `DockerExecutionPolicy`, and `CodexSandboxExecutionPolicy`.
- **redaction_rules** (tuple[RedactionRule, ...] | list[RedactionRule] | None) - Optional - Optional redaction rules to sanitize command output before returning it to the model.
- **tool_description** (str | None) - Optional - Optional override for the registered shell tool description.
- **shell_command** (Sequence[str] | str | None) - Optional - Optional shell executable (string) or argument sequence used to launch the persistent session. Defaults to `/bin/bash`.
- **env** (Mapping[str, Any] | None) - Optional - Optional environment variables to supply to the shell session. Values are coerced to strings before command execution.

### Constructor Example (Python)
```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ShellToolMiddleware,
    HostExecutionPolicy,
    DockerExecutionPolicy,
    RedactionRule,
)

# Basic shell tool with host execution
agent = create_agent(
    model="gpt-4o",
    tools=[search_tool],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            execution_policy=HostExecutionPolicy(),
        ),
    ],
)

# Docker isolation with startup commands
agent_docker = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            startup_commands=["pip install requests", "export PYTHONPATH=/workspace"],
            execution_policy=DockerExecutionPolicy(
                image="python:3.11-slim",
                command_timeout=60.0,
            ),
        ),
    ],
)

# With output redaction
agent_redacted = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            redaction_rules=[
                RedactionRule(pii_type="api_key", detector=r"sk-[a-zA-Z0-9]{32}"),
            ],
        ),
    ],
)
```

### Notes
- **Security consideration**: Use appropriate execution policies (`HostExecutionPolicy`, `DockerExecutionPolicy`, or `CodexSandboxExecutionPolicy`) to match your deployment's security requirements.
- **Limitation**: Persistent shell sessions do not currently work with interrupts (human-in-the-loop).
```

--------------------------------

### Configure LangChain Model Retry Middleware to Continue on Failure (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Explains how to set `on_failure='continue'` in `ModelRetryMiddleware` to return an `AIMessage` with error details instead of re-raising an exception when all retries are exhausted. This allows the agent to gracefully handle failures and potentially recover or inform the user.

```python
retry_continue = ModelRetryMiddleware(
    max_retries=4,
    on_failure="continue",  # Return AIMessage with error instead of raising
)
```

--------------------------------

### Stream Tool Calls with Messages and Updates in Python

Source: https://docs.langchain.com/oss/python/langchain/streaming

Demonstrates streaming tool calls from a LangChain agent using stream_mode=['messages', 'updates']. The code defines a get_weather tool, creates an agent with OpenAI's GPT-5.2 model, and processes both incremental message chunks and completed messages. It renders partial tool call JSON chunks as they are generated and displays final parsed tool calls with their results.

```python
from typing import Any

from langchain.agents import create_agent
from langchain.messages import AIMessage, AIMessageChunk, AnyMessage, ToolMessage


def get_weather(city: str) -> str:
    """Get weather for a given city."""

    return f"It's always sunny in {city}!"


agent = create_agent("openai:gpt-5.2", tools=[get_weather])


def _render_message_chunk(token: AIMessageChunk) -> None:
    if token.text:
        print(token.text, end="|")
    if token.tool_call_chunks:
        print(token.tool_call_chunks)
    # N.B. all content is available through token.content_blocks


def _render_completed_message(message: AnyMessage) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        print(f"Tool calls: {message.tool_calls}")
    if isinstance(message, ToolMessage):
        print(f"Tool response: {message.content_blocks}")


input_message = {"role": "user", "content": "What is the weather in Boston?"}
for stream_mode, data in agent.stream(
    {"messages": [input_message]},
    stream_mode=["messages", "updates"],  # [!code highlight]
):
    if stream_mode == "messages":
        token, metadata = data
        if isinstance(token, AIMessageChunk):
            _render_message_chunk(token)  # [!code highlight]
    if stream_mode == "updates":
        for source, update in data.items():
            if source in ("model", "tools"):  # `source` captures node name
                _render_completed_message(update["messages"][-1])  # [!code highlight]
```

--------------------------------

### Initialize Router LLM for Multi-Agent Workflow

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

Create a router language model using init_chat_model with GPT-4o-mini for cost-effective query classification and routing decisions. This model will determine which agents to invoke.

```python
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

router_llm = init_chat_model("openai:gpt-4o-mini")
```

--------------------------------

### Approve Tool Call Decision - LangGraph Python

Source: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

Approve a tool call as-is and execute it without modifications. Decisions are provided as a list matching the order of actions in the interrupt request. This decision type allows the agent to proceed with the original tool execution.

```python
agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "approve",
                }
            ]
        }
    ),
    config=config
)
```

--------------------------------

### Write long-term memory from LangChain agent tools

Source: https://docs.langchain.com/oss/python/langchain/long-term-memory

Implements a tool that updates and saves user information to the long-term memory store. Demonstrates using TypedDict for structured data, accessing runtime context, and persisting agent-modified data back to the store.

```python
from dataclasses import dataclass
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.store.memory import InMemoryStore


# InMemoryStore saves data to an in-memory dictionary. Use a DB-backed store in production.
store = InMemoryStore()

@dataclass
class Context:
    user_id: str

# TypedDict defines the structure of user information for the LLM
class UserInfo(TypedDict):
    name: str

# Tool that allows agent to update user information (useful for chat applications)
@tool
def save_user_info(user_info: UserInfo, runtime: ToolRuntime[Context]) -> str:
    """Save user info."""
    # Access the store - same as that provided to `create_agent`
    store = runtime.store
    user_id = runtime.context.user_id
    # Store data in the store (namespace, key, data)
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[save_user_info],
    store=store,
    context_schema=Context
)
```

--------------------------------

### Define Custom PII Detectors with Regex for Langchain Agents

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This snippet illustrates how to create custom PII detection rules within `PIIMiddleware` using regex patterns. It demonstrates defining a custom detector for 'api_key' using a string regex for blocking, and another for 'phone_number' using a compiled regex for masking, enabling tailored PII handling.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
import re


# Method 1: Regex pattern string
agent1 = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
        ),
    ],
)

# Method 2: Compiled regex pattern
agent2 = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        PIIMiddleware(
            "phone_number",
            detector=re.compile(r"\+?\d{1,3}[\s.-]?\d{3,4}[\s.-]?\d{4}"),
            strategy="mask",
        ),
    ],
)
```

--------------------------------

### Initialize LangChain SQLDatabase for SQLite (Python)

Source: https://docs.langchain.com/oss/python/langchain/sql-agent

This Python code initializes the `SQLDatabase` wrapper from `langchain_community.utilities` by connecting to a local `Chinook.db` SQLite database. It demonstrates how to retrieve and print the database dialect, list available table names, and execute a sample SQL query to fetch data.

```python
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")

print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')
```

--------------------------------

### SummarizationMiddleware Configuration

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Initialize and configure the SummarizationMiddleware for an agent to automatically summarize conversation history based on token limits, message counts, or context size fractions. This middleware preserves recent messages while compressing older context.

```APIDOC
## SummarizationMiddleware

### Description
Automatically summarize conversation history when approaching token limits, preserving recent messages while compressing older context. Useful for long-running conversations that exceed context windows and multi-turn dialogues with extensive history.

### Class
SummarizationMiddleware

### Initialization
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
        ),
    ],
)
```

### Parameters

#### Required Parameters
- **model** (string | BaseChatModel) - Model for generating summaries. Can be a model identifier string (e.g., 'openai:gpt-4o-mini') or a BaseChatModel instance.
- **trigger** (ContextSize | list[ContextSize] | None) - Condition(s) for triggering summarization. Can be a single ContextSize tuple or a list of ContextSize tuples (OR logic). Specify one of: fraction (float, 0-1), tokens (int), or messages (int). At least one condition must be specified.

#### Optional Parameters
- **keep** (ContextSize) - Default: ('messages', 20) - How much context to preserve after summarization. Specify exactly one of: fraction (float, 0-1), tokens (int), or messages (int).
- **token_counter** (function) - Custom token counting function. Defaults to character-based counting.
- **summary_prompt** (string) - Custom prompt template for summarization. Uses built-in template if not specified. Template should include {messages} placeholder.
- **trim_tokens_to_summarize** (number) - Default: 4000 - Maximum number of tokens to include when generating the summary.
- **summary_prefix** (string) - Prefix to add to the summary message. Uses default prefix if not provided.

#### Deprecated Parameters
- **max_tokens_before_summary** (number) - Deprecated: Use trigger: {"tokens": value} instead.
- **messages_to_keep** (number) - Deprecated: Use keep: {"messages": value} instead.

### Trigger Conditions

Control when summarization runs:

- **Single condition**: Specified condition must be met
- **Array of conditions**: Any condition must be met (OR logic)

Condition types:
- `fraction` (float): Fraction of model's context size (0-1)
- `tokens` (int): Absolute token count
- `messages` (int): Message count

### Keep Conditions

Define how much context to preserve after summarization:

- `fraction` (float): Fraction of model's context size to keep (0-1)
- `tokens` (int): Absolute token count to keep
- `messages` (int): Number of recent messages to keep

### Request Example
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[your_weather_tool, your_calculator_tool],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20),
            token_counter=custom_counter_function,
            summary_prompt="Summarize: {messages}",
            trim_tokens_to_summarize=4000,
            summary_prefix="Previous conversation summary:"
        ),
    ],
)
```

### Response

The middleware monitors message token counts and automatically summarizes older messages when thresholds are reached. Recent messages are preserved based on the keep parameter configuration.

### Notes

- The `fraction` conditions for `trigger` and `keep` rely on a chat model's profile data if using langchain>=1.1
- If profile data is not available, use another condition or specify manually using init_chat_model with custom_profile parameter
- Summarization is triggered when any condition in the trigger list is met (OR logic)
- The keep parameter defines exactly how much context remains after summarization
```

--------------------------------

### Invoke Agent with Context for After Model Hook

Source: https://docs.langchain.com/oss/python/langchain/runtime

Call agent.invoke() with message input and context data. The context is passed through the Runtime object to middleware functions, including the after_model hook, enabling access to user-specific information during post-execution logging.

```python
agent.invoke(
    {"messages": [{"role": "user", "content": "What's my name?"}]},
    context=Context(user_name="John Smith")
)
```

--------------------------------

### Split documents with RecursiveCharacterTextSplitter

Source: https://docs.langchain.com/oss/python/langchain/rag

Split a large Document into smaller chunks of specified size with overlap using RecursiveCharacterTextSplitter. This recursive splitter uses common separators (newlines, spaces) to maintain context between chunks. Enables efficient storage and retrieval of relevant document sections for language model processing.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")
```

--------------------------------

### Install LangChain Dependencies with pip or uv

Source: https://docs.langchain.com/oss/python/langchain/studio

Install required Python packages (langchain and langchain-openai) from the project root directory. Supports both pip and uv package managers.

```shell
pip install langchain langchain-openai
```

```shell
uv add langchain langchain-openai
```

--------------------------------

### Define Skill TypedDict and Skills List in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Defines a TypedDict structure for skills with name, description, and content properties. Creates a list of skill definitions (sales_analytics and inventory_management) that contain database schemas, business logic rules, and example SQL queries for agent use.

```python
from typing import TypedDict, NotRequired

class Skill(TypedDict):
    """A skill that can be progressively disclosed to the agent."""
    name: str
    description: str
    content: str

SKILLS: list[Skill] = [
    {
        "name": "sales_analytics",
        "description": "Database schema and business logic for sales data analysis including customers, orders, and revenue.",
        "content": """# Sales Analytics Schema

## Tables

### customers
- customer_id (PRIMARY KEY)
- name
- email
- signup_date
- status (active/inactive)
- customer_tier (bronze/silver/gold/platinum)

### orders
- order_id (PRIMARY KEY)
- customer_id (FOREIGN KEY -> customers)
- order_date
- status (pending/completed/cancelled/refunded)
- total_amount
- sales_region (north/south/east/west)

### order_items
- item_id (PRIMARY KEY)
- order_id (FOREIGN KEY -> orders)
- product_id
- quantity
- unit_price
- discount_percent

## Business Logic

**Active customers**: status = 'active' AND signup_date <= CURRENT_DATE - INTERVAL '90 days'

**Revenue calculation**: Only count orders with status = 'completed'. Use total_amount from orders table, which already accounts for discounts.

**Customer lifetime value (CLV)**: Sum of all completed order amounts for a customer.

**High-value orders**: Orders with total_amount > 1000

## Example Query

-- Get top 10 customers by revenue in the last quarter
SELECT
    c.customer_id,
    c.name,
    c.customer_tier,
    SUM(o.total_amount) as total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
  AND o.order_date >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY c.customer_id, c.name, c.customer_tier
ORDER BY total_revenue DESC
LIMIT 10;""",
    },
    {
        "name": "inventory_management",
        "description": "Database schema and business logic for inventory tracking including products, warehouses, and stock levels.",
        "content": """# Inventory Management Schema

## Tables

### products
- product_id (PRIMARY KEY)
- product_name
- sku
- category
- unit_cost
- reorder_point (minimum stock level before reordering)
- discontinued (boolean)

### warehouses
- warehouse_id (PRIMARY KEY)
- warehouse_name
- location
- capacity

### inventory
- inventory_id (PRIMARY KEY)
- product_id (FOREIGN KEY -> products)
- warehouse_id (FOREIGN KEY -> warehouses)
- quantity_on_hand
- last_updated

### stock_movements
- movement_id (PRIMARY KEY)
- product_id (FOREIGN KEY -> products)
- warehouse_id (FOREIGN KEY -> warehouses)
- movement_type (inbound/outbound/transfer/adjustment)
- quantity (positive for inbound, negative for outbound)
- movement_date
- reference_number

## Business Logic

**Available stock**: quantity_on_hand from inventory table where quantity_on_hand > 0

**Products needing reorder**: Products where total quantity_on_hand across all warehouses is less than or equal to the product's reorder_point

**Active products only**: Exclude products where discontinued = true unless specifically analyzing discontinued items

**Stock valuation**: quantity_on_hand * unit_cost for each product

## Example Query

-- Find products below reorder point across all warehouses
SELECT
    p.product_id,
    p.product_name,
    p.reorder_point,
    SUM(i.quantity_on_hand) as total_stock,
    p.unit_cost,
    (p.reorder_point - SUM(i.quantity_on_hand)) as units_to_reorder
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE p.discontinued = false
GROUP BY p.product_id, p.product_name, p.reorder_point, p.unit_cost
HAVING SUM(i.quantity_on_hand) <= p.reorder_point
ORDER BY units_to_reorder DESC;""",
    },
]

```

--------------------------------

### Integrate Custom SSN Detector with LangChain's PIIMiddleware in Python

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This code snippet shows how to instantiate a LangChain agent and configure its middleware to use a custom PII detector. The `PIIMiddleware` is set to detect 'ssn' using the previously defined `detect_ssn` function and to apply a 'hash' strategy for handling detected PII.

```python
agent3 = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        PIIMiddleware(
            "ssn",
            detector=detect_ssn,
            strategy="hash",
        ),
    ],
)
```

--------------------------------

### Define LangChain Tool to Retrieve User Preferences from Store (Python)

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

This Python snippet illustrates creating a LangChain tool, `get_preference`, that reads persisted user preferences from the agent's `runtime.store`. It shows how to access the store using a user ID from the runtime context to retrieve and return specific preference values.

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@tool
def get_preference(
    preference_key: str,
    runtime: ToolRuntime[Context]
) -> str:
    """Get user preference from Store."""
    user_id = runtime.context.user_id

    # Read from Store: get existing preferences
    store = runtime.store
    existing_prefs = store.get(("preferences",), user_id)

    if existing_prefs:
        value = existing_prefs.value.get(preference_key)
        return f"{preference_key}: {value}" if value else f"No preference set for {preference_key}"
    else:
        return "No preferences found"

agent = create_agent(
    model="gpt-4o",
    tools=[get_preference],
    context_schema=Context,
    store=InMemoryStore()
)
```

--------------------------------

### LLMToolEmulator - Custom Model Configuration

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Specify a custom LLM model for generating emulated tool responses. This allows you to use different models for emulation than the agent's primary model.

```APIDOC
## LLMToolEmulator - Custom Model Configuration

### Description
Use a custom LLM model for generating emulated tool responses instead of the agent's default model. This enables you to test with different models or use more cost-effective models for emulation.

### Example: Custom Model for Emulation
```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return "Email sent"

# Use custom model for emulation
agent = create_agent(
    model="gpt-4o",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator(model="claude-sonnet-4-5-20250929")],
)
```

### Model Parameter
- **model** can be a model identifier string (e.g., 'anthropic:claude-sonnet-4-5-20250929')
- **model** can be a BaseChatModel instance
- Defaults to the agent's model if not specified
- See init_chat_model for more information on available model identifiers
```

--------------------------------

### Disable Error Handling in ToolStrategy

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Configure ToolStrategy to disable error handling by setting handle_errors to False. This causes all validation errors to be raised immediately without retry.

```python
response_format = ToolStrategy(
    schema=ProductRating,
    handle_errors=False  # All errors raised
)
```

--------------------------------

### Inject Compliance Rules from Runtime Context into LLM Prompt (LangChain Python)

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

This middleware function injects compliance rules from the agent's `Runtime Context` into the LLM prompt. It uses user jurisdiction, industry, and specified frameworks (e.g., GDPR, HIPAA) to build and append relevant regulatory constraints to the messages, guiding the LLM's response generation.

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@dataclass
class Context:
    user_jurisdiction: str
    industry: str
    compliance_frameworks: list[str]

@wrap_model_call
def inject_compliance_rules(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject compliance constraints from Runtime Context."""
    # Read from Runtime Context: get compliance requirements
    jurisdiction = request.runtime.context.user_jurisdiction
    industry = request.runtime.context.industry
    frameworks = request.runtime.context.compliance_frameworks

    # Build compliance constraints
    rules = []
    if "GDPR" in frameworks:
        rules.append("- Must obtain explicit consent before processing personal data")
        rules.append("- Users have right to data deletion")
    if "HIPAA" in frameworks:
```

--------------------------------

### LLMToolEmulator - Selective Tool Emulation

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Configure the LLMToolEmulator to emulate only specific tools while leaving others to execute normally. This allows partial testing and development of agent behavior.

```APIDOC
## LLMToolEmulator - Selective Tool Emulation

### Description
Emulate only specific tools by name or instance while allowing other tools to execute normally. This is useful when you want to test certain tools with AI-generated responses while keeping others functional.

### Example: Emulate Specific Tools Only
```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return "Email sent"

# Emulate specific tools only
agent = create_agent(
    model="gpt-4o",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator(tools=["get_weather"])],
)
```

### Configuration
- **tools** parameter accepts a list of tool names (strings) or BaseTool instances
- Only the specified tools will use LLM-generated responses
- All other tools will execute normally
```

--------------------------------

### Configure VCR with Sensitive Data Filtering in conftest.py

Source: https://docs.langchain.com/oss/python/langchain/test

Sets up a pytest fixture that configures vcrpy to mask sensitive headers and query parameters in recorded cassettes. This prevents API keys, authorization tokens, and other confidential information from being stored in version control.

```python
import pytest

@pytest.fixture(scope="session")
def vcr_config():
    return {
        "filter_headers": [
            ("authorization", "XXXX"),
            ("x-api-key", "XXXX"),
            # ... other headers you want to mask
        ],
        "filter_query_parameters": [
            ("api_key", "XXXX"),
            ("key", "XXXX"),
        ],
    }
```

--------------------------------

### Initialize IBM watsonx Embeddings with LangChain

Source: https://docs.langchain.com/oss/python/langchain/rag

This snippet demonstrates how to install the `langchain-ibm` package and initialize `WatsonxEmbeddings` using an API key and specific model details. It requires setting the `WATSONX_APIKEY` environment variable for authentication and uses a specified URL and project ID.

```shell
pip install -qU langchain-ibm
```

```python
import getpass
import os

if not os.environ.get("WATSONX_APIKEY"):
    os.environ["WATSONX_APIKEY"] = getpass.getpass("Enter API key for IBM watsonx: ")

from langchain_ibm import WatsonxEmbeddings

embeddings = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="<WATSONX PROJECT_ID>",
)
```

--------------------------------

### ShellToolMiddleware with output redaction for sensitive data

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Configures ShellToolMiddleware with redaction rules to sanitize command output before returning it to the model. Uses pattern matching to detect and redact sensitive information such as API keys. Useful for preventing exposure of credentials or personal information in agent interactions.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import (
    ShellToolMiddleware,
    RedactionRule,
)

agent_redacted = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ShellToolMiddleware(
            workspace_root="/workspace",
            redaction_rules=[
                RedactionRule(pii_type="api_key", detector=r"sk-[a-zA-Z0-9]{32}"),
            ],
        ),
    ],
)
```

--------------------------------

### Define JSON Schema for Product Review Validation

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Defines a JSON schema object that specifies the structure and constraints for product review data validation. Includes fields for rating (1-5 range), sentiment (enum with positive/negative values), and key_points (array of strings). Marks sentiment and key_points as required fields.

```json
{
    "description": "The rating of the product (1-5)",
    "minimum": 1,
    "maximum": 5
}
{
    "type": "string",
    "enum": ["positive", "negative"],
    "description": "The sentiment of the review"
}
{
    "type": "array",
    "items": {"type": "string"},
    "description": "The key points of the review"
}
"required": ["sentiment", "key_points"]
```

--------------------------------

### Configure LangChain Model Retry Middleware with Default Settings (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Illustrates the basic usage of `ModelRetryMiddleware` within a LangChain agent using its default settings. This configuration provides automatic retries for failed model calls with predefined exponential backoff, simplifying error handling for transient issues.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool],
    middleware=[ModelRetryMiddleware()],
)
```

--------------------------------

### ToolCallLimitMiddleware Configuration

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Configure tool call limits globally or for specific tools to prevent excessive API calls and protect against runaway agent loops. Supports both thread-level limits (across conversation turns) and run-level limits (per invocation).

```APIDOC
## ToolCallLimitMiddleware

### Description
Middleware for controlling agent execution by limiting the number of tool calls. Can enforce global limits across all tools or tool-specific limits. Useful for preventing expensive API calls, enforcing rate limits, and protecting against runaway agent loops.

### Class
langchain.agents.middleware.ToolCallLimitMiddleware

### Parameters

#### tool_name
- **Type**: string
- **Required**: No
- **Description**: Name of specific tool to limit. If not provided, limits apply to all tools globally.

#### thread_limit
- **Type**: number
- **Required**: No (at least one limit required)
- **Description**: Maximum tool calls across all runs in a thread (conversation). Persists across multiple invocations with the same thread ID. Requires a checkpointer to maintain state. None means no thread limit.

#### run_limit
- **Type**: number
- **Required**: No (at least one limit required)
- **Description**: Maximum tool calls per single invocation (one user message → response cycle). Resets with each new user message. None means no run limit.

#### exit_behavior
- **Type**: string
- **Required**: No
- **Default**: "continue"
- **Description**: Behavior when limit is reached. Options are:
  - 'continue' (default): Block exceeded tool calls with error messages, let other tools and the model continue. The model decides when to end based on the error messages.
  - 'error': Raise a ToolCallLimitExceededError exception, stopping execution immediately.
  - 'end': Stop execution immediately with a ToolMessage and AI message for the exceeded tool call. Only works when limiting a single tool; raises NotImplementedError if other tools have pending calls.

### Usage Examples

#### Global Tool Call Limit
```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10)
    ]
)
```

#### Tool-Specific Limits
```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool],
    middleware=[
        ToolCallLimitMiddleware(thread_limit=20, run_limit=10),
        ToolCallLimitMiddleware(
            tool_name="search",
            thread_limit=5,
            run_limit=3
        )
    ]
)
```

#### Multiple Limiters with Different Exit Behaviors
```python
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware

global_limiter = ToolCallLimitMiddleware(thread_limit=20, run_limit=10)
search_limiter = ToolCallLimitMiddleware(tool_name="search", thread_limit=5, run_limit=3)
database_limiter = ToolCallLimitMiddleware(tool_name="query_database", thread_limit=10)
strict_limiter = ToolCallLimitMiddleware(tool_name="scrape_webpage", run_limit=2, exit_behavior="error")

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, database_tool, scraper_tool],
    middleware=[global_limiter, search_limiter, database_limiter, strict_limiter]
)
```

### Use Cases
- Preventing excessive calls to expensive external APIs
- Limiting web searches or database queries
- Enforcing rate limits on specific tool usage
- Protecting against runaway agent loops

### Notes
- At least one of `thread_limit` or `run_limit` must be specified
- Thread limits require a checkpointer to maintain state across invocations
- Run limits reset with each new user message/invocation
- The 'end' exit behavior only works when limiting a single tool
```

--------------------------------

### Trim Messages Before Model with @before_model Middleware

Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

Implements a @before_model middleware function that keeps only the last few messages in the agent state to fit within the context window. The function receives the current AgentState and returns a dictionary with modified messages or None if no changes are needed. This is useful for managing memory in long-running conversations.

```python
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from typing import Any


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


agent = create_agent(
    "gpt-5-nano",
    tools=[],
    middleware=[trim_messages],
    checkpointer=InMemorySaver()
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}

agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
```

--------------------------------

### Python LangChain Tool for Context Retrieval

Source: https://docs.langchain.com/oss/python/langchain/rag

This Python code defines a `retrieve_context` tool using the LangChain `@tool` decorator. It performs a similarity search on a `vector_store` based on a user `query`, serializes the retrieved document content and metadata, and returns both the stringified content and the raw document artifacts. The `response_format` ensures raw documents are attached to the `ToolMessage` for deeper application control.

```python
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
```

--------------------------------

### Configure pytest VCR Marker in pytest.ini

Source: https://docs.langchain.com/oss/python/langchain/test

Registers the vcr marker in pytest configuration and sets the record mode to 'once', which records HTTP interactions on the first test run and replays them on subsequent runs.

```ini
[pytest]
markers =
    vcr: record/replay HTTP via VCR
addopts = --record-mode=once
```

--------------------------------

### Python: Augment Langchain Message with Retrieved Context

Source: https://docs.langchain.com/oss/python/langchain/rag

This Python snippet demonstrates how to take a 'last_message' and augment its content by incorporating 'docs_content' (retrieved documents). It constructs a new message with the combined information, preparing it for an LLM call within a RAG system, and returns the updated message and the context.

```python
augmented_message_content = (
    f"{last_message.text}\n\n"
    "Use the following context to answer the query:\n"
    f"{docs_content}"
)
return {
    "messages": [last_message.model_copy(update={"content": augmented_message_content})],
    "context": retrieved_docs,
}
```

--------------------------------

### SkillMiddleware Class - Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Custom middleware class extending AgentMiddleware that injects skill descriptions into the system prompt during model calls. It builds a formatted skills list during initialization and appends it to the system message, enabling the model to reference available skills when processing requests.

```python
class SkillMiddleware(AgentMiddleware):
    """Middleware that injects skill descriptions into the system prompt."""

    # Register the load_skill tool as a class variable
    tools = [load_skill]

    def __init__(self):
        """Initialize and generate the skills prompt from SKILLS."""
        # Build skills prompt from the SKILLS list
        skills_list = []
        for skill in SKILLS:
            skills_list.append(
                f"- **{skill['name']}**: {skill['description']}"
            )
        self.skills_prompt = "\n".join(skills_list)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Sync: Inject skill descriptions into system prompt."""
        # Build the skills addendum
        skills_addendum = (
            f"\n\n## Available Skills\n\n{self.skills_prompt}\n\n"
            "Use the load_skill tool when you need detailed information "
            "about handling a specific type of request."
        )

        # Append to system message content blocks
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": skills_addendum}
        ]
        new_system_message = SystemMessage(content=new_content)
        modified_request = request.override(system_message=new_system_message)
        return handler(modified_request)
```

--------------------------------

### Initialize Amazon OpenSearch Vector Store with LangChain

Source: https://docs.langchain.com/oss/python/langchain/rag

This snippet illustrates how to install `boto3` and configure `OpenSearchVectorSearch` for Amazon OpenSearch. It requires AWS credentials, the OpenSearch host URL, region, and index name, using `RequestsHttpConnection` for secure HTTPS communication.

```shell
pip install -qU  boto3
```

```python
from opensearchpy import RequestsHttpConnection

service = "es"  # must set the service as 'es'
region = "us-east-2"
credentials = boto3.Session(
    aws_access_key_id="xxxxxx", aws_secret_access_key="xxxxx"
).get_credentials()
awsauth = AWS4Auth("xxxxx", "xxxxxx", region, service, session_token=credentials.token)

vector_store = OpenSearchVectorSearch.from_documents(
    docs,
    embeddings,
    opensearch_url="host url",
    http_auth=awsauth,
    timeout=300,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    index_name="test-index",
)
```

--------------------------------

### Update Resolution Specialist Prompt with Backward Transition Logic

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Modify the resolution specialist's system prompt to document the available backward transition tools and specify when they should be used. The prompt instructs the agent to use go_back_to_warranty or go_back_to_classification tools when customers indicate previous information was incorrect.

```python
RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping with device issues.

CURRENT STAGE: Resolution
CUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}

At this step, you need to:
1. For SOFTWARE issues: provide troubleshooting steps using provide_solution
2. For HARDWARE issues:
   - If IN WARRANTY: explain warranty repair process using provide_solution
   - If OUT OF WARRANTY: escalate_to_human for paid repair options

If the customer indicates any information was wrong, use:
- go_back_to_warranty to correct warranty status
- go_back_to_classification to correct issue type

Be specific and helpful in your solutions."""
```

--------------------------------

### Configure Structured Output Response Format in LangChain

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Demonstrates two functionally equivalent methods for specifying structured output response format in LangChain. When a provider natively supports structured output, you can use either direct class reference or explicit ProviderStrategy wrapper. If the provider does not support structured output, the system automatically falls back to a tool calling strategy.

```python
# Method 1: Direct structured output class
response_format = ProductReview

# Method 2: Explicit provider strategy
response_format = ProviderStrategy(ProductReview)

# Both are functionally equivalent when provider supports structured output
# If not supported, falls back to tool calling strategy
```

--------------------------------

### Initialize Milvus Vector Store with LangChain

Source: https://docs.langchain.com/oss/python/langchain/rag

This snippet illustrates how to install `langchain-milvus` and set up a `Milvus` vector store. It connects to a Milvus instance via a specified URI and allows configuring index parameters like `index_type` and `metric_type` for optimal performance.

```shell
pip install -qU langchain-milvus
```

```python
from langchain_milvus import Milvus

URI = "./milvus_example.db"

vector_store = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={"index_type": "FLAT", "metric_type": "L2"},
)
```

--------------------------------

### Configure Model Profile for Native Structured Output in Python

Source: https://docs.langchain.com/oss/python/langchain/structured-output

This Python example demonstrates how to manually configure a model's profile to explicitly declare support for native structured output. It involves creating a `custom_profile` dictionary with the `"structured_output": True` flag and passing it to the `init_chat_model` function.

```python
custom_profile = {
    "structured_output": True,
    # ...
}
model = init_chat_model("...", profile=custom_profile)
```

--------------------------------

### Configure LangSmith Environment Variables for Tracing in Bash

Source: https://docs.langchain.com/oss/python/langchain/test

This Bash snippet sets the `LANGSMITH_API_KEY` and `LANGSMITH_TRACING` environment variables. These are essential for enabling LangSmith integration and ensuring that evaluation results and traces are automatically logged to the platform.

```bash
export LANGSMITH_API_KEY="your_langsmith_api_key"
export LANGSMITH_TRACING="true"
```

--------------------------------

### Handle Specific Exception Type in ToolStrategy

Source: https://docs.langchain.com/oss/python/langchain/structured-output

Configure ToolStrategy to retry only on a specific exception type and raise all other exceptions. When handle_errors is an exception type like ValueError, only that exception type triggers a retry with the default error message.

```python
ToolStrategy(
    schema=ProductRating,
    handle_errors=ValueError  # Only retry on ValueError, raise others
)
```

--------------------------------

### Initialize LangChain PostgreSQL Vector Store

Source: https://docs.langchain.com/oss/python/langchain/rag

This Python snippet demonstrates how to initialize a `PGVectorStore` using `langchain_postgres`. It requires a PostgreSQL connection string to create a database engine and an embedding service to store vector embeddings for a specified table. This setup is crucial for persistent storage of vectorized data.

```python
from langchain_postgres import PGEngine, PGVectorStore

pg_engine = PGEngine.from_connection_string(
    url="postgresql+psycopg://..."
)

vector_store = PGVectorStore.create_sync(
    engine=pg_engine,
    table_name='test_table',
    embedding_service=embedding
)
```

--------------------------------

### Initialize AstraDB Vector Store with LangChain

Source: https://docs.langchain.com/oss/python/langchain/rag

This snippet shows how to install `langchain-astradb` and connect to AstraDB using `AstraDBVectorStore`. It requires an API endpoint, application token, namespace, and a collection name to store vector embeddings, leveraging an existing embeddings object.

```shell
pip install -U "langchain-astradb"
```

```python
from langchain_astradb import AstraDBVectorStore

vector_store = AstraDBVectorStore(
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    collection_name="astra_vector_langchain",
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)
```

--------------------------------

### Configure LangChain Model Retry Middleware to Re-raise Exception (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Demonstrates setting `on_failure='error'` in `ModelRetryMiddleware` to re-raise the original exception when all retry attempts are exhausted. This option is suitable for strict error handling where agent execution should stop immediately upon unrecoverable model call failures.

```python
strict_retry = ModelRetryMiddleware(
    max_retries=2,
    on_failure="error",  # Re-raise exception instead of returning message
)
```

--------------------------------

### Initialize MongoDB Atlas Vector Search with LangChain

Source: https://docs.langchain.com/oss/python/langchain/rag

This snippet shows how to install `langchain-mongodb` and configure `MongoDBAtlasVectorSearch`. It requires an embedding function, a MongoDB collection, an Atlas Vector Search index name, and can specify a relevance score function for search queries.

```shell
pip install -qU langchain-mongodb
```

```python
from langchain_mongodb import MongoDBAtlasVectorSearch

vector_store = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine",
)
```

--------------------------------

### Define LangChain Tool to Check Authentication from State (Python)

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

This Python code defines a LangChain tool, `check_authentication`, that accesses the `runtime.state` to determine if a user is authenticated. It demonstrates how tools can read current session information stored in the agent's state to inform their actions.

```python
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent

@tool
def check_authentication(
    runtime: ToolRuntime
) -> str:
    """Check if user is authenticated."""
    # Read from State: check current auth status
    current_state = runtime.state
    is_authenticated = current_state.get("authenticated", False)

    if is_authenticated:
        return "User is authenticated"
    else:
        return "User is not authenticated"

agent = create_agent(
    model="gpt-4o",
    tools=[check_authentication]
)
```

--------------------------------

### Stream custom tool updates with get_stream_writer - Python

Source: https://docs.langchain.com/oss/python/langchain/streaming

Uses get_stream_writer from langgraph.config to emit arbitrary custom updates from within tool functions during agent execution. Enables streaming progress updates and intermediate data from tools by calling writer() with custom messages, accessed via stream_mode='custom' in the agent.stream() call.

```python
from langchain.agents import create_agent
from langgraph.config import get_stream_writer


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    # stream any arbitrary data
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-5-20250929",
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom"
):
    print(chunk)
```

--------------------------------

### Map Classifications to LangGraph Send Objects (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This Python example illustrates how a list of classification objects, each containing a 'source' and 'query', is transformed into a list of `Send` objects. These `Send` objects are used in LangGraph to specify target nodes (agents) and the state to pass to them for parallel execution.

```python
# Classifications: [{"source": "github", "query": "..."}, {"source": "notion", "query": "..."}]
# Becomes:
[Send("github", {"query": "..."}), Send("notion", {"query": "..."})]
```

--------------------------------

### Create skill loading tool with LangChain decorator

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Defines a tool function decorated with @tool that loads full skill content on-demand. The function searches through a SKILLS list and returns the matching skill's complete content as a string, or an error message if not found. This tool integrates with LangChain's agent framework to provide detailed skill information when needed.

```python
from langchain.tools import tool

@tool
def load_skill(skill_name: str) -> str:
    """Load the full content of a skill into the agent's context.

    Use this when you need detailed information about how to handle a specific
    type of request. This will provide you with comprehensive instructions,
    policies, and guidelines for the skill area.

    Args:
        skill_name: The name of the skill to load (e.g., "expense_reporting", "travel_booking")
    """
    # Find and return the requested skill
    for skill in SKILLS:
        if skill["name"] == skill_name:
            return f"Loaded skill: {skill_name}\n\n{skill['content']}"

    # Skill not found
    available = ", ".join(s["name"] for s in SKILLS)
    return f"Skill '{skill_name}' not found. Available skills: {available}"
```

--------------------------------

### Implement Customer Support State Machine with LangChain and LangGraph

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

This comprehensive Python script defines a customer support state machine using LangChain agents and LangGraph's state management. It includes custom tools for recording warranty status and issue types, escalating to human support, and providing solutions, along with dynamically generated prompts for each step of the workflow. The system transitions through steps like `warranty_collector`, `issue_classifier`, and `resolution_specialist` based on the `SupportState`.

```python
"""
Customer Support State Machine Example

This example demonstrates the state machine pattern.
A single agent dynamically changes its behavior based on the current_step state,
creating a state machine for sequential information collection.
"""

import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from typing import Callable, Literal
from typing_extensions import NotRequired

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, SummarizationMiddleware
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool, ToolRuntime

model = init_chat_model("anthropic:claude-3-5-sonnet-latest")


# Define the possible workflow steps
SupportStep = Literal["warranty_collector", "issue_classifier", "resolution_specialist"]


class SupportState(AgentState):
    """State for customer support workflow."""

    current_step: NotRequired[SupportStep]
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]


@tool
def record_warranty_status(
    status: Literal["in_warranty", "out_of_warranty"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the customer's warranty status and transition to issue classification."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Warranty status recorded as: {status}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "warranty_status": status,
            "current_step": "issue_classifier",
        }
    )


@tool
def record_issue_type(
    issue_type: Literal["hardware", "software"],
    runtime: ToolRuntime[None, SupportState],
) -> Command:
    """Record the type of issue and transition to resolution specialist."""
    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Issue type recorded as: {issue_type}",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
            "issue_type": issue_type,
            "current_step": "resolution_specialist",
        }
    )


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the case to a human support specialist."""
    # In a real system, this would create a ticket, notify staff, etc.
    return f"Escalating to human support. Reason: {reason}"


@tool
def provide_solution(solution: str) -> str:
    """Provide a solution to the customer's issue."""
    return f"Solution provided: {solution}"


# Define prompts as constants
WARRANTY_COLLECTOR_PROMPT = """You are a customer support agent helping with device issues.\n\nCURRENT STEP: Warranty verification\n\nAt this step, you need to:\n1. Greet the customer warmly\n2. Ask if their device is under warranty\n3. Use record_warranty_status to record their response and move to the next step\n\nBe conversational and friendly. Don't ask multiple questions at once."""

ISSUE_CLASSIFIER_PROMPT = """You are a customer support agent helping with device issues.\n\nCURRENT STEP: Issue classification\nCUSTOMER INFO: Warranty status is {warranty_status}\n\nAt this step, you need to:\n1. Ask the customer to describe their issue\n2. Determine if it's a hardware issue (physical damage, broken parts) or software issue (app crashes, performance)\n3. Use record_issue_type to record the classification and move to the next step\n\nIf unclear, ask clarifying questions before classifying."""

RESOLUTION_SPECIALIST_PROMPT = """You are a customer support agent helping with device issues.\n\nCURRENT STEP: Resolution\nCUSTOMER INFO: Warranty status is {warranty_status}, issue type is {issue_type}\n\nAt this step, you need to:\n1. For SOFTWARE issues: provide troubleshooting steps using provide_solution\n2. For HARDWARE issues:\n   - If IN WARRANTY: explain warranty repair process using provide_solution\n   - If OUT OF WARRANTY: escalate_to_human for paid repair options\n\nBe specific and helpful in your solutions."""


# Step configuration: maps step name to (prompt, tools, required_state)
STEP_CONFIG = {
    "warranty_collector": {
        "prompt": WARRANTY_COLLECTOR_PROMPT,
        "tools": [record_warranty_status],
        "requires": [],
    },
    "issue_classifier": {
        "prompt": ISSUE_CLASSIFIER_PROMPT,
        "tools": [record_issue_type],
        "requires": ["warranty_status"],
    },
    "resolution_specialist": {
```

--------------------------------

### Pass Conversation Context to Sub-Agents with ToolRuntime

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Shows how to access and forward conversation history to sub-agents using ToolRuntime. This pattern allows sub-agents to understand full context for resolving ambiguous requests like temporal references (e.g., 'schedule for the same time tomorrow').

```python
from langchain.tools import tool, ToolRuntime

@tool
def schedule_event(
    request: str,
    runtime: ToolRuntime
) -> str:
    """Schedule calendar events using natural language."""
    # Customize context received by sub-agent
    original_user_message = next(
        message for message in runtime.state["messages"]
        if message.type == "human"
    )
    prompt = (
        "You are assisting with the following user inquiry:\n\n"
        f"{original_user_message.text}\n\n"
        "You are tasked with the following sub-request:\n\n"
        f"{request}"
    )
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": prompt}],
    })
    return result["messages"][-1].text
```

--------------------------------

### Set Custom LangSmith Project Name via Environment Variable

Source: https://docs.langchain.com/oss/python/langchain/observability

Configure a static custom project name for all LangSmith traces by setting the LANGSMITH_PROJECT environment variable. This ensures all traces from your application are logged to the specified project.

```bash
export LANGSMITH_PROJECT=my-agent-project
```

--------------------------------

### Decorate Test with VCR Marker for HTTP Recording

Source: https://docs.langchain.com/oss/python/langchain/test

Applies the vcr marker to a pytest test function to enable recording and replaying of HTTP calls. On first run, real network calls are made and responses are saved to a cassette file; subsequent runs use the cassette to mock responses.

```python
@pytest.mark.vcr()
def test_agent_trajectory():
    # ...
```

--------------------------------

### Access custom state fields using ToolRuntime in Python

Source: https://docs.langchain.com/oss/python/langchain/tools

This Python function demonstrates how to access custom state fields within a LangChain agent's runtime. It retrieves user preferences from the `runtime.state` object, returning a specific preference or 'Not set' if not found. The `runtime` parameter is automatically handled by the framework and not exposed to the model.

```python
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```

--------------------------------

### Python: Asynchronously Stream TTS Audio with Cartesia

Source: https://docs.langchain.com/oss/python/langchain/voice-agent

This asynchronous function `tts_stream` processes an input event stream, identifies agent text events, sends them to Cartesia for synthesis, and merges the resulting audio chunks back into the stream. It utilizes `CartesiaTTS` for text-to-speech conversion and `merge_async_iters` for concurrent stream processing to create a unified output stream with audio.

```python
from cartesia_tts import CartesiaTTS
from utils import merge_async_iters

async def tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with Audio)

    Merges two concurrent streams:
    1. process_upstream(): passes through events and sends text to Cartesia
    2. tts.receive_events(): yields audio chunks from Cartesia
    """
    tts = CartesiaTTS()

    async def process_upstream() -> AsyncIterator[VoiceAgentEvent]:
        """Process upstream events and send agent text to Cartesia."""
        async for event in event_stream:
            # Pass through all events
            yield event
            # Send agent text to Cartesia for synthesis
            if event.type == "agent_chunk":
                await tts.send_text(event.text)

    try:
        # Merge upstream events with TTS audio events
        # Both streams run concurrently
        async for event in merge_async_iters(
            process_upstream(),
            tts.receive_events()
        ):
            yield event
    finally:
        await tts.close()
```

--------------------------------

### Test Multi-Turn Customer Support Workflow in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Example test case demonstrating a complete customer support conversation flow with multiple turns. Creates a thread ID for conversation persistence and invokes the agent with sequential user messages representing warranty inquiry, issue description, and resolution request.

```python
if __name__ == "__main__":
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke(
        {"messages": [HumanMessage("Hi, my phone screen is cracked")]},
        config
    )

    result = agent.invoke(
        {"messages": [HumanMessage("Yes, it's still under warranty")]},
        config
    )

    result = agent.invoke(
        {"messages": [HumanMessage("The screen is physically cracked from dropping it")]},
        config
    )

    result = agent.invoke(
        {"messages": [HumanMessage("What should I do?")]},
        config
    )
    for msg in result['messages']:
        msg.pretty_print()
```

--------------------------------

### Execute Pytest with LangSmith Output Integration in Bash

Source: https://docs.langchain.com/oss/python/langchain/test

This Bash command executes a Pytest file, `test_trajectory.py`, with the `--langsmith-output` flag. This flag instructs Pytest to automatically log all evaluation results and test traces to your configured LangSmith project, streamlining experiment tracking.

```bash
pytest test_trajectory.py --langsmith-output
```

--------------------------------

### Install langchain-mcp-adapters library

Source: https://docs.langchain.com/oss/python/langchain/mcp

Install the `langchain-mcp-adapters` library using pip or uv to enable LangChain agents to interact with MCP servers and their defined tools. This library is a prerequisite for integrating MCP tools.

```bash
pip install langchain-mcp-adapters
```

```bash
uv add langchain-mcp-adapters
```

--------------------------------

### State-Aware System Prompt with Message Count in Python

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Creates a dynamic system prompt that accesses the conversation message count from the request state. The prompt adjusts its behavior based on conversation length, becoming more concise for longer conversations. Uses the @dynamic_prompt decorator to integrate with LangChain agent middleware.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)

    base = "You are a helpful assistant."

    if message_count > 10:
        base += "\nThis is a long conversation - be extra concise."

    return base

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[state_aware_prompt]
)
```

--------------------------------

### Define ProviderStrategy Class for Native Structured Output in LangChain

Source: https://docs.langchain.com/oss/python/langchain/structured-output

The ProviderStrategy class is a generic type used in LangChain to configure native structured output from LLM providers. It requires a 'schema' to define the desired output format and optionally accepts a 'strict' boolean parameter to enforce schema adherence, which requires langchain>=1.2. This class acts as a blueprint for implementing structured response handling.

```python
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    strict: bool | None = None
```

--------------------------------

### Access runtime context with ToolRuntime in Python

Source: https://docs.langchain.com/oss/python/langchain/tools

This Python example shows how to define a custom context schema and access immutable contextual data via `runtime.context` within a LangChain tool. The `get_account_info` tool retrieves user details based on a `user_id` provided in the `UserContext`. It also demonstrates creating an agent that is configured to utilize this context schema and invoke the tool with specific context.

```python
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime


USER_DATABASE = {
    "user123": {
        "name": "Alice Johnson",
        "account_type": "Premium",
        "balance": 5000,
        "email": "alice@example.com"
    },
    "user456": {
        "name": "Bob Smith",
        "account_type": "Standard",
        "balance": 1200,
        "email": "bob@example.com"
    }
}

@dataclass
class UserContext:
    user_id: str

@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

    if user_id in USER_DATABASE:
        user = USER_DATABASE[user_id]
        return f"Account holder: {user['name']}\nType: {user['account_type']}\nBalance: ${user['balance']}"
    return "User not found"

model = ChatOpenAI(model="gpt-4o")
agent = create_agent(
    model,
    tools=[get_account_info],
    context_schema=UserContext,
    system_prompt="You are a financial assistant."
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's my current balance?"}]},
    context=UserContext(user_id="user123")
)
```

--------------------------------

### WebSocket Endpoint for Audio Streaming with LangChain Pipeline

Source: https://docs.langchain.com/oss/python/langchain/voice-agent

Implements an async WebSocket endpoint that receives audio bytes from a client, transforms them through a LangChain pipeline (using RunnableGenerators for composition), and streams back text-to-speech audio chunks. The pipeline processes audio transcription, agent reasoning, and speech synthesis concurrently to achieve low-latency natural conversation.

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def websocket_audio_stream():
        """Yield audio bytes from WebSocket."""
        while True:
            data = await websocket.receive_bytes()
            yield data

    # Transform audio through pipeline
    output_stream = pipeline.atransform(websocket_audio_stream())

    # Send TTS audio back to client
    async for event in output_stream:
        if event.type == "tts_chunk":
            await websocket.send_bytes(event.audio)
```

--------------------------------

### Apply Metadata and Tags Using LangSmith Tracing Context (Python)

Source: https://docs.langchain.com/oss/python/langchain/observability

This example illustrates the use of `ls.tracing_context` to wrap a block of code, applying custom metadata and tags to all traces generated within that context. This method offers fine-grained control for annotating specific operations or workflows with detailed information for LangSmith analysis.

```python
with ls.tracing_context(
    project_name="email-agent-test",
    enabled=True,
    tags=["production", "email-assistant", "v1.0"],
    metadata={"user_id": "user_123", "session_id": "session_456", "environment": "production"}):
    response = agent.invoke(
        {"messages": [{"role": "user", "content": "Send a welcome email"}]}
    )
```

--------------------------------

### Content Filter Middleware - Class Syntax

Source: https://docs.langchain.com/oss/python/langchain/guardrails

Implements a deterministic guardrail using class-based middleware to block requests containing banned keywords. The before_agent hook validates the first user message and returns a jump_to directive to skip processing if inappropriate content is detected.

```python
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langgraph.runtime import Runtime

class ContentFilterMiddleware(AgentMiddleware):
    """Deterministic guardrail: Block requests containing banned keywords."""

    def __init__(self, banned_keywords: list[str]):
        super().__init__()
        self.banned_keywords = [kw.lower() for kw in banned_keywords]

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        # Get the first user message
        if not state["messages"]:
            return None

        first_message = state["messages"][0]
        if first_message.type != "human":
            return None

        content = first_message.content.lower()

        # Check for banned keywords
        for keyword in self.banned_keywords:
            if keyword in content:
                # Block execution before any processing
                return {
                    "messages": [{
                        "role": "assistant",
                        "content": "I cannot process requests containing inappropriate content. Please rephrase your request."
                    }],
                    "jump_to": "end"
                }

        return None

# Use the custom guardrail
from langchain.agents import create_agent

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, calculator_tool],
    middleware=[
        ContentFilterMiddleware(
            banned_keywords=["hack", "exploit", "malware"]
        ),
    ],
)

# This request will be blocked before any processing
result = agent.invoke({
    "messages": [{"role": "user", "content": "How do I hack into a database?"}]
})
```

--------------------------------

### Configure SummarizationMiddleware for message history - Python

Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

Creates an agent with SummarizationMiddleware to automatically summarize message history when token count exceeds 4000, keeping the last 20 messages. The middleware uses a lighter model (gpt-4o-mini) for summarization to optimize performance. Uses InMemorySaver for checkpoint management and demonstrates multi-turn conversation with memory persistence.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
"""
================================== Ai Message ==================================

Your name is Bob!
"""
```

--------------------------------

### Configure MultiServerMCPClient for multiple MCP servers in Python

Source: https://docs.langchain.com/oss/python/langchain/mcp

Demonstrates how to initialize `MultiServerMCPClient` to access tools from multiple MCP servers. It configures a math server using `stdio` transport and a weather server using `http` transport, then creates a LangChain agent to invoke tools from both servers.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent


client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",  # Local subprocess communication
            "command": "python",
            # Absolute path to your math_server.py file
            "args": ["/path/to/math_server.py"],
        },
        "weather": {
            "transport": "http",  # HTTP-based remote server
            # Ensure you start your weather server on port 8000
            "url": "http://localhost:8000/mcp",
        }
    }
)

tools = await client.get_tools()
agent = create_agent(
    "claude-sonnet-4-5-20250929",
    tools
)
math_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]}
)
weather_response = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "what is the weather in nyc?"}]}
)

```

--------------------------------

### Process Interrupt Decisions Using Command Objects

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Uses LangGraph Command objects to specify approval, edit, or rejection decisions for each interrupted action by referencing the interrupt ID, enabling selective approval or modification of sensitive tool calls.

```python
from langgraph.types import Command

resume = {}
for interrupt_ in interrupts:
```

--------------------------------

### Define custom MCP tools with FastMCP in Python

Source: https://docs.langchain.com/oss/python/langchain/mcp

Illustrates how to create two custom MCP servers using FastMCP: a math server with `stdio` transport and a weather server with `streamable-http` transport. Each server defines tools (`add`, `multiply`, `get_weather`) that agents can invoke.

```python
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

```python
from fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

--------------------------------

### Define State Schemas with TypedDict

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

Create three state schema classes to manage the multi-agent workflow: AgentInput for subagent queries, AgentOutput for subagent results, and RouterState for tracking classifications, results, and final answers. Uses a reducer pattern with operator.add to collect parallel results.

```python
from typing import Annotated, Literal, TypedDict
import operator


class AgentInput(TypedDict):
    """Simple input state for each subagent."""
    query: str


class AgentOutput(TypedDict):
    """Output from each subagent."""
    source: str
    result: str


class Classification(TypedDict):
    """A single routing decision: which agent to call with what query."""
    source: Literal["github", "notion", "slack"]
    query: str


class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    results: Annotated[list[AgentOutput], operator.add]
    final_answer: str
```

--------------------------------

### Invoke LangChain Router Workflow and Display Results (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This Python code demonstrates how to execute a LangChain router workflow with a specific query. It then prints the original query, the classifications made by the router (showing which agents were invoked and their specific queries), and the final synthesized answer from the workflow.

```python
result = workflow.invoke({
    "query": "How do I authenticate API requests?"
})

print("Original query:", result["query"])
print("\nClassifications:")
for c in result["classifications"]:
    print(f"  {c['source']}: {c['query']}")
print("\n" + "=" * 60 + "\n")
print("Final Answer:")
print(result["final_answer"])
```

--------------------------------

### Mocking Chat Model Responses in LangChain Python

Source: https://docs.langchain.com/oss/python/langchain/test

This Python code demonstrates how to mock responses from a chat model using LangChain's `GenericFakeChatModel`. It initializes the fake model with an iterator of `AIMessage` or string responses, allowing for deterministic testing of chat model interactions without making actual API calls. The first code block shows the initialization and initial invocation. The second block illustrates a subsequent invocation, retrieving the next item from the pre-defined response iterator, useful for simulating multi-turn conversations or sequential model outputs.

```python
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel

model = GenericFakeChatModel(messages=iter([
    AIMessage(content="", tool_calls=[ToolCall(name="foo", args={"bar": "baz"}, id="call_1")]),
    "bar"
]))

model.invoke("hello")
# AIMessage(content='', ..., tool_calls=[{'name': 'foo', 'args': {'bar': 'baz'}, 'id': 'call_1', 'type': 'tool_call'}])
```

```python
model.invoke("hello, again!")
# AIMessage(content='bar', ...)
```

--------------------------------

### Compile LangChain Router Workflow with Conditional Edges (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This Python code block demonstrates how to assemble the complete LangChain workflow using `StateGraph`. It defines each function (classify, query_github, query_notion, query_slack, synthesize) as a node and establishes the flow using `add_edge` and `add_conditional_edges`. The `add_conditional_edges` call, specifically, uses the `route_to_agents` function to enable parallel execution of multiple agent nodes based on classification results.

```python
workflow = (
    StateGraph(RouterState)
    .add_node("classify", classify_query)
    .add_node("github", query_github)
    .add_node("notion", query_notion)
    .add_node("slack", query_slack)
    .add_node("synthesize", synthesize_results)
    .add_edge(START, "classify")
    .add_conditional_edges("classify", route_to_agents, ["github", "notion", "slack"])
    .add_edge("github", "synthesize")
    .add_edge("notion", "synthesize")
    .add_edge("slack", "synthesize")
    .add_edge("synthesize", END)
    .compile()
)
```

--------------------------------

### Invoke and Print LangChain Router Workflow Results in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This Python script demonstrates how to execute the defined LangChain StateGraph workflow with a sample query. It invokes the 'workflow' with an initial 'query', then prints the original query, the classifications made by the router, and the final synthesized answer provided by the agentic system.

```python
  if __name__ == "__main__":
      result = workflow.invoke({
          "query": "How do I authenticate API requests?"
      })

      print("Original query:", result["query"])
      print("\nClassifications:")
      for c in result["classifications"]:
          print(f"  {c['source']}: {c['query']}")
      print("\n" + "=" * 60 + "\n")
      print("Final Answer:")
      print(result["final_answer"])
```

--------------------------------

### Inspect and Display Interrupt Event Details

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Iterates through collected interrupt events to extract and display action request details including interrupt ID, description, tool name, and associated arguments for review.

```python
for interrupt_ in interrupts:
    for request in interrupt_.value["action_requests"]:
        print(f"INTERRUPTED: {interrupt_.id}")
        print(f"{request['description']}\n")
```

--------------------------------

### Install LangChain Anthropic Integration

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Installs the necessary Python packages for integrating Anthropic chat models with LangChain. This command ensures that the `langchain` library and its Anthropic-specific dependencies are installed or updated.

```shell
pip install -U "langchain[anthropic]"
```

--------------------------------

### Install LangChain Google Gemini Integration

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Installs the necessary Python packages for integrating Google Gemini chat models with LangChain. This command ensures that the `langchain` library and its Google GenAI-specific dependencies are installed or updated.

```shell
pip install -U "langchain[google-genai]"
```

--------------------------------

### Dynamic Tool Selection with Class-Based Middleware

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Implements a class-based middleware by extending AgentMiddleware to dynamically select relevant tools based on request state and runtime context. The wrap_model_call method filters tools before passing to the handler, improving performance and accuracy while enabling permission-based tool filtering.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable


class ToolSelectorMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Middleware to select relevant tools based on state/context."""
        # Select a small, relevant subset of tools based on state/context
        relevant_tools = select_relevant_tools(request.state, request.runtime)
        return handler(request.override(tools=relevant_tools))

agent = create_agent(
    model="gpt-4o",
    tools=all_tools,  # All available tools need to be registered upfront
    middleware=[ToolSelectorMiddleware()],
)
```

--------------------------------

### Wrap-style Hook with Class - Retry Middleware

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Implements AgentMiddleware class with configurable wrap_model_call method for model call interception. Provides retry logic with configurable max_retries parameter and error logging, allowing short-circuit, normal, or retry execution patterns.

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from typing import Callable

class RetryMiddleware(AgentMiddleware):
    def __init__(self, max_retries: int = 3):
        super().__init__()
        self.max_retries = max_retries

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        for attempt in range(self.max_retries):
            try:
                return handler(request)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"Retry {attempt + 1}/{self.max_retries} after error: {e}")
```

--------------------------------

### Dynamic Model Selection with Class-Based Middleware

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Implements dynamic model selection using AgentMiddleware class with wrap_model_call method. Selects different language models based on conversation message count for optimized performance and cost efficiency. Overrides model in request handler.

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

complex_model = init_chat_model("gpt-4o")
simple_model = init_chat_model("gpt-4o-mini")

class DynamicModelMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # Use different model based on conversation length
        if len(request.messages) > 10:
            model = complex_model
        else:
            model = simple_model
        return handler(request.override(model=model))
```

--------------------------------

### Install LangChain HuggingFace Integration

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Installs the necessary Python packages for integrating HuggingFace models with LangChain. This command ensures that the `langchain` library and its HuggingFace-specific dependencies are installed or updated.

```shell
pip install -U "langchain[huggingface]"
```

--------------------------------

### Node-style Hook with Class - Message Limit Middleware

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Implements AgentMiddleware class with before_model and after_model methods using hook_config decorator. Provides configurable max_messages parameter and can jump to end state when limit is reached, along with logging of model responses.

```python
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any

class MessageLimitMiddleware(AgentMiddleware):
    def __init__(self, max_messages: int = 50):
        super().__init__()
        self.max_messages = max_messages

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        if len(state["messages"]) == self.max_messages:
            return {
                "messages": [AIMessage("Conversation limit reached.")],
                "jump_to": "end"
            }
        return None

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"Model returned: {state['messages'][-1].content}")
        return None
```

--------------------------------

### Install LangChain AWS Bedrock Integration

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Installs the necessary Python packages for integrating AWS Bedrock chat models with LangChain. This command ensures that the `langchain` library and its AWS-specific dependencies are installed or updated.

```shell
pip install -U "langchain[aws]"
```

--------------------------------

### Define Sales Analytics Skill with Database Schema

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Defines a complete sales analytics skill with database schema documentation including customers, orders, and order_items tables, business logic rules, and an example SQL query for revenue analysis.

```python
{
    "name": "sales_analytics",
    "description": "Database schema and business logic for sales data analysis including customers, orders, and revenue.",
    "content": """# Sales Analytics Schema

## Tables

### customers
- customer_id (PRIMARY KEY)
- name
- email
- signup_date
- status (active/inactive)
- customer_tier (bronze/silver/gold/platinum)

### orders
- order_id (PRIMARY KEY)
- customer_id (FOREIGN KEY -> customers)
- order_date
- status (pending/completed/cancelled/refunded)
- total_amount
- sales_region (north/south/east/west)

### order_items
- item_id (PRIMARY KEY)
- order_id (FOREIGN KEY -> orders)
- product_id
- quantity
- unit_price
- discount_percent

## Business Logic

**Active customers**: status = 'active' AND signup_date <= CURRENT_DATE - INTERVAL '90 days'

**Revenue calculation**: Only count orders with status = 'completed'. Use total_amount from orders table, which already accounts for discounts.

**Customer lifetime value (CLV)**: Sum of all completed order amounts for a customer.

**High-value orders**: Orders with total_amount > 1000

## Example Query

SELECT
    c.customer_id,
    c.name,
    c.customer_tier,
    SUM(o.total_amount) as total_revenue
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'completed'
  AND o.order_date >= CURRENT_DATE - INTERVAL '3 months'
GROUP BY c.customer_id, c.name, c.customer_tier
ORDER BY total_revenue DESC
LIMIT 10;"""
}
```

--------------------------------

### Select Model from User Preferences in Store

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Retrieves user's preferred model from persistent storage using the InMemoryStore. Looks up user preferences by user_id and applies the stored model selection if available. Includes a Context dataclass to pass user information through the request pipeline.

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

# Initialize available models once
MODEL_MAP = {
    "gpt-4o": init_chat_model("gpt-4o"),
    "gpt-4o-mini": init_chat_model("gpt-4o-mini"),
    "claude-sonnet": init_chat_model("claude-sonnet-4-5-20250929"),
}

@wrap_model_call
def store_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on Store preferences."""
    user_id = request.runtime.context.user_id

    # Read from Store: get user's preferred model
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)

    if user_prefs:
        preferred_model = user_prefs.value.get("preferred_model")
        if preferred_model and preferred_model in MODEL_MAP:
            request = request.override(model=MODEL_MAP[preferred_model])

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[store_based_model],
    context_schema=Context,
    store=InMemoryStore()
)
```

--------------------------------

### Install LangChain OpenAI Integration

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Installs the necessary Python packages for integrating OpenAI chat models with LangChain. This command ensures that the `langchain` library and its OpenAI-specific dependencies are installed or updated.

```shell
pip install -U "langchain[openai]"
```

--------------------------------

### Monitor Tool Calls with Class-Based Middleware

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Implements a class-based middleware by extending AgentMiddleware to monitor tool execution with logging and error handling. The wrap_tool_call method intercepts tool calls, logs execution details, and handles exceptions while preserving the handler's return type.

```python
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents.middleware import AgentMiddleware
from langchain.messages import ToolMessage
from langgraph.types import Command
from typing import Callable

class ToolMonitoringMiddleware(AgentMiddleware):
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        print(f"Executing tool: {request.tool_call['name']}")
        print(f"Arguments: {request.tool_call['args']}")
        try:
            result = handler(request)
            print(f"Tool completed successfully")
            return result
        except Exception as e:
            print(f"Tool failed: {e}")
            raise
```

--------------------------------

### Initialize Google Gemini Chat Model in LangChain

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Demonstrates two methods to initialize a Google Gemini chat model in LangChain. Both methods require setting the `GOOGLE_API_KEY` environment variable for authentication with the Google Generative AI service.

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GOOGLE_API_KEY"] = "..."

model = init_chat_model("google_genai:gemini-2.5-flash-lite")
```

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "..."

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
```

--------------------------------

### Store-based Output Format Selection with User Preferences

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Configure response format based on user preferences stored in InMemoryStore. Retrieves user preference data using user_id and selects between verbose and concise response schemas. Requires runtime context with user_id and a store instance to retrieve preference data.

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from pydantic import BaseModel, Field
from typing import Callable
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

class VerboseResponse(BaseModel):
    """Verbose response with details."""
    answer: str = Field(description="Detailed answer")
    sources: list[str] = Field(description="Sources used")

class ConciseResponse(BaseModel):
    """Concise response."""
    answer: str = Field(description="Brief answer")

@wrap_model_call
def store_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select output format based on Store preferences."""
    user_id = request.runtime.context.user_id

    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)

    if user_prefs:
        style = user_prefs.value.get("response_style", "concise")
        if style == "verbose":
            request = request.override(response_format=VerboseResponse)
        else:
            request = request.override(response_format=ConciseResponse)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[store_based_output],
    context_schema=Context,
    store=InMemoryStore()
)
```

--------------------------------

### Define Pydantic Schema for Customer Support Ticket in Python

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

This Python code defines a Pydantic `BaseModel` named `CustomerSupportTicket` to structure customer issue data. It specifies fields like category, priority, summary, and customer_sentiment, each with a type and a descriptive `Field` annotation. This schema can be used to validate and parse unstructured text into a consistent, machine-readable format for downstream processing.

```python
from pydantic import BaseModel, Field

class CustomerSupportTicket(BaseModel):
    """Structured ticket information extracted from customer message."""

    category: str = Field(
        description="Issue category: 'billing', 'technical', 'account', or 'product'"
    )
    priority: str = Field(
        description="Urgency level: 'low', 'medium', 'high', or 'critical'"
    )
    summary: str = Field(
        description="One-sentence summary of the customer's issue"
    )
    customer_sentiment: str = Field(
        description="Customer's emotional tone: 'frustrated', 'neutral', or 'satisfied'"
    )
```

--------------------------------

### Select Model Based on Conversation State Length

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Implements state-based model selection by analyzing conversation message count. Routes to larger context window models for longer conversations and efficient models for short interactions. Uses the @wrap_model_call decorator to intercept and modify model requests.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

# Initialize models once outside the middleware
large_model = init_chat_model("claude-sonnet-4-5-20250929")
standard_model = init_chat_model("gpt-4o")
efficient_model = init_chat_model("gpt-4o-mini")

@wrap_model_call
def state_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on State conversation length."""
    # request.messages is a shortcut for request.state["messages"]
    message_count = len(request.messages)  # [!code highlight]

    if message_count > 20:
        # Long conversation - use model with larger context window
        model = large_model
    elif message_count > 10:
        # Medium conversation
        model = standard_model
    else:
        # Short conversation - use efficient model
        model = efficient_model

    request = request.override(model=model)  # [!code highlight]

    return handler(request)

agent = create_agent(
    model="gpt-4o-mini",
    tools=[...],
    middleware=[state_based_model]
)
```

--------------------------------

### Initialize AWS Bedrock Chat Model in LangChain

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Demonstrates two methods to initialize an AWS Bedrock chat model in LangChain. It requires proper AWS credentials configuration as outlined in the AWS Bedrock documentation.

```python
from langchain.chat_models import init_chat_model

# Follow the steps here to configure your credentials:
# https://docs.aws.amazon.com/bedrock/latest/userguide/getting-started.html

model = init_chat_model(
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    model_provider="bedrock_converse",
)
```

```python
from langchain_aws import ChatBedrock

model = ChatBedrock(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
```

--------------------------------

### Select Model Based on Runtime Context and Cost Tier

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Implements context-based model selection using environment and cost tier information from Runtime Context. Routes premium users to best-performing models, budget users to efficient models, and standard users to balanced models. Useful for cost-aware deployments.

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

@dataclass
class Context:
    cost_tier: str
    environment: str

# Initialize models once outside the middleware
premium_model = init_chat_model("claude-sonnet-4-5-20250929")
standard_model = init_chat_model("gpt-4o")
budget_model = init_chat_model("gpt-4o-mini")

@wrap_model_call
def context_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select model based on Runtime Context."""
    # Read from Runtime Context: cost tier and environment
    cost_tier = request.runtime.context.cost_tier
    environment = request.runtime.context.environment

    if environment == "production" and cost_tier == "premium":
        # Production premium users get best model
        model = premium_model
    elif cost_tier == "budget":
        # Budget tier gets efficient model
        model = budget_model
    else:
        # Standard tier
        model = standard_model

    request = request.override(model=model)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[context_based_model],
    context_schema=Context
)
```

--------------------------------

### Runtime Context-Aware System Prompt with Role and Environment in Python

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Creates a dynamic system prompt that accesses user role and deployment environment from Runtime Context. The prompt adjusts permissions and safety guidelines based on user role (admin/viewer) and deployment environment (production/staging). Uses a Context dataclass to define available context fields.

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class Context:
    user_role: str
    deployment_env: str

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    # Read from Runtime Context: user role and environment
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    base = "You are a helpful assistant."

    if user_role == "admin":
        base += "\nYou have admin access. You can perform all operations."
    elif user_role == "viewer":
        base += "\nYou have read-only access. Guide users to read operations only."

    if env == "production":
        base += "\nBe extra careful with any data modifications."

    return base

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[context_aware_prompt],
    context_schema=Context
)
```

--------------------------------

### Configure Structured Output with Pydantic BaseModel

Source: https://docs.langchain.com/oss/python/langchain/models

Demonstrates how to use Pydantic BaseModel with with_structured_output() to enforce typed responses from language models. The include_raw parameter captures both the raw API response and parsed structured data. This approach ensures type safety and enables validation of model outputs against defined schemas.

```python
year: int = Field(..., description="The year the movie was released")
director: str = Field(..., description="The director of the movie")
rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie, include_raw=True)
response = model_with_structure.invoke("Provide details about the movie Inception")
response
# {
#     "raw": AIMessage(...),
#     "parsed": Movie(title=..., year=..., ...),
#     "parsing_error": None,
# }
```

--------------------------------

### Runtime Context-based Output Format Selection

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Select response format based on runtime context attributes like user role and environment. Admin users in production receive detailed technical responses with debug information and system status, while regular users receive simplified responses. Evaluates both role and environment conditions to determine appropriate schema.

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from pydantic import BaseModel, Field
from typing import Callable

@dataclass
class Context:
    user_role: str
    environment: str

class AdminResponse(BaseModel):
    """Response with technical details for admins."""
    answer: str = Field(description="Answer")
    debug_info: dict = Field(description="Debug information")
    system_status: str = Field(description="System status")

class UserResponse(BaseModel):
    """Simple response for regular users."""
    answer: str = Field(description="Answer")

@wrap_model_call
def context_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select output format based on Runtime Context."""
    user_role = request.runtime.context.user_role
    environment = request.runtime.context.environment

    if user_role == "admin" and environment == "production":
        request = request.override(response_format=AdminResponse)
    else:
        request = request.override(response_format=UserResponse)

    return handler(request)
```

--------------------------------

### Initialize Anthropic Chat Model in LangChain

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Demonstrates two methods to initialize an Anthropic chat model in LangChain: using the generic `init_chat_model` function or by directly instantiating `ChatAnthropic`. Both methods require setting the `ANTHROPIC_API_KEY` environment variable for authentication.

```python
import os
from langchain.chat_models import init_chat_model

os.environ["ANTHROPIC_API_KEY"] = "sk-..."

model = init_chat_model("claude-sonnet-4-5-20250929")
```

```python
import os
from langchain_anthropic import ChatAnthropic

os.environ["ANTHROPIC_API_KEY"] = "sk-..."

model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
```

--------------------------------

### Define After Model Hook with Decorator

Source: https://docs.langchain.com/oss/python/langchain/runtime

Create a post-model hook function using the @after_model decorator that receives AgentState and Runtime context. This function executes after the model completes inference and can access runtime context data like user information. Returns None or a dict for state updates.

```python
@after_model
def log_after_model(state: AgentState, runtime: Runtime[Context]) -> dict | None:
    print(f"Completed request for user: {runtime.context.user_name}")
    return None
```

--------------------------------

### Define Movie Schema using Pydantic in Python

Source: https://docs.langchain.com/oss/python/langchain/models

This code defines a movie schema using Pydantic, which includes fields for title, year, director, and rating. It leverages Pydantic's features for field validation and descriptions, allowing for structured and validated output from the language model. The example shows how to instantiate the `Movie` model.

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # Movie(title="Inception", year=2010, director="Christopher Nolan", rating=8.8)
```

--------------------------------

### Define Nested Structures with Pydantic BaseModel

Source: https://docs.langchain.com/oss/python/langchain/models

Shows how to create nested Pydantic BaseModel schemas with complex hierarchical data structures. This example defines an Actor model nested within MovieDetails, enabling type-safe representation of complex movie data including cast lists, genres, and optional budget information.

```python
from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_with_structure = model.with_structured_output(MovieDetails)
```

--------------------------------

### Initialize Anthropic Claude Chat Model with ChatAnthropic Class

Source: https://docs.langchain.com/oss/python/langchain/models

Initialize an Anthropic Claude model directly using the ChatAnthropic class from langchain_anthropic. Requires ANTHROPIC_API_KEY environment variable. Provides direct access to Anthropic-specific configuration.

```python
import os
from langchain_anthropic import ChatAnthropic

os.environ["ANTHROPIC_API_KEY"] = "sk-..."

model = ChatAnthropic(model="claude-sonnet-4-5-20250929")
```

--------------------------------

### Python: Implement Cartesia Text-to-Speech Client

Source: https://docs.langchain.com/oss/python/langchain/voice-agent

This `CartesiaTTS` class provides a custom client for interacting with the Cartesia TTS API via WebSockets. It handles connection management, sending text for synthesis, and receiving audio chunks, allowing for configurable voice, model, sample rate, and encoding. The `_ensure_connection` method establishes a WebSocket connection if one doesn't exist.

```python
import base64
import json
import websockets

class CartesiaTTS:
    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = "f6ff7c0c-e396-40a9-a70b-f7607edb6937",
        model_id: str = "sonic-3",
        sample_rate: int = 24000,
        encoding: str = "pcm_s16le",
    ):
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY")
        self.voice_id = voice_id
        self.model_id = model_id
        self.sample_rate = sample_rate
        self.encoding = encoding
        self._ws: WebSocketClientProtocol | None = None

    def _generate_context_id(self) -> str:
        """Generate a valid context_id for Cartesia."""
        timestamp = int(time.time() * 1000)
        counter = self._context_counter
        self._context_counter += 1
        return f"ctx_{timestamp}_{counter}"

    async def send_text(self, text: str | None) -> None:
        """Send text to Cartesia for synthesis."""
        if not text or not text.strip():
            return

        ws = await self._ensure_connection()
        payload = {
            "model_id": self.model_id,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": self.voice_id,
            },
            "output_format": {
                "container": "raw",
                "encoding": self.encoding,
                "sample_rate": self.sample_rate,
            },
            "language": self.language,
            "context_id": self._generate_context_id(),
        }
        await ws.send(json.dumps(payload))

    async def receive_events(self) -> AsyncIterator[TTSChunkEvent]:
        """Yield audio chunks as they arrive from Cartesia."""
        async for raw_message in self._ws:
            message = json.loads(raw_message)

            # Decode and yield audio chunks
            if "data" in message and message["data"]:
                audio_chunk = base64.b64decode(message["data"])
                if audio_chunk:
                    yield TTSChunkEvent.create(audio_chunk)

    async def _ensure_connection(self) -> WebSocketClientProtocol:
        """Establish WebSocket connection if not already connected."""
        if self._ws is None:
            url = (
                f"wss://api.cartesia.ai/tts/websocket"
                f"?api_key={self.api_key}&cartesia_version={self.cartesia_version}"
            )
            self._ws = await websockets.connect(url)

        return self._ws
```

--------------------------------

### Add Context to System Message - Class-Based Middleware

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Demonstrates how to modify system messages using the class-based AgentMiddleware approach. This method extends AgentMiddleware and overrides wrap_model_call to add additional context blocks to the system message.

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from typing import Callable


class ContextMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # Always work with content blocks
        new_content = list(request.system_message.content_blocks) + [
            {"type": "text", "text": "Additional context."}
        ]
        new_system_message = SystemMessage(content=new_content)
        return handler(request.override(system_message=new_system_message))
```

--------------------------------

### Create Constrained Tool with Skill Prerequisites

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Implements a constrained tool that checks if required skills have been loaded before allowing execution. The write_sql_query tool validates that the appropriate skill has been loaded and returns an error message if prerequisites are not met.

```python
@tool
def write_sql_query(
    query: str,
    vertical: str,
    runtime: ToolRuntime,
) -> str:
    """Write and validate a SQL query for a specific business vertical.

    This tool helps format and validate SQL queries. You must load the
    appropriate skill first to understand the database schema.

    Args:
        query: The SQL query to write
        vertical: The business vertical (sales_analytics or inventory_management)
    """
    skills_loaded = runtime.state.get("skills_loaded", [])

    if vertical not in skills_loaded:
        return (
            f"Error: You must load the '{vertical}' skill first "
            f"to understand the database schema before writing queries. "
            f"Use load_skill('{vertical}') to load the schema."
        )

    return (
        f"SQL Query for {vertical}:\n\n"
        f"```sql\n{query}\n```\n\n"
        f"✓ Query validated against {vertical} schema\n"
        f"Ready to execute against the database."
    )
```

--------------------------------

### Edit Tool Call Decision - LangGraph Python

Source: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

Modify a tool call before execution by providing an edited action with new tool name and arguments. The edited_action object specifies the tool name and updated arguments. Conservative modifications are recommended to avoid model re-evaluation and unexpected multiple executions.

```python
agent.invoke(
    Command(
        resume={
            "decisions": [
                {
                    "type": "edit",
                    "edited_action": {
                        "name": "new_tool_name",
                        "args": {"key1": "new_value", "key2": "original_value"},
                    }
                }
            ]
        }
    ),
    config=config
)
```

--------------------------------

### Initialize PGVectorStore with Engine

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Configures PostgreSQL vector store using PGEngine for connection management. Provides abstraction layer over raw connection strings and supports additional connection pooling and configuration options.

```python
from langchain_postgres import PGEngine, PGVectorStore

pg_engine = PGEngine.from_connection_string(
    url="postgresql+psycopg://..."
)
```

--------------------------------

### Example: Refresh Anthropic Model Profiles (Bash)

Source: https://docs.langchain.com/oss/python/langchain/models

Provides a concrete example of using `uv run` with `langchain-model-profiles` to refresh profile data specifically for the Anthropic provider, specifying the data directory within the LangChain monorepo for the `langchain_anthropic` package.

```bash
uv run --with langchain-model-profiles --provider anthropic --data-dir langchain_anthropic/data
```

--------------------------------

### Runtime Context-Based Tool Selection by User Role

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Filter tools based on user permissions defined in runtime context (user role). Implements role-based access control with admin getting all tools, editors restricted from delete operations, and viewers limited to read-only tools. Uses middleware to evaluate user role and override available tools.

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@dataclass
class Context:
    user_role: str

@wrap_model_call
def context_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Runtime Context permissions."""
    # Read from Runtime Context: get user role
    user_role = request.runtime.context.user_role

    if user_role == "admin":
        # Admins get all tools
        pass
    elif user_role == "editor":
        # Editors can't delete
        tools = [t for t in request.tools if t.name != "delete_data"]
        request = request.override(tools=tools)
    else:
        # Viewers get read-only tools
        tools = [t for t in request.tools if t.name.startswith("read_")]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[read_data, write_data, delete_data],
    middleware=[context_based_tools],
    context_schema=Context
)
```

--------------------------------

### Inject User Writing Style from Store into LLM Prompt (LangChain Python)

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

This middleware function injects a user's writing style, retrieved from an `InMemoryStore`, into the LLM prompt. It constructs a style guide based on stored examples and appends it to the messages, helping the LLM draft responses consistent with the user's preferred style.

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@wrap_model_call
def inject_writing_style(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Inject user's email writing style from Store."""
    user_id = request.runtime.context.user_id

    # Read from Store: get user's writing style examples
    store = request.runtime.store
    writing_style = store.get(("writing_style",), user_id)

    if writing_style:
        style = writing_style.value
        # Build style guide from stored examples
        style_context = f"""Your writing style:
- Tone: {style.get('tone', 'professional')}
- Typical greeting: "{style.get('greeting', 'Hi')}"
- Typical sign-off: "{style.get('sign_off', 'Best')}"
- Example email you've written:
{style.get('example_email', '')}"""

        # Append at end - models pay more attention to final messages
        messages = [
            *request.messages,
            {"role": "user", "content": style_context}
        ]
        request = request.override(messages=messages)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[inject_writing_style],
    context_schema=Context,
    store=InMemoryStore()
)
```

--------------------------------

### Format Query Classifications in Python

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This snippet illustrates how to format the output of a query classification step. It expects a 'result' object containing a 'classifications' attribute and returns it as a dictionary, preparing the data for subsequent routing decisions in a multi-agent system.

```python
      return {"classifications": result.classifications}
```

--------------------------------

### Inject Compliance Rules Middleware in Python

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Implements a middleware function that appends jurisdiction-specific compliance rules to messages before sending them to the model. Rules are conditionally added based on industry type (healthcare, finance) and inserted at the end of the message list for higher model attention. Uses request.override() for transient updates without modifying saved state.

```python
rules.append("- Cannot share patient health information without authorization")
rules.append("- Must use secure, encrypted communication")
if industry == "finance":
    rules.append("- Cannot provide financial advice without proper disclaimers")

if rules:
    compliance_context = f"""Compliance requirements for {jurisdiction}:
{chr(10).join(rules)}"""

    # Append at end - models pay more attention to final messages
    messages = [
        *request.messages,
        {"role": "user", "content": compliance_context}
    ]
    request = request.override(messages=messages)

return handler(request)
```

--------------------------------

### Validate and Filter Responses with @after_model Middleware

Source: https://docs.langchain.com/oss/python/langchain/short-term-memory

Implements an @after_model middleware function that processes messages after model generation to remove responses containing sensitive keywords. The function receives the AgentState, checks the last message for stop words like 'password' or 'secret', and returns a dictionary with RemoveMessage instructions if sensitive content is detected, or None otherwise.

```python
from langchain.messages import RemoveMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import after_model
from langgraph.runtime import Runtime


@after_model
def validate_response(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove messages containing sensitive words."""
    STOP_WORDS = ["password", "secret"]
    last_message = state["messages"][-1]
    if any(word in last_message.content for word in STOP_WORDS):
        return {"messages": [RemoveMessage(id=last_message.id)]}
    return None

agent = create_agent(
    model="gpt-5-nano",
    tools=[],
    middleware=[validate_response],
    checkpointer=InMemorySaver(),
)
```

--------------------------------

### Define Structured Output Schema for Query Classification (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

This Python code defines the `ClassificationResult` Pydantic model, which serves as the structured output schema for the Language Model (LLM) used in query classification. It specifies that the LLM's output should be a list of `Classification` objects, each containing a 'source' and a 'query' for targeted sub-questions, enabling structured data extraction from the LLM response.

```python
class ClassificationResult(BaseModel):
    """Result of classifying a user query into agent-specific sub-questions."""
    classifications: list[Classification] = Field(
        description="List of agents to invoke with their targeted sub-questions"
    )
```

--------------------------------

### Initialize HuggingFace Chat Model in LangChain

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Demonstrates two methods to initialize a HuggingFace chat model in LangChain. It requires setting the `HUGGINGFACEHUB_API_TOKEN` environment variable and can specify model parameters like `temperature` and `max_tokens`.

```python
import os
from langchain.chat_models import init_chat_model

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."

model = init_chat_model(
    "microsoft/Phi-3-mini-4k-instruct",
    model_provider="huggingface",
    temperature=0.7,
    max_tokens=1024,
)
```

```python
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."

llm = HuggingFaceEndpoint(
```

--------------------------------

### Store-Aware System Prompt with User Preferences in Python

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Implements a dynamic system prompt that retrieves user preferences from long-term memory using the Store API. The prompt customizes instructions based on stored user communication style preferences. Requires defining a Context dataclass and initializing an InMemoryStore.

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id

    # Read from Store: get user preferences
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)

    base = "You are a helpful assistant."

    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."

    return base

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=InMemoryStore()
)
```

--------------------------------

### Initialize Google Gemini Chat Model with ChatGoogleGenerativeAI Class

Source: https://docs.langchain.com/oss/python/langchain/models

Initialize a Google Gemini model directly using the ChatGoogleGenerativeAI class from langchain_google_genai. Requires GOOGLE_API_KEY environment variable and provides direct access to Google-specific configuration options.

```python
import os
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "..."

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
```

--------------------------------

### Create PGVectorStore with PostgreSQL Engine

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Instantiates a PostgreSQL-based vector store using PGVectorStore with a specified table name and embedding service. Requires an active PostgreSQL engine connection and embedding service instance.

```python
vector_store = PGVectorStore.create_sync(
    engine=pg_engine,
    table_name='test_table',
    embedding_service=embedding
)
```

--------------------------------

### Setup and Initialize Pinecone Vector Store

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Installs langchain-pinecone package and initializes PineconeVectorStore with Pinecone client, API key, and index. Requires Pinecone account credentials and an existing index.

```shell
pip install -qU langchain-pinecone
```

```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

pc = Pinecone(api_key=...)
index = pc.Index(index_name)

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
```

--------------------------------

### State-based Output Format Selection

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Dynamically select response format based on conversation state by analyzing message count. Returns simple response formats for early conversations (less than 3 messages) and detailed formats as conversation progresses. Uses the @wrap_model_call decorator to intercept and modify model requests.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from pydantic import BaseModel, Field
from typing import Callable

class SimpleResponse(BaseModel):
    """Simple response for early conversation."""
    answer: str = Field(description="A brief answer")

class DetailedResponse(BaseModel):
    """Detailed response for established conversation."""
    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Explanation of reasoning")
    confidence: float = Field(description="Confidence score 0-1")

@wrap_model_call
def state_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Select output format based on State."""
    message_count = len(request.messages)

    if message_count < 3:
        request = request.override(response_format=SimpleResponse)
    else:
        request = request.override(response_format=DetailedResponse)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[...],
    middleware=[state_based_output]
)
```

--------------------------------

### Store-Based Tool Selection Using Feature Flags

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

Filter tools based on user preferences and feature flags stored in a data store. Retrieves enabled features for a specific user from InMemoryStore and restricts the toolset accordingly. Requires Context dataclass with user_id and supports persistent feature flag management.

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@wrap_model_call
def store_based_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Filter tools based on Store preferences."""
    user_id = request.runtime.context.user_id

    # Read from Store: get user's enabled features
    store = request.runtime.store
    feature_flags = store.get(("features",), user_id)

    if feature_flags:
        enabled_features = feature_flags.value.get("enabled_tools", [])
        # Only include tools that are enabled for this user
        tools = [t for t in request.tools if t.name in enabled_features]
        request = request.override(tools=tools)

    return handler(request)

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, analysis_tool, export_tool],
    middleware=[store_based_tools],
    context_schema=Context,
    store=InMemoryStore()
)
```

--------------------------------

### Multiple Decisions for Multiple Tool Calls - LangGraph Python

Source: https://docs.langchain.com/oss/python/langchain/human-in-the-loop

Handle multiple tool calls under review by providing a decision for each action in the same order as they appear in the interrupt. Supports mixed decision types (approve, edit, reject) for handling complex multi-action approval workflows.

```python
{
    "decisions": [
        {"type": "approve"},
        {
            "type": "edit",
            "edited_action": {
                "name": "tool_name",
                "args": {"param": "new_value"}
            }
        },
        {
            "type": "reject",
            "message": "This action is not allowed"
        }
    ]
}
```

--------------------------------

### Configure model to return log probabilities with logprobs parameter

Source: https://docs.langchain.com/oss/python/langchain/models

Initialize a chat model with logprobs=True to retrieve token-level log probabilities representing the likelihood of each generated token. The log probabilities are accessible via the response_metadata['logprobs'] property.

```python
model = init_chat_model(
    model="gpt-4o",
    model_provider="openai"
).bind(logprobs=True)

response = model.invoke("Why do parrots talk?")
print(response.response_metadata["logprobs"])
```

--------------------------------

### Cache Control with System Messages - Class-Based Middleware (Anthropic)

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Demonstrates cache control implementation for system messages using the class-based middleware approach with Anthropic models. This extends AgentMiddleware to add cacheable content blocks with ephemeral cache control directives.

```python
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.messages import SystemMessage
from typing import Callable


class CachedContextMiddleware(AgentMiddleware):
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # Always work with content blocks
        new_content = list(request.system_message.content_blocks) + [
            {
                "type": "text",
                "text": "Here is a large document to analyze:\n\n<document>...</document>",
                "cache_control": {"type": "ephemeral"}  # This content will be cached
            }
        ]

        new_system_message = SystemMessage(content=new_content)
        return handler(request.override(system_message=new_system_message))
```

--------------------------------

### Initialize Azure OpenAI Chat Model in LangChain

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Demonstrates two methods to initialize an Azure OpenAI chat model in LangChain. It requires setting `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, and `OPENAI_API_VERSION` environment variables, and specifying `azure_deployment`.

```python
import os
from langchain.chat_models import init_chat_model

os.environ["AZURE_OPENAI_API_KEY"] = "..."
os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

model = init_chat_model(
    "azure_openai:gpt-4.1",
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
)
```

```python
import os
from langchain_openai import AzureChatOpenAI

os.environ["AZURE_OPENAI_API_KEY"] = "..."
os.environ["AZURE_OPENAI_ENDPOINT"] = "..."
os.environ["OPENAI_API_VERSION"] = "2025-03-01-preview"

model = AzureChatOpenAI(
    model="gpt-4.1",
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
)
```

--------------------------------

### Complete reasoning output from model response in Python

Source: https://docs.langchain.com/oss/python/langchain/models

Retrieves and processes complete reasoning output from a model's full response. This approach invokes the model once and extracts all reasoning steps from the response content blocks, then joins them into a single string. Useful for obtaining the complete reasoning chain after generation is finished.

```python
response = model.invoke("Why do parrots have colorful feathers?")
reasoning_steps = [b for b in response.content_blocks if b["type"] == "reasoning"]
print(" ".join(step["reasoning"] for step in reasoning_steps))
```

--------------------------------

### Initialize OpenAI Chat Model in LangChain

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant

Demonstrates two methods to initialize an OpenAI chat model in LangChain: using the generic `init_chat_model` function or by directly instantiating `ChatOpenAI`. Both methods require setting the `OPENAI_API_KEY` environment variable for authentication.

```python
import os
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-..."

model = init_chat_model("gpt-4.1")
```

```python
import os
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "sk-..."

model = ChatOpenAI(model="gpt-4.1")
```

--------------------------------

### Access Model Profile and Capabilities

Source: https://docs.langchain.com/oss/python/langchain/models

Demonstrates how to access the model profile dictionary to discover supported features and capabilities. The profile attribute reveals model constraints and feature support including token limits, image input support, reasoning capabilities, and tool calling functionality (requires langchain>=1.1).

```python
model.profile
# {
#   "max_input_tokens": 400000,
#   "image_inputs": True,
#   "reasoning_output": True,
#   "tool_calling": True,
#   ...
```

--------------------------------

### Batch multiple requests to language model

Source: https://docs.langchain.com/oss/python/langchain/models

Execute multiple independent requests to a chat model in parallel using the batch() method. Returns all responses after processing completes. This is a client-side parallelization distinct from provider-specific batch APIs.

```python
responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])
for response in responses:
    print(response)
```

--------------------------------

### PIIMiddleware with Custom PII Detector

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Configure PIIMiddleware to use a custom function for detecting Personally Identifiable Information (PII) like SSN, allowing for advanced validation logic beyond simple regex patterns.

```APIDOC
## PIIMiddleware with Custom PII Detector

### Description
Configure PIIMiddleware to use a custom function for detecting Personally Identifiable Information (PII) like SSN, allowing for advanced validation logic beyond simple regex patterns.

### Method
Middleware Configuration

### Endpoint
`PIIMiddleware`

### Parameters
#### Configuration Options for `PIIMiddleware`
- **pii_type** (string) - Required - Type of PII to detect. Can be a built-in type (`email`, `credit_card`, `ip`, `mac_address`, `url`) or a custom type name.
- **strategy** (string) - Optional - How to handle detected PII. Options: `'block'` (Raise exception), `'redact'` (Replace with `[REDACTED_{PII_TYPE}]`), `'mask'` (Partially mask), `'hash'` (Replace with deterministic hash). Default: `redact`.
- **detector** (function | regex) - Optional - Custom detector function or regex pattern. If not provided, uses built-in detector for the PII type. The function must accept a string (`content`) and return a list of dictionaries with `'text'`, `'start'`, and `'end'` keys.
- **apply_to_input** (boolean) - Optional - Check user messages before model call. Default: `True`.
- **apply_to_output** (boolean) - Optional - Check AI messages after model call. Default: `False`.
- **apply_to_tool_results** (boolean) - Optional - Check tool result messages after execution. Default: `False`.

### Request Example
```python
# Custom detector function example
def detect_ssn(content: str) -> list[dict[str, str | int]]:
    """Detect SSN with validation.

    Returns a list of dictionaries with 'text', 'start', and 'end' keys.
    """
    import re
    matches = []
    pattern = r"\\d{3}-\\d{2}-\\d{4}"
    for match in re.finditer(pattern, content):
        ssn = match.group(0)
        # Validate: first 3 digits shouldn't be 000, 666, or 900-999
        first_three = int(ssn[:3])
        if first_three not in [0, 666] and not (900 <= first_three <= 999):
            matches.append({
                "text": ssn,
                "start": match.start(),
                "end": match.end(),
            })
    return matches

# Agent configuration using PIIMiddleware with the custom detector
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware

agent3 = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        PIIMiddleware(
            "ssn",
            detector=detect_ssn,
            strategy="hash",
        ),
    ],
)
```
```

--------------------------------

### Warranty Recording State Update

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Demonstrates the Command object returned by the record_warranty_status tool, which updates the warranty_status state and transitions the current_step to issue_classifier for the next turn.

```python
Command(update={
    "warranty_status": "in_warranty",
    "current_step": "issue_classifier"
})
```

--------------------------------

### Dynamic Tool Selection with Decorator Pattern

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Implements a decorator-based middleware using @wrap_model_call to dynamically select relevant tools based on request state and runtime context. Filters available tools before passing to the handler, reducing prompt complexity and improving model accuracy. Requires all tools to be registered upfront.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@wrap_model_call
def select_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Middleware to select relevant tools based on state/context."""
    # Select a small, relevant subset of tools based on state/context
    relevant_tools = select_relevant_tools(request.state, request.runtime)
    return handler(request.override(tools=relevant_tools))

agent = create_agent(
    model="gpt-4o",
    tools=all_tools,  # All available tools need to be registered upfront
    middleware=[select_tools],
)
```

--------------------------------

### Issue Classification State Update

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Shows the state update from the record_issue_type tool, which captures the issue classification and transitions to the resolution_specialist step with the warranty status formatted into the prompt.

```python
Command(update={
    "issue_type": "hardware",
    "current_step": "resolution_specialist"
})
```

--------------------------------

### Update Existing Model Profile (Python)

Source: https://docs.langchain.com/oss/python/langchain/models

Shows how to update an existing LangChain chat model's profile by merging new key-value pairs into the `profile` dictionary and creating a new model instance using `model_copy` to avoid mutating shared state of the original model.

```python
new_profile = model.profile | {"key": "value"}
model.model_copy(update={"profile": new_profile})
```

--------------------------------

### Install LangChain Python Library

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Installs the core `langchain` library, which is essential for building agent-based applications. Users can choose their preferred package manager: `pip`, `uv`, or `conda` to integrate LangChain into their Python projects.

```bash
pip install langchain
```

```bash
uv add langchain
```

```bash
conda install langchain -c conda-forge
```

--------------------------------

### Install Voyage AI Embeddings with LangChain

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Installs the `langchain-voyageai` package to enable integration with Voyage AI models. The provided content only includes the installation command, with the initialization code for the embeddings model being incomplete.

```shell
pip install -qU langchain-voyageai
```

--------------------------------

### Retry interceptor with exponential backoff

Source: https://docs.langchain.com/oss/python/langchain/mcp

Implement retry logic for failed tool calls with exponential backoff delay between attempts. The interceptor catches exceptions, calculates wait time using exponential backoff formula (2^attempt), and re-attempts up to max_retries times before raising the error.

```python
import asyncio

async def retry_interceptor(
    request: MCPToolCallRequest,
    handler,
    max_retries: int = 3,
    delay: float = 1.0,
):
    """Retry failed tool calls with exponential backoff."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return await handler(request)
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff
                print(f"Tool {request.name} failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
    raise last_error

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[retry_interceptor],
)
```

--------------------------------

### Classify User Query with Structured Output LLM (Python)

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

The `classify_query` function uses a Language Model (LLM) configured for structured output to analyze an incoming user query. It determines which knowledge bases (GitHub, Notion, Slack) are relevant and generates optimized sub-questions for each. The function's system prompt guides the LLM to return only relevant sources and tailored questions, facilitating effective routing.

```python
def classify_query(state: RouterState) -> dict:
    """Classify query and determine which agents to invoke."""
    structured_llm = router_llm.with_structured_output(ClassificationResult)  # [!code highlight]

    result = structured_llm.invoke([
        {
            "role": "system",
            "content": """Analyze this query and determine which knowledge bases to consult.
For each relevant source, generate a targeted sub-question optimized for that source.

Available sources:
- github: Code, API references, implementation details, issues, pull requests
- notion: Internal documentation, processes, policies, team wikis
- slack: Team discussions, informal knowledge sharing, recent conversations

Return ONLY the sources that are relevant to the query. Each source should have
a targeted sub-question optimized for that specific knowledge domain.

Example for "How do I authenticate API requests?":
- github: "What authentication code exists? Search for auth middleware, JWT handling"
- notion: "What authentication documentation exists? Look for API auth guides"
(slack omitted because it's not relevant for this technical question)"""
        },
        {"role": "user", "content": state["query"]}
    ])

    return {"classifications": result.classifications}
```

--------------------------------

### Refresh Model Profiles with CLI (Bash)

Source: https://docs.langchain.com/oss/python/langchain/models

Shows how to use the `langchain-profiles refresh` command to download the latest model data from `models.dev`, merge augmentations from `profile_augmentations.toml`, and update profile data for a specific provider, ensuring local profiles are up-to-date.

```bash
langchain-profiles refresh --provider <provider> --data-dir <data_dir>
```

--------------------------------

### Dynamic Model Selection with Wrap Decorator

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Wraps model calls to dynamically select between models based on conversation length. Uses decorator pattern with @wrap_model_call to intercept model requests and override the model before execution. Switches between gpt-4o for complex conversations and gpt-4o-mini for simpler ones.

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable


complex_model = init_chat_model("gpt-4o")
simple_model = init_chat_model("gpt-4o-mini")

@wrap_model_call
def dynamic_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # Use different model based on conversation length
    if len(request.messages) > 10:
        model = complex_model
    else:
        model = simple_model
    return handler(request.override(model=model))
```

--------------------------------

### Initial Message State Transition

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs-customer-support

Shows the state structure for the first turn of the conversation where the default current_step is set to warranty_collector and the middleware applies the warranty collector prompt and tools.

```python
{
    "messages": [HumanMessage("Hi, my phone screen is cracked")],
    "current_step": "warranty_collector"
}
```

--------------------------------

### Python tool to transfer control and update state

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/handoffs

Defines a LangChain tool `transfer_to_specialist` that returns a `Command` object. This command updates the workflow state by setting `current_step` to 'specialist' and includes a `ToolMessage` to inform the LLM about the transfer, ensuring the conversation history remains valid.

```python
from langchain.tools import tool
from langchain.messages import ToolMessage
from langgraph.types import Command

@tool
def transfer_to_specialist(runtime) -> Command:
    """Transfer to the specialist agent."""
    return Command(
        update={
            "messages": [
                ToolMessage(  # [!code highlight]
                    content="Transferred to specialist",
                    tool_call_id=runtime.tool_call_id  # [!code highlight]
                )
            ],
            "current_step": "specialist"  # Triggers behavior change
        }
    )
```

--------------------------------

### Configure LangSmith environment variables for tracing

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Set environment variables to enable LangSmith tracing for monitoring LLM calls and chain execution. Required after signing up at smith.langchain.com to inspect application behavior.

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```

--------------------------------

### Python ElicitResult Response Actions for MCP Client

Source: https://docs.langchain.com/oss/python/langchain/mcp

This snippet illustrates different `ElicitResult` actions (`accept`, `decline`, `cancel`) that an MCP client's elicitation callback can return. It shows how to provide data with an "accept" action or signal declination/cancellation.

```python
# Accept with data
ElicitResult(action="accept", content={"email": "user@example.com", "age": 25})

# Decline (user doesn't want to provide info)
ElicitResult(action="decline")

# Cancel (abort the operation)
ElicitResult(action="cancel")
```

--------------------------------

### Initialize Anthropic Claude Chat Model with init_chat_model

Source: https://docs.langchain.com/oss/python/langchain/models

Initialize an Anthropic Claude model using LangChain's init_chat_model function. Requires ANTHROPIC_API_KEY environment variable. Supports Claude Sonnet models and other Anthropic models.

```python
import os
from langchain.chat_models import init_chat_model

os.environ["ANTHROPIC_API_KEY"] = "sk-..."

model = init_chat_model("claude-sonnet-4-5-20250929")
```

--------------------------------

### Initialize Chat Model with Custom Profile - Python

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Create a custom chat model profile with manually specified token limits when model profile data is unavailable. This approach ensures SummarizationMiddleware's `fraction` conditions work correctly by providing explicit context size information.

```python
from langchain.chat_models import init_chat_model

custom_profile = {
    "max_input_tokens": 100_000,
}
model = init_chat_model("gpt-4o", profile=custom_profile)
```

--------------------------------

### Define Inventory Management Skill with Database Schema

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Defines a complete inventory management skill with database schema for products, warehouses, inventory tracking, and stock movements. Includes business logic for stock valuation and reorder point calculations.

```python
{
    "name": "inventory_management",
    "description": "Database schema and business logic for inventory tracking including products, warehouses, and stock levels.",
    "content": """# Inventory Management Schema

## Tables

### products
- product_id (PRIMARY KEY)
- product_name
- sku
- category
- unit_cost
- reorder_point (minimum stock level before reordering)
- discontinued (boolean)

### warehouses
- warehouse_id (PRIMARY KEY)
- warehouse_name
- location
- capacity

### inventory
- inventory_id (PRIMARY KEY)
- product_id (FOREIGN KEY -> products)
- warehouse_id (FOREIGN KEY -> warehouses)
- quantity_on_hand
- last_updated

### stock_movements
- movement_id (PRIMARY KEY)
- product_id (FOREIGN KEY -> products)
- warehouse_id (FOREIGN KEY -> warehouses)
- movement_type (inbound/outbound/transfer/adjustment)
- quantity (positive for inbound, negative for outbound)
- movement_date
- reference_number

## Business Logic

**Available stock**: quantity_on_hand from inventory table where quantity_on_hand > 0

**Products needing reorder**: Products where total quantity_on_hand across all warehouses is less than or equal to the product's reorder_point

**Active products only**: Exclude products where discontinued = true unless specifically analyzing discontinued items

**Stock valuation**: quantity_on_hand * unit_cost for each product

## Example Query

SELECT
    p.product_id,
    p.product_name,
    p.reorder_point,
    SUM(i.quantity_on_hand) as total_stock,
    p.unit_cost,
    (p.reorder_point - SUM(i.quantity_on_hand)) as units_to_reorder
FROM products p
JOIN inventory i ON p.product_id = i.product_id
WHERE p.discontinued = false
GROUP BY p.product_id, p.product_name, p.reorder_point, p.unit_cost
HAVING SUM(i.quantity_on_hand) <= p.reorder_point
ORDER BY units_to_reorder DESC;"""
}
```

--------------------------------

### Build RAG Pipeline with LangGraph StateGraph and Nodes

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow

Constructs a complete RAG workflow using LangGraph that integrates query rewriting, vector-based document retrieval, and agent-based reasoning. The workflow uses TypedDict State to manage data flow between nodes, OpenAI embeddings for vector storage, and structured output for query transformation. This example includes a WNBA knowledge base with team rosters, game results, and player statistics.

```python
from typing import TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

class State(TypedDict):
    question: str
    rewritten_query: str
    documents: list[str]
    answer: str

# WNBA knowledge base with rosters, game results, and player stats
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_texts([
    # Rosters
    "New York Liberty 2024 roster: Breanna Stewart, Sabrina Ionescu, Jonquel Jones, Courtney Vandersloot.",
    "Las Vegas Aces 2024 roster: A'ja Wilson, Kelsey Plum, Jackie Young, Chelsea Gray.",
    "Indiana Fever 2024 roster: Caitlin Clark, Aliyah Boston, Kelsey Mitchell, NaLyssa Smith.",
    # Game results
    "2024 WNBA Finals: New York Liberty defeated Minnesota Lynx 3-2 to win the championship.",
    "June 15, 2024: Indiana Fever 85, Chicago Sky 79. Caitlin Clark had 23 points and 8 assists.",
    "August 20, 2024: Las Vegas Aces 92, Phoenix Mercury 84. A'ja Wilson scored 35 points.",
    # Player stats
    "A'ja Wilson 2024 season stats: 26.9 PPG, 11.9 RPG, 2.6 BPG. Won MVP award.",
    "Caitlin Clark 2024 rookie stats: 19.2 PPG, 8.4 APG, 5.7 RPG. Won Rookie of the Year.",
    "Breanna Stewart 2024 stats: 20.4 PPG, 8.5 RPG, 3.5 APG.",
])
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

@tool
def get_latest_news(query: str) -> str:
    """Get the latest WNBA news and updates."""
    # Your news API here
    return "Latest: The WNBA announced expanded playoff format for 2025..."

agent = create_agent(
    model="openai:gpt-4o",
    tools=[get_latest_news],
)

model = ChatOpenAI(model="gpt-4o")

class RewrittenQuery(BaseModel):
    query: str

def rewrite_query(state: State) -> dict:
    """Rewrite the user query for better retrieval."""
    system_prompt = """Rewrite this query to retrieve relevant WNBA information.
The knowledge base contains: team rosters, game results with scores, and player statistics (PPG, RPG, APG).
Focus on specific player names, team names, or stat categories mentioned."""
    response = model.with_structured_output(RewrittenQuery).invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": state["question"]}
    ])
    return {"rewritten_query": response.query}

def retrieve(state: State) -> dict:
    """Retrieve documents based on the rewritten query."""
    docs = retriever.invoke(state["rewritten_query"])
    return {"documents": [doc.page_content for doc in docs]}

def call_agent(state: State) -> dict:
    """Generate answer using retrieved context."""
    context = "\n\n".join(state["documents"])
    prompt = f"Context:\n{context}\n\nQuestion: {state['question']}"
    response = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return {"answer": response["messages"][-1].content_blocks}

workflow = (
    StateGraph(State)
    .add_node("rewrite", rewrite_query)
    .add_node("retrieve", retrieve)
    .add_node("agent", call_agent)
    .add_edge(START, "rewrite")
    .add_edge("rewrite", "retrieve")
    .add_edge("retrieve", "agent")
    .add_edge("agent", END)
    .compile()
)

result = workflow.invoke({"question": "Who won the 2024 WNBA Championship?"})
print(result["answer"])
```

--------------------------------

### Initialize Chroma Vector Store

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Creates a Chroma vector store with optional persistent storage to disk. Allows specifying collection name and local persistence directory for data durability. Perfect for lightweight vector search with local storage capabilities.

```python
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
```

--------------------------------

### LLMToolEmulator - Complete Example

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Comprehensive example demonstrating all configuration options for the LLMToolEmulator middleware including emulating all tools, specific tools, and custom model selection.

```APIDOC
## LLMToolEmulator - Complete Example

### Description
Full working example showing how to configure LLMToolEmulator with different options including emulating all tools, specific tools, and using a custom model.

### Complete Example
```python
from langchain.agents import create_agent
from langchain.agents.middleware import LLMToolEmulator
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Weather in {location}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return "Email sent"

# Emulate all tools (default behavior)
agent = create_agent(
    model="gpt-4o",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator()],
)

# Emulate specific tools only
agent2 = create_agent(
    model="gpt-4o",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator(tools=["get_weather"])],
)

# Use custom model for emulation
agent4 = create_agent(
    model="gpt-4o",
    tools=[get_weather, send_email],
    middleware=[LLMToolEmulator(model="claude-sonnet-4-5-20250929")],
)
```

### Configuration Patterns
- **Emulate all tools**: LLMToolEmulator() with no parameters
- **Emulate specific tools**: LLMToolEmulator(tools=["tool_name1", "tool_name2"])
- **Custom model**: LLMToolEmulator(model="model_identifier")
- **Combine options**: LLMToolEmulator(tools=["tool_name"], model="model_identifier")
```

--------------------------------

### ReasoningContentBlock Python Dictionary Structure

Source: https://docs.langchain.com/oss/python/langchain/messages

Model reasoning steps content block representation. Contains required type field set to 'reasoning', optional reasoning string for the reasoning content, and extras object for provider-specific data like signatures.

```python
{
    "type": "reasoning",
    "reasoning": "The user is asking about...",
    "extras": {"signature": "abc123"}
}
```

--------------------------------

### ContextEditingMiddleware Configuration

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Configures the ContextEditingMiddleware to manage conversation context by clearing older tool call outputs when token limits are reached, while preserving recent results. This helps keep context windows manageable in long conversations with many tool calls.

```APIDOC
## ContextEditingMiddleware Configuration

### Description
Manages conversation context by clearing older tool call outputs when token limits are reached, while preserving recent results. This helps keep context windows manageable in long conversations with many tool calls.

### Configuration Parameters
- **edits** (list[ContextEdit]) - Required - List of `ContextEdit` strategies to apply. Default: `[ClearToolUsesEdit()]`.
- **token_count_method** (string) - Optional - Token counting method. Options: `'approximate'` or `'model'`. Default: `'approximate'`.

### Request Example
```python
from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ContextEditingMiddleware(
            edits=[
                ClearToolUsesEdit(
                    trigger=100000,
                    keep=3,
                ),
            ],
        ),
    ],
)
```
```

--------------------------------

### Configure LangChain Model Retry Middleware with Dynamic Exception Filtering (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Demonstrates using a custom callable function with `retry_on` to dynamically decide whether to retry an exception. This approach provides advanced control, allowing complex logic to evaluate the exception (e.g., checking HTTP status codes) before initiating a retry.

```python
def should_retry(error: Exception) -> bool:
    # Only retry on rate limit errors
    if isinstance(error, TimeoutError):
        return True
    # Or check for specific HTTP status codes
    if hasattr(error, "status_code"):
        return error.status_code in (429, 503)
    return False

retry_with_filter = ModelRetryMiddleware(
    max_retries=3,
    retry_on=should_retry,
)
```

--------------------------------

### Install core LangChain and LangGraph packages

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

Installs the necessary `langchain` and `langgraph` libraries using pip, uv, or conda for building multi-agent systems and workflow orchestration. These packages are fundamental for implementing the router pattern described in this tutorial.

```bash
pip install langchain langgraph
```

```bash
uv add langchain langgraph
```

```bash
conda install langchain langgraph -c conda-forge
```

--------------------------------

### Create HumanMessage with Provider-Native Content in Python

Source: https://docs.langchain.com/oss/python/langchain/messages

Demonstrates how to construct a `HumanMessage` in LangChain using a provider-native list format for its `content` property. This example includes both text and an image URL, showcasing multi-modal input. The structure of this content might vary based on the specific LLM provider being used.

```python
human_message = HumanMessage(content=[
    {"type": "text", "text": "Hello, how are you?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
])
```

--------------------------------

### Initialize ChatHuggingFace with LLM Configuration

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/skills-sql-assistant

Configures a ChatHuggingFace model with temperature and max_length parameters for controlled text generation. The temperature parameter controls randomness while max_length limits output tokens.

```python
model = ChatHuggingFace(
    llm=HuggingFacePipeline(
        model_id="gpt2",
        task="text-generation",
        model_kwargs={
            "temperature": 0.7,
            "max_length": 1024,
        }
    )
)
```

--------------------------------

### Install Vector Store Dependencies

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Shell commands for installing required packages for various vector store implementations. Each implementation has specific dependencies that must be installed before use.

```shell
pip install -U "langchain-core"
```

```shell
pip install -qU boto3
```

```shell
pip install -U "langchain-astradb"
```

```shell
pip install -qU langchain-chroma
```

```shell
pip install -qU langchain-community faiss-cpu
```

```shell
pip install -qU langchain-milvus
```

```shell
pip install -qU langchain-mongodb
```

```shell
pip install -qU langchain-postgres
```

--------------------------------

### Create sample Document objects with metadata

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Instantiate LangChain Document objects with page_content and metadata attributes. Documents represent units of text with associated metadata such as source information, useful for organizing retrieved information.

```python
from langchain_core.documents import Document

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]
```

--------------------------------

### Add Documents to Vector Store

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Indexes documents into the vector store using the add_documents method, which processes and stores document embeddings. Returns list of document IDs for reference and tracking.

```python
ids = vector_store.add_documents(documents=all_splits)
```

--------------------------------

### Define Movie Schema using TypedDict in Python

Source: https://docs.langchain.com/oss/python/langchain/models

This code defines a movie schema using Python's `TypedDict`. This is an alternative to Pydantic and is useful when runtime validation isn't needed. It showcases how to define the movie details (title, year, director, rating) using `TypedDict` and `Annotated` for descriptions.

```python
from typing_extensions import TypedDict, Annotated

class MovieDict(TypedDict):
    """A movie with details."""
    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "The year the movie was released"]
    director: Annotated[str, ..., "The director of the movie"]
    rating: Annotated[float, ..., "The movie's rating out of 10"]

model_with_structure = model.with_structured_output(MovieDict)
response = model_with_structure.invoke("Provide details about the movie Inception")
print(response)  # {'title': 'Inception', 'year': 2010, 'director': 'Christopher Nolan', 'rating': 8.8}
```

--------------------------------

### Use LangChain Configurable Models with Declarative Tool Binding

Source: https://docs.langchain.com/oss/python/langchain/models

This example demonstrates combining LangChain configurable models with declarative operations, specifically 'bind_tools'. It shows defining Pydantic models for tools, binding them to a configurable chat model, and then invoking the model to utilize these tools with different underlying LLMs (e.g., GPT-4.1-mini, Claude) specified at runtime.

```python
from pydantic import BaseModel, Field


class GetWeather(BaseModel):
    """Get the current weather in a given location"""

        location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


class GetPopulation(BaseModel):
    """Get the current population in a given location"""

        location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


model = init_chat_model(temperature=0)
model_with_tools = model.bind_tools([GetWeather, GetPopulation])

model_with_tools.invoke(
    "what's bigger in 2024 LA or NYC", config={"configurable": {"model": "gpt-4.1-mini"}}
).tool_calls
```

```python
model_with_tools.invoke(
    "what's bigger in 2024 LA or NYC",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},
).tool_calls
```

--------------------------------

### Define Advanced LangChain Tool Input Schema with Pydantic or JSON

Source: https://docs.langchain.com/oss/python/langchain/tools

Specify complex input schemas for LangChain tools using either Pydantic models or raw JSON schemas. This method offers granular control over input validation, default values, and data types, ensuring robust tool execution.

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherInput(BaseModel):
    """Input for weather queries."""
    location: str = Field(description="City name or coordinates")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="Temperature unit preference"
    )
    include_forecast: bool = Field(
        default=False,
        description="Include 5-day forecast"
    )

@tool(args_schema=WeatherInput)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

```python
weather_schema = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location", "units", "include_forecast"]
}

@tool(args_schema=weather_schema)
def get_weather(location: str, units: str = "celsius", include_forecast: bool = False) -> str:
    """Get current weather and optional forecast."""
    temp = 22 if units == "celsius" else 72
    result = f"Current weather in {location}: {temp} degrees {units[0].upper()}"
    if include_forecast:
        result += "\nNext 5 days: Sunny"
    return result
```

--------------------------------

### Implement Rate Limiting for MCP Tool Calls with Python Interceptors

Source: https://docs.langchain.com/oss/python/langchain/mcp

This interceptor prevents excessive calls to expensive tools by checking a rate limit before execution. If the tool is rate-limited, it returns a ToolMessage indicating the error; otherwise, it proceeds with the call and logs its success.

```python
from langchain.messages import ToolMessage

async def rate_limit_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Rate limit expensive MCP tool calls."""
    runtime = request.runtime
    tool_call_id = runtime.tool_call_id  # [!code highlight]

    # Check rate limit (simplified example)
    if is_rate_limited(request.name):
        return ToolMessage(
            content="Rate limit exceeded. Please try again later.",
            tool_call_id=tool_call_id,  # [!code highlight]
        )

    result = await handler(request)

    # Log successful tool call
    log_tool_execution(tool_call_id, request.name, success=True)

    return result

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[rate_limit_interceptor],
)
```

--------------------------------

### Accumulate Streamed Tool Calls in Python

Source: https://docs.langchain.com/oss/python/langchain/models

Shows how to accumulate tool call chunks during streaming to build complete tool calls. Chunks are progressively combined using addition operations to reconstruct the full tool call information.

```python
gathered = None
for chunk in model_with_tools.stream("What's the weather in Boston?"):
    gathered = chunk if gathered is None else gathered + chunk
    print(gathered.tool_calls)
```

--------------------------------

### Initialize AWS Bedrock Chat Model with ChatBedrock Class

Source: https://docs.langchain.com/oss/python/langchain/models

Initialize an AWS Bedrock model directly using the ChatBedrock class from langchain_aws with Claude 3.5 Sonnet. Requires AWS credentials to be configured per AWS Bedrock setup documentation.

```python
from langchain_aws import ChatBedrock

model = ChatBedrock(model="anthropic.claude-3-5-sonnet-20240620-v1:0")
```

--------------------------------

### Create a Basic Logging Interceptor for MCP Tool Calls in Python

Source: https://docs.langchain.com/oss/python/langchain/mcp

This example illustrates the fundamental structure of an interceptor, demonstrating how to log tool calls before and after their execution. It highlights how interceptors act as middleware, allowing observation or modification of tool interactions.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest

async def logging_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Log tool calls before and after execution."""
    print(f"Calling tool: {request.name} with args: {request.args}")
    result = await handler(request)
    print(f"Tool {request.name} returned: {result}")
    return result

client = MultiServerMCPClient(
    {"math": {"transport": "stdio", "command": "python", "args": ["/path/to/server.py"]}},
    tool_interceptors=[logging_interceptor],  # [!code highlight]
)
```

--------------------------------

### Initialize PGVector Vector Store

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Creates PostgreSQL vector store using psycopg driver with embeddings and collection name. Requires PostgreSQL connection string with pgvector extension enabled for vector operations.

```python
from langchain_postgres import PGVector

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="my_docs",
    connection="postgresql+psycopg://...",
)
```

--------------------------------

### Define Nested Structures with TypedDict

Source: https://docs.langchain.com/oss/python/langchain/models

Provides an alternative approach to nested structures using TypedDict from typing_extensions instead of Pydantic. This method offers a lightweight alternative for defining structured outputs with type hints and annotations, including optional fields with descriptions.

```python
from typing_extensions import Annotated, TypedDict

class Actor(TypedDict):
    name: str
    role: str

class MovieDetails(TypedDict):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: Annotated[float | None, ..., "Budget in millions USD"]

model_with_structure = model.with_structured_output(MovieDetails)
```

--------------------------------

### Define Detailed Persona with System Message in LangChain (Python)

Source: https://docs.langchain.com/oss/python/langchain/messages

Illustrates using a `SystemMessage` to define a detailed persona and guidelines for a LangChain model. This includes setting the model's role, expertise, and expected response characteristics for more nuanced interactions.

```python
from langchain.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
```

--------------------------------

### Initialize Google Vertex AI Embeddings with LangChain

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Installs the `langchain-google-vertexai` package and initializes the VertexAIEmbeddings model. This setup allows for using Google Cloud's Vertex AI for text embeddings.

```shell
pip install -qU langchain-google-vertexai
```

```python
from langchain_google_vertexai import VertexAIEmbeddings

embeddings = VertexAIEmbeddings(model="text-embedding-005")
```

--------------------------------

### Stream Semantic Events from LangChain Chat Models

Source: https://docs.langchain.com/oss/python/langchain/models

Demonstrates using astream_events() to stream semantic events from chat models with event type filtering. This approach simplifies filtering based on event types and aggregates the full message in the background. Includes start, stream, and end event handlers that capture input, individual tokens, and the complete message output.

```python
async for event in model.astream_events("Hello"):

    if event["event"] == "on_chat_model_start":
        print(f"Input: {event['data']['input']}")

    elif event["event"] == "on_chat_model_stream":
        print(f"Token: {event['data']['chunk'].text}")

    elif event["event"] == "on_chat_model_end":
        print(f"Full message: {event['data']['output'].text}")

    else:
        pass
```

--------------------------------

### Append Structured Content via Interceptor - Python

Source: https://docs.langchain.com/oss/python/langchain/mcp

Uses an MCP tool interceptor to automatically append structured content from artifact to tool messages for visibility in conversation history. The interceptor handler processes the request, extracts structured content, and converts it to TextContent format before returning the modified result.

```python
import json

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest
from mcp.types import TextContent

async def append_structured_content(request: MCPToolCallRequest, handler):
    """Append structured content from artifact to tool message."""
    result = await handler(request)
    if result.structuredContent:
        result.content += [
            TextContent(type="text", text=json.dumps(result.structuredContent)),
        ]
    return result

client = MultiServerMCPClient({...}, tool_interceptors=[append_structured_content])
```

--------------------------------

### Initialize FAISS Vector Store

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Sets up FAISS vector store with in-memory document storage and flat L2 index for similarity search. Requires computing embedding dimension from embedding model and initializing FAISS index structure.

```python
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
```

--------------------------------

### Set up VoyageAI Embeddings in LangChain

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Installs the LangChain VoyageAI integration package and initializes the VoyageAIEmbeddings model. This setup includes a check for the 'VOYAGE_API_KEY' environment variable, prompting the user for it if not found. The 'model' parameter specifies which Voyage AI embedding model to use.

```shell
pip install -qU langchain-voyageai
```

```python
import getpass
import os

if not os.environ.get("VOYAGE_API_KEY"):
    os.environ["VOYAGE_API_KEY"] = getpass.getpass("Enter API key for Voyage AI: ")

from langchain_voyageai import VoyageAIEmbeddings

embeddings = VoyageAIEmbeddings(model="voyage-3")
```

--------------------------------

### Invoke Model for Multimodal Output (Python)

Source: https://docs.langchain.com/oss/python/langchain/models

Demonstrates how to invoke a LangChain chat model to generate a multimodal response, such as an image. The example shows accessing the structured content blocks from the `AIMessage` result, which can include text and other media types like base64-encoded images.

```python
response = model.invoke("Create a picture of a cat")
print(response.content_blocks)
# [
#     {"type": "text", "text": "Here's a picture of a cat"},
#     {"type": "image", "base64": "...", "mime_type": "image/jpeg"},
# ]
```

--------------------------------

### Composing multiple interceptors in onion order

Source: https://docs.langchain.com/oss/python/langchain/mcp

Demonstrates how multiple interceptors compose in 'onion' order where the first interceptor is the outermost layer. The execution flows through each interceptor before reaching the tool, then returns through them in reverse order. Configure via tool_interceptors parameter.

```python
async def outer_interceptor(request, handler):
    print("outer: before")
    result = await handler(request)
    print("outer: after")
    return result

async def inner_interceptor(request, handler):
    print("inner: before")
    result = await handler(request)
    print("inner: after")
    return result

client = MultiServerMCPClient(
    {...},
    tool_interceptors=[outer_interceptor, inner_interceptor],
)

# Execution order:
# outer: before -> inner: before -> tool execution -> inner: after -> outer: after
```

--------------------------------

### Define a Custom SSN Detector Function in Python

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This Python function demonstrates how to create a custom detector for Social Security Numbers (SSN) with built-in validation. It uses regular expressions to find patterns and then applies additional logic to exclude invalid SSN ranges (000, 666, 900-999). The function returns a list of dictionaries, each containing the detected text, and its start and end positions.

```python
def detect_ssn(content: str) -> list[dict[str, str | int]]:
    """Detect SSN with validation.

    Returns a list of dictionaries with 'text', 'start', and 'end' keys.
    """
    import re
    matches = []
    pattern = r"\d{3}-\d{2}-\d{4}"
    for match in re.finditer(pattern, content):
        ssn = match.group(0)
        # Validate: first 3 digits shouldn't be 000, 666, or 900-999
        first_three = int(ssn[:3])
        if first_three not in [0, 666] and not (900 <= first_three <= 999):
            matches.append({
                "text": ssn,
                "start": match.start(),
                "end": match.end(),
            })
    return matches
```

--------------------------------

### Force Use of Any Tool in LangChain Python

Source: https://docs.langchain.com/oss/python/langchain/models

Configures the model to force selection of any available tool from the bound tools list using the tool_choice parameter set to 'any'. This ensures the model will always attempt to use a tool regardless of whether it would naturally choose to do so.

```python
model_with_tools = model.bind_tools([tool_1], tool_choice="any")
```

--------------------------------

### Define Signature for Custom LangChain Detector Functions in Python

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

This snippet illustrates the required function signature for any custom detector used with LangChain's PIIMiddleware. The detector function must accept a string `content` and return a list of dictionaries, where each dictionary specifies the `text` of the match, its `start` index, and its `end` index.

```python
def detector(content: str) -> list[dict[str, str | int]]:
    return [
        {"text": "matched_text", "start": 0, "end": 12},
        # ... more matches
    ]
```

--------------------------------

### Event Stream Output Example

Source: https://docs.langchain.com/oss/python/langchain/models

Shows the expected output format when streaming events from a chat model. Displays sequential token output followed by the complete aggregated message, demonstrating real-time token delivery and final result.

```txt
Input: Hello
Token: Hi
Token:  there
Token: !
Token:  How
Token:  can
Token:  I
...
Full message: Hi there! How can I help today?
```

--------------------------------

### Audio Input - Base64 and File ID in Python

Source: https://docs.langchain.com/oss/python/langchain/messages

Construct messages with audio content using base64-encoded data with appropriate MIME type (e.g., audio/wav) or provider-managed file identifiers. Audio data is paired with text prompts in the message content structure.

```python
# From base64 data
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this audio."},
        {
            "type": "audio",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "audio/wav",
        },
    ]
}

# From provider-managed File ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this audio."},
        {"type": "audio", "file_id": "file-abc123"},
    ]
}
```

--------------------------------

### Define Domain-Specific Tools using LangChain Tool Decorator

Source: https://docs.langchain.com/oss/python/langchain/multi-agent/router-knowledge-base

Create seven stub tools across three knowledge verticals: GitHub (search_code, search_issues, search_prs), Notion (search_notion, get_page), and Slack (search_slack, get_thread). Each tool returns mock data simulating real API responses.

```python
from langchain.tools import tool


@tool
def search_code(query: str, repo: str = "main") -> str:
    """Search code in GitHub repositories."""
    return f"Found code matching '{query}' in {repo}: authentication middleware in src/auth.py"


@tool
def search_issues(query: str) -> str:
    """Search GitHub issues and pull requests."""
    return f"Found 3 issues matching '{query}': #142 (API auth docs), #89 (OAuth flow), #203 (token refresh)"


@tool
def search_prs(query: str) -> str:
    """Search pull requests for implementation details."""
    return f"PR #156 added JWT authentication, PR #178 updated OAuth scopes"


@tool
def search_notion(query: str) -> str:
    """Search Notion workspace for documentation."""
    return f"Found documentation: 'API Authentication Guide' - covers OAuth2 flow, API keys, and JWT tokens"


@tool
def get_page(page_id: str) -> str:
    """Get a specific Notion page by ID."""
    return f"Page content: Step-by-step authentication setup instructions"


@tool
def search_slack(query: str) -> str:
    """Search Slack messages and threads."""
    return f"Found discussion in #engineering: 'Use Bearer tokens for API auth, see docs for refresh flow'"


@tool
def get_thread(thread_id: str) -> str:
    """Get a specific Slack thread."""
    return f"Thread discusses best practices for API key rotation"
```

--------------------------------

### Execute Parallel Tool Calls in Python

Source: https://docs.langchain.com/oss/python/langchain/models

Shows how to invoke a model with tools to make multiple parallel tool calls and then execute all tools simultaneously. The model intelligently determines when parallel execution is appropriate based on operation independence. Supports async execution for true parallelism.

```python
model_with_tools = model.bind_tools([get_weather])

response = model_with_tools.invoke(
    "What's the weather in Boston and Tokyo?"
)


# The model may generate multiple tool calls
print(response.tool_calls)
# [
#   {'name': 'get_weather', 'args': {'location': 'Boston'}, 'id': 'call_1'},
#   {'name': 'get_weather', 'args': {'location': 'Tokyo'}, 'id': 'call_2'},
# ]


# Execute all tools (can be done in parallel with async)
results = []
for tool_call in response.tool_calls:
    if tool_call['name'] == 'get_weather':
        result = get_weather.invoke(tool_call)
    ...
    results.append(result)
```

--------------------------------

### Create LangChain VectorStore Retriever with as_retriever (Python)

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

This Python example illustrates how to instantiate a LangChain `VectorStoreRetriever` directly from a `vector_store` object using its `as_retriever` method. It configures the retriever for `similarity` search and specifies `k` as a search argument, then demonstrates performing batch retrieval.

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
```

--------------------------------

### Modify Tool Call Arguments with Python Interceptors

Source: https://docs.langchain.com/oss/python/langchain/mcp

This interceptor demonstrates how to alter the arguments of a tool call before it's executed. It uses `request.override()` to create an immutable modified request, ensuring the original request remains unchanged, and then passes this modified request to the next handler.

```python
async def double_args_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Double all numeric arguments before execution."""
    modified_args = {k: v * 2 for k, v in request.args.items()}
    modified_request = request.override(args=modified_args)  # [!code highlight]
    return await handler(modified_request)
```

--------------------------------

### Parse Anthropic AIMessage Content to Standard Blocks in Python

Source: https://docs.langchain.com/oss/python/langchain/messages

Shows how an `AIMessage` containing Anthropic-specific `thinking` content is transformed into LangChain's standard `ReasoningContentBlock` representation. By accessing the `content_blocks` property, the provider-native format is lazily parsed into a consistent, type-safe structure, along with regular text content, resulting in a list like `[{'type': 'reasoning', 'reasoning': '...', 'extras': {'signature': 'WaUjzkyp...'}}, {'type': 'text', 'text': '...'}]`.

```python
from langchain.messages import AIMessage

message = AIMessage(
    content=[
        {"type": "thinking", "thinking": "...", "signature": "WaUjzkyp..."},
        {"type": "text", "text": "..."},
    ],
    response_metadata={"model_provider": "anthropic"}
)
message.content_blocks
```

--------------------------------

### Reconstruct AIMessage from Streamed Chunks (Python)

Source: https://docs.langchain.com/oss/python/langchain/models

Illustrates how to accumulate `AIMessageChunk` objects returned by `stream()` into a complete `AIMessage` object. Each chunk can be added to the previous one to gradually build the full response text and access final `content_blocks`.

```python
full = None  # None | AIMessageChunk
for chunk in model.stream("What color is the sky?"):
    full = chunk if full is None else full + chunk
    print(full.text)

# The
# The sky
# The sky is
# The sky is typically
# The sky is typically blue
# ...

print(full.content_blocks)
```

--------------------------------

### Create Basic Configurable LangChain Chat Model

Source: https://docs.langchain.com/oss/python/langchain/models

This example demonstrates initializing a LangChain chat model with default configurable fields. It shows how to invoke the model, overriding the underlying LLM (e.g., GPT-5-Nano, Claude) at runtime by passing different 'model' values within the 'config' parameter.

```python
from langchain.chat_models import init_chat_model

configurable_model = init_chat_model(temperature=0)

configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "gpt-5-nano"}},  # Run with GPT-5-Nano
)
configurable_model.invoke(
    "what's your name",
    config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},  # Run with Claude
)
```

--------------------------------

### Install LangChain Model Profiles CLI (Bash)

Source: https://docs.langchain.com/oss/python/langchain/models

Provides the command to install the `langchain-model-profiles` CLI tool, which is used for refreshing and managing model profile data from `models.dev` and local augmentations within LangChain projects.

```bash
pip install langchain-model-profiles
```

--------------------------------

### Process Interrupt Decisions with Edit and Accept Actions

Source: https://docs.langchain.com/oss/python/langchain/streaming

Handles user decisions for pending interrupts by creating a decision list that specifies actions to take for each interrupt. This example demonstrates editing one tool call (changing Boston to Boston, U.K.) while accepting another, maintaining the order of decisions to match collected interrupts.

```python
def _get_interrupt_decisions(interrupt: Interrupt) -> list[dict]:
    return [
        {
            "type": "edit",
            "edited_action": {
                "name": "get_weather",
                "args": {"city": "Boston, U.K."},
            },
        },
    ]
```

--------------------------------

### Configure LangChain Model Retry Middleware for Specific Exceptions (Python)

Source: https://docs.langchain.com/oss/python/langchain/middleware/built-in

Shows how to configure `ModelRetryMiddleware` to retry only on specified exception types, such as `TimeoutError` or `ConnectionError`. This allows for fine-grained control over which errors trigger a retry, preventing unnecessary retries for non-transient issues.

```python
class TimeoutError(Exception):
    """Custom exception for timeout errors."""
    pass

class ConnectionError(Exception):
    """Custom exception for connection errors."""
    pass

retry = ModelRetryMiddleware(
    max_retries=4,
    retry_on=(TimeoutError, ConnectionError),
    backoff_factor=1.5,
)
```

--------------------------------

### Stream reasoning output from model in Python

Source: https://docs.langchain.com/oss/python/langchain/models

Streams reasoning output from a model by filtering content blocks for reasoning type. This allows real-time access to the model's step-by-step reasoning process as it generates a response. The code extracts reasoning steps from each chunk and prints them if available, otherwise prints the text content.

```python
for chunk in model.stream("Why do parrots have colorful feathers?"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    print(reasoning_steps if reasoning_steps else chunk.text)
```

--------------------------------

### Process Interrupt Decisions with Conditional Filtering

Source: https://docs.langchain.com/oss/python/langchain/streaming

Creates a function to process interrupt action requests, applying conditional logic to either edit or approve actions based on request description content. This processes a list of action requests from an interrupt, building a decision list that differentiates between requests containing 'boston' (which get edited with modified arguments) and other requests (which get approved).

```python
if "boston" in request["description"].lower()
else {"type": "approve"}
for request in interrupt.value["action_requests"]
]

decisions = {}
for interrupt in interrupts:
    decisions[interrupt.id] = {
        "decisions": _get_interrupt_decisions(interrupt)
    }
```

--------------------------------

### Node-style Hook with Decorator - Message Limit Validation

Source: https://docs.langchain.com/oss/python/langchain/middleware/custom

Implements before_model and after_model decorators to validate message limits and log model responses. The before_model hook can jump to the 'end' state when the message limit is reached, while after_model logs each model response for debugging.

```python
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any


@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 50:
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
    return None

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None
```

--------------------------------

### Save User Preferences to Persistent Store in LangChain

Source: https://docs.langchain.com/oss/python/langchain/context-engineering

This Python example demonstrates a LangChain tool (`save_preference`) that saves user-specific data to a persistent `Store`. It utilizes `ToolRuntime` with a `Context` schema to access user ID, retrieves and updates existing preferences from the `InMemoryStore`, and then writes the merged data back, enabling data persistence across sessions.

```python
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime
from langchain.agents import create_agent
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@tool
def save_preference(
    preference_key: str,
    preference_value: str,
    runtime: ToolRuntime[Context]
) -> str:
    """Save user preference to Store."""
    user_id = runtime.context.user_id

    # Read existing preferences
    store = runtime.store
    existing_prefs = store.get(("preferences",), user_id)

    # Merge with new preference
    prefs = existing_prefs.value if existing_prefs else {}
    prefs[preference_key] = preference_value

    # Write to Store: save updated preferences
    store.put(("preferences",), user_id, prefs)

    return f"Saved preference: {preference_key} = {preference_value}"

agent = create_agent(
    model="gpt-4o",
    tools=[save_preference],
    context_schema=Context,
    store=InMemoryStore()
)
```

--------------------------------

### Error handling with fallback values

Source: https://docs.langchain.com/oss/python/langchain/mcp

Use try-except in interceptors to catch specific error types and return fallback values instead of failing. Handles TimeoutError and ConnectionError with custom fallback messages. Prevents tool execution failures from breaking workflows.

```python
async def fallback_interceptor(
    request: MCPToolCallRequest,
    handler,
):
    """Return a fallback value if tool execution fails."""
    try:
        return await handler(request)
    except TimeoutError:
        return f"Tool {request.name} timed out. Please try again later."
    except ConnectionError:
        return f"Could not connect to {request.name} service. Using cached data."
```

--------------------------------

### Configure LangChain Chat Model with In-Memory Rate Limiter

Source: https://docs.langchain.com/oss/python/langchain/models

This snippet demonstrates how to initialize and apply LangChain's built-in `InMemoryRateLimiter` to a chat model. It configures the rate limiter with parameters like `requests_per_second`, `check_every_n_seconds`, and `max_bucket_size` to manage invocation rates, helping to prevent exceeding provider limits.

```python
from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # 1 request every 10s
    check_every_n_seconds=0.1,  # Check every 100ms whether allowed to make a request
    max_bucket_size=10  # Controls the maximum burst size.
)

model = init_chat_model(
    model="gpt-5",
    model_provider="openai",
    rate_limiter=rate_limiter  # [!code highlight]
)
```

--------------------------------

### Define HumanMessage with Optional Metadata (Python)

Source: https://docs.langchain.com/oss/python/langchain/messages

Demonstrates how to create a `HumanMessage` in LangChain, allowing for optional metadata fields like `name` for user identification and `id` for unique tracing. Note that the behavior of the `name` field can vary across different model providers.

```python
human_msg = HumanMessage(
    content="Hello!",
    name="alice",  # Optional: identify different users
    id="msg_123",  # Optional: unique identifier for tracing
)
```

--------------------------------

### Define a Python LangChain Tool with Real-time Streaming Updates

Source: https://docs.langchain.com/oss/python/langchain/tools

This Python code defines a LangChain tool named `get_weather` that utilizes `runtime.stream_writer` to send custom updates as the tool executes. It takes a city and a `ToolRuntime` object as input, using the `stream_writer` from `ToolRuntime` to provide real-time feedback. This functionality requires the tool to be invoked within a LangGraph execution context.

```python
from langchain.tools import tool, ToolRuntime

@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"
```

--------------------------------

### Set up Isaacus Embeddings in LangChain

Source: https://docs.langchain.com/oss/python/langchain/knowledge-base

Installs the LangChain Isaacus integration package and initializes the IsaacusEmbeddings model. This setup includes a check for the 'ISAACUS_API_KEY' environment variable, prompting the user for it if not found. The 'model' parameter specifies which Isaacus embedding model to use.

```shell
pip install -qU langchain-isaacus
```

```python
import getpass
import os

if not os.environ.get("ISAACUS_API_KEY"):
os.environ["ISAACUS_API_KEY"] = getpass.getpass("Enter API key for Isaacus: ")

from langchain_isaacus import IsaacusEmbeddings

embeddings = IsaacusEmbeddings(model="kanon-2-embedder")
```