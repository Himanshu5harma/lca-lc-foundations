from dotenv import load_dotenv
import ssl
import httpx
import truststore

load_dotenv()

from langchain.tools import tool
from typing import Dict, Any
from tavily import TavilyClient

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
http_client = httpx.Client(verify=ssl_context)

chat_model = init_chat_model(
    model="gpt-5.4-mini",
    http_client=http_client,
)

tavily_client = TavilyClient()

@tool
def web_search(query: str) -> Dict[str, Any]:

    """Search the web for information"""

    return tavily_client.search(query)

system_prompt = """

You are a personal chef. The user will give you a list of ingredients they have left over in their house.

Using the web search tool, search the web for recipes that can be made with the ingredients they have.

Return recipe suggestions and eventually the recipe instructions to the user, if requested.

"""

agent = create_agent(
    model=chat_model,
    tools=[web_search],
    system_prompt=system_prompt
)


response = agent.invoke({"messages": [HumanMessage(content="I have some leftover chicken and rice. What can I make?")]})

print(response['messages'][-1].content)