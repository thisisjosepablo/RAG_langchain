from dotenv import load_dotenv
import os
from langchain.agents import create_agent
import urllib.error
import urllib.request
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent

load_dotenv()

SYSTEM_PROMPT = """
    You are a literary data assistant.

## Capabilities

- `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file.
"""

@tool
def fetch_text_from_url(url: str)-> str:
    """Fetch the document from a URL"""
    req = urllib.request.Request(
        url=url,
        headers= {
            "User-Agent": "Mozilla/5.0"
        }
    )
    try:

        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetch failed {e}"

    text = raw.decode("utf-8", errors="replace")
    return text


model = init_chat_model(
    model = "claude-haiku-4-5-20251001",
    temperature = 0.5, 
    timeout=600,
    max_tokens=25000,
    streaming=True,
)

checkpointer = InMemorySaver()

agent = create_agent(
    model=model,
    tools= [fetch_text_from_url],
    system_prompt= SYSTEM_PROMPT,
    checkpointer= checkpointer
)


content = """
    I want to start studying about science of sports, specifically running and
    muscles training. As well, I would like to study about nutrition and food.

    That is the reason I want you to give a list of 10 books that I MUST read for being
    a master the teaching of good habits of running, muscles training and nutrition.

    I only need a list with the title of the book and the author. Example:
    '1)Book title 1 - Name author 1
     2) Book title 2 - Name author 2
     3) etc'
"""

agent_result = agent.invoke(
    input= {'messages': 
            [{"role": "user", "content" : content}]
            },
    config={"configurable": {"thread_id": "great-gatsby-lc"}}
)

print(agent_result["messages"][-1].content_blocks)