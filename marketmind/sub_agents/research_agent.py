from google.adk import Agent
from ..tools import get_market_news

research_agent_config = Agent(
    name="research_agent",
    instruction="You are a fundamental research analyst. Find the latest news and summarize the market sentiment for the requested ticker.",
    tools=[get_market_news]
)
