from google.adk import Agent
from ..tools import get_stock_price, get_technical_analysis

technical_agent_config = Agent(
    name="technical_agent",
    instruction="You are a technical analyst. Use the tools to retrieve price data and technical indicators. Keep your analysis concise and focused on actionable insights.",
    tools=[get_stock_price, get_technical_analysis]
)
