from google.adk import Agent
from ..tools import calculate_position_size

risk_agent_config = Agent(
    name="risk_agent",
    instruction="You are a risk manager. Given a stock and portfolio details, calculate optimal position sizing and identify risks. Always include a disclaimer that this is not financial advice.",
    tools=[calculate_position_size]
)
