import os
import asyncio
from dotenv import load_dotenv
from google.adk import Agent
from google.antigravity.utils.interactive import async_input
from .tools import get_stock_price, get_technical_analysis, calculate_position_size, get_market_news, manage_watchlist, update_user_profile, get_session_state
from .sub_agents.research_agent import research_agent_config
from .sub_agents.technical_agent import technical_agent_config
from .sub_agents.risk_agent import risk_agent_config

# Load environment variables from .env file
load_dotenv()

# Provide system instructions to give MarketMind its persona
root_agent = Agent(
    name="marketmind",
    model="gemini-2.5-flash",
    instruction=(
        "You are MarketMind, a helpful stock market research orchestrator. "
        "Use your sub-agents to analyze the user's queries. "
        "At the start of a conversation, use get_session_state to check for returning users. "
        "If they ask about a ticker they've queried before, reference previous analyses from the state dictionary. "
        "Maintain a conversational context of at least 5 turns."
    ),
    sub_agents=[research_agent_config, technical_agent_config, risk_agent_config],
    tools=[get_stock_price, get_technical_analysis, calculate_position_size, get_market_news, manage_watchlist, update_user_profile, get_session_state]
)

async def main():
    async with root_agent:
        print("Testing basic greeting...")
        response = await root_agent.chat("Hello!")
        print(f"Agent: {await response.text()}")
        print("\nDisclaimer: This is AI-generated analysis for educational purposes only. Not financial advice.\n")
        
        print("Starting interactive loop. Type 'exit' or 'quit' to end.")
        while True:
            try:
                user_input = await async_input("User: ")
                if user_input.strip().lower() in ['exit', 'quit']:
                    break
                
                response = await root_agent.chat(user_input)
                print("Agent: ", end="")
                async for chunk in response.text:
                    print(chunk.text if hasattr(chunk, 'text') else str(chunk), end="", flush=True)
                
                print("\n\nDisclaimer: This is AI-generated analysis for educational purposes only. Not financial advice.")
                print()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

if __name__ == "__main__":
    asyncio.run(main())
