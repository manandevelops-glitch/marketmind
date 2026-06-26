import streamlit as st
import asyncio
from marketmind.agent import config
from google.antigravity import Agent

st.set_page_config(page_title="MarketMind", page_icon="📈", layout="wide")

# Set a persistent session ID for the ADK
config.session_id = "streamlit_session"

# Initialize session state for UI
if "messages" not in st.session_state:
    st.session_state.messages = []

if "capital" not in st.session_state:
    st.session_state.capital = 500000.0

if "risk_percent" not in st.session_state:
    st.session_state.risk_percent = 2.0

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# Title & Description
st.title("📈 MarketMind AI")
st.markdown("Your multi-agent stock market research orchestrator.")

# Sidebar
with st.sidebar:
    st.header("Portfolio Settings")
    
    # Portfolio Capital Input
    new_capital = st.number_input(
        "Portfolio Capital (₹)", 
        min_value=10000.0, 
        max_value=10000000.0, 
        value=st.session_state.capital,
        step=10000.0
    )
    
    # Risk Percentage Slider
    new_risk = st.slider(
        "Risk Percentage (%)",
        min_value=1.0,
        max_value=5.0,
        value=st.session_state.risk_percent,
        step=0.5
    )
    
    st.subheader("Your Watchlist")
    if st.session_state.watchlist:
        for ticker in st.session_state.watchlist:
            st.markdown(f"- **{ticker}**")
    else:
        st.write("Watchlist is empty.")
        
    st.session_state.capital = new_capital
    st.session_state.risk_percent = new_risk

# Function to run the agent asynchronously
async def run_agent_query(user_input):
    async with Agent(config) as agent:
        # Send the user query to the agent
        response = await agent.chat(user_input)
        
        # We will iterate over the text chunks
        placeholder = st.empty()
        full_text = ""
        
        async for chunk in response.text:
            text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            full_text += text
            placeholder.markdown(full_text + "▌")
            
        # Append disclaimer
        full_text += "\n\n*Disclaimer: This is AI-generated analysis for educational purposes only. Not financial advice.*"
        placeholder.markdown(full_text)
        
        # Now collect tool calls to track subagents and state changes
        subagents_used = []
        async for tc in response.tool_calls:
            name = tc.name
            if "agent" in name.lower():
                subagents_used.append(name)
            
            # Sync UI state if the agent modified the watchlist
            if name == "manage_watchlist":
                args = tc.args or {}
                action = args.get("action", "").lower()
                ticker = args.get("ticker", "").upper()
                if action == "add" and ticker not in st.session_state.watchlist:
                    st.session_state.watchlist.append(ticker)
                elif action == "remove" and ticker in st.session_state.watchlist:
                    st.session_state.watchlist.remove(ticker)
                    
        return full_text, list(set(subagents_used))

# Main Chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("subagents"):
            with st.expander("Sub-agents used"):
                for sa in msg["subagents"]:
                    st.write(f"- {sa}")

if prompt := st.chat_input("Ask MarketMind about stocks, news, or risk sizing..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("MarketMind is thinking..."):
            try:
                # Run the agent in asyncio loop
                final_text, subagents = asyncio.run(run_agent_query(prompt))
                
                # Show expander if subagents were used
                if subagents:
                    with st.expander("Sub-agents used"):
                        for sa in subagents:
                            st.write(f"- {sa}")
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_text,
                    "subagents": subagents
                })
                
                # Rerun to update sidebar watchlist if changed
                st.rerun()
            except Exception as e:
                st.error(f"Error communicating with agent: {str(e)}")
