# MarketMind

MarketMind is an AI-powered stock market research orchestrator built using the Google Antigravity SDK and Streamlit. It uses a multi-agent architecture to retrieve stock prices, perform technical analysis, manage risks, and fetch the latest market news.

## Features
- **Multi-Agent Delegation**: Orchestrates between Research, Technical, and Risk sub-agents.
- **Session Memory**: Remembers your watchlist and portfolio parameters across sessions.
- **Security Guardrails**: Built-in prompt injection and safety filtering using the ADK hooks system.
- **Web Interface**: Clean, dark-themed interactive Streamlit chat UI.

## Getting Started

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API key

### 2. Setup
Clone the repository, set up a virtual environment, and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory and add your Gemini API key:
```bash
GEMINI_API_KEY=your_api_key_here
```

### 4. Run the Web App
Start the Streamlit application:
```bash
streamlit run app.py
```

### 5. Run the Tests
We use pytest and pytest-asyncio to evaluate the agent functionality:
```bash
pytest tests/test_agent.py -v
```

> **Note**: Free-tier Gemini API keys may experience rate limiting (`429 Resource Exhausted`) during heavy usage or when running the full test suite.
