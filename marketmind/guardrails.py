import re
from google.antigravity import types
from google.antigravity.hooks import hooks

# 1. Input guardrail (before_agent_callback)

@hooks.pre_turn
async def before_agent_callback_turn(data: types.Content) -> types.HookResult:
    """
    Blocks queries that try to override agent instructions.
    Security rationale: Prevents prompt injection attacks that could trick the
    agent into ignoring its system instructions or acting maliciously.
    """
    text_content = ""
    if isinstance(data, str):
        text_content = data
    elif hasattr(data, "parts"):
        for part in data.parts:
            if hasattr(part, "text"):
                text_content += part.text.lower()
    else:
        text_content = str(data).lower()
        
    text_content = text_content.lower()
    
    blocked_phrases = ["ignore previous instructions", "you are now", "system prompt"]
    for phrase in blocked_phrases:
        if phrase in text_content:
            return types.HookResult(
                allow=False, 
                message="Security Violation: Prompt injection attempt detected."
            )
            
    return types.HookResult(allow=True)

@hooks.pre_tool_call_decide
async def before_agent_callback_tool(data: types.ToolCall) -> types.HookResult:
    """
    Validates ticker inputs and capital input.
    Security rationale: Prevents injection attacks through ticker parameters
    and ensures numerical bounds for risk management logic.
    """
    args = data.args or {}
    
    # Validate ticker
    if "ticker" in args:
        ticker = str(args["ticker"])
        # Allow alphabetic characters and the .NS / .BO suffixes
        clean_ticker = ticker.replace(".NS", "").replace(".BO", "").replace(".", "").strip()
        if not clean_ticker.isalpha():
            return types.HookResult(
                allow=False, 
                message=f"Validation Error: Ticker '{ticker}' contains invalid characters. Only alphabetic characters are allowed."
            )
            
    # Validate capital
    if "capital" in args:
        try:
            capital = float(args["capital"])
            if not (10000 <= capital <= 10000000):
                return types.HookResult(
                    allow=False, 
                    message=f"Validation Error: Capital must be between ₹10,000 and ₹1,00,00,000."
                )
        except ValueError:
            return types.HookResult(
                allow=False,
                message="Validation Error: Capital must be a valid number."
            )
            
    return types.HookResult(allow=True)

# 2. Output guardrail (after_agent_callback)

@hooks.post_turn
async def after_agent_callback(data: str):
    """
    Checks that no response contains specific buy/sell recommendations without the risk disclaimer.
    Security rationale: Prevents the agent from giving direct financial advice which could
    lead to liability issues. Note: In ADK, post_turn is an InspectHook, so it observes the output.
    """
    text = data.lower()
    
    # Check for strong recommendation keywords
    recommendations = ["buy now", "strong buy", "definitely sell", "sell immediately", "invest in"]
    
    # Check if disclaimer is present
    disclaimer = "not financial advice"
    
    has_recommendation = any(rec in text for rec in recommendations)
    has_disclaimer = disclaimer in text
    
    if has_recommendation and not has_disclaimer:
        print("\n[SECURITY WARNING]: Response contains buy/sell recommendations without a risk disclaimer!")
