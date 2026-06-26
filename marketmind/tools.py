import os
import yfinance as yf
import pandas as pd
import numpy as np
from google.adk.tools import ToolContext
from google import genai
from google.genai import types

def get_stock_price(ticker: str) -> dict:
    """
    Fetches the current price, day change percentage, and 52-week high/low for a stock.
    Automatically appends '.NS' for NSE stocks if not present.
    
    Args:
        ticker: The stock ticker symbol (e.g., 'RELIANCE').
        
    Returns:
        A dictionary containing the price, change_percent, 52_week_high, and 52_week_low.
    """
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = f"{ticker}.NS"
        
    stock = yf.Ticker(ticker)
    info = stock.info
    
    current_price = info.get("currentPrice", info.get("regularMarketPrice"))
    previous_close = info.get("previousClose", info.get("regularMarketPreviousClose"))
    
    change_percent = None
    if current_price and previous_close:
        change_percent = ((current_price - previous_close) / previous_close) * 100
        
    return {
        "ticker": ticker,
        "current_price": current_price,
        "change_percent": round(change_percent, 2) if change_percent is not None else None,
        "52_week_high": info.get("fiftyTwoWeekHigh"),
        "52_week_low": info.get("fiftyTwoWeekLow")
    }

def get_technical_analysis(ticker: str, ctx: ToolContext = None) -> dict:
    """
    Fetches 6 months of daily data to calculate technical indicators:
    20-day EMA, 50-day EMA, 14-day RSI, and 14-day ATR.
    Determines bullish, bearish, or neutral signals.
    
    Args:
        ticker: The stock ticker symbol.
        ctx: The tool context for state management.
        
    Returns:
        A dictionary containing the current price, EMAs, RSI, ATR, and overall signal.
    """
    if not ticker.endswith(".NS") and not ticker.endswith(".BO"):
        ticker = f"{ticker}.NS"
        
    # Fetch 6 months of daily data
    df = yf.download(ticker, period="6m", interval="1d", progress=False)
    
    if df.empty:
        return {"error": f"No data found for {ticker}"}
        
    # Handle multi-index columns from yfinance download if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    current_price = float(close.iloc[-1])
    
    # EMAs
    ema_20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
    ema_50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    
    # RSI (14-day Wilder's Smoothing equivalent)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    rsi_14 = float(rsi_series.iloc[-1])
    
    # ATR (14-day)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = float(tr.rolling(window=14).mean().iloc[-1])
    
    # Determine signal
    signal = "NEUTRAL"
    if current_price > ema_20 and ema_20 > ema_50 and rsi_14 < 70:
        signal = "BULLISH"
    elif current_price < ema_20 and ema_20 < ema_50 and rsi_14 > 30:
        signal = "BEARISH"
        
    result = {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "ema_20": round(ema_20, 2),
        "ema_50": round(ema_50, 2),
        "rsi_14": round(rsi_14, 2),
        "atr_14": round(atr_14, 2),
        "signal": signal
    }
    
    if ctx is not None:
        last_analyses = ctx.get_state("last_analyses", {})
        last_analyses[ticker] = result
        ctx.set_state("last_analyses", last_analyses)
        
    return result

def calculate_position_size(capital: float, risk_percent: float, entry_price: float, stop_loss: float) -> dict:
    """
    Calculates the number of shares to buy based on account capital,
    risk percentage, entry price, and stop loss.
    
    Args:
        capital: Total account capital.
        risk_percent: Percentage of capital willing to risk (e.g., 2.0 for 2%).
        entry_price: The planned entry price per share.
        stop_loss: The planned stop loss price per share.
        
    Returns:
        A dictionary with calculated position sizing details.
    """
    if entry_price <= stop_loss:
        return {"error": "Entry price must be greater than stop loss for a long position."}
        
    risk_amount = capital * (risk_percent / 100.0)
    risk_per_share = entry_price - stop_loss
    
    shares = int(risk_amount // risk_per_share)
    total_cost = shares * entry_price
    
    # Example reward target (1:2 risk/reward)
    target_price = entry_price + (risk_per_share * 2)
    reward_if_target = (target_price - entry_price) * shares
    
    return {
        "shares_to_buy": shares,
        "total_cost": round(total_cost, 2),
        "risk_amount": round(risk_amount, 2),
        "reward_if_target": round(reward_if_target, 2),
        "target_price_2r": round(target_price, 2)
    }

def get_market_news(query: str) -> str:
    """
    Uses the Gemini model's built-in Google Search grounding to fetch
    recent news about the query and summarize the top 3 headlines.
    
    Args:
        query: The topic or company to search for news about.
        
    Returns:
        A summarized string of the top 3 recent headlines.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY is not set."
        
    client = genai.Client(api_key=api_key)
    
    try:
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=f"Find the latest news for '{query}'. Provide a summary of the top 3 recent headlines.",
            config=types.GenerateContentConfig(
                tools=[{"google_search": {}}],
                temperature=0.2,
            )
        )
        return response.text
    except Exception as e:
        return f"Error fetching news: {str(e)}"

def manage_watchlist(action: str, ticker: str, ctx: ToolContext) -> str:
    """
    Adds or removes a stock ticker from the user's watchlist in the session state.
    
    Args:
        action: 'add' or 'remove'.
        ticker: The stock ticker symbol.
        ctx: The tool context for state management.
    """
    watchlist = ctx.get_state("watchlist", [])
    ticker = ticker.upper()
    
    if action.lower() == "add":
        if ticker not in watchlist:
            watchlist.append(ticker)
            ctx.set_state("watchlist", watchlist)
            return f"Added {ticker} to watchlist. Current watchlist: {watchlist}"
        return f"{ticker} is already in the watchlist."
    elif action.lower() == "remove":
        if ticker in watchlist:
            watchlist.remove(ticker)
            ctx.set_state("watchlist", watchlist)
            return f"Removed {ticker} from watchlist. Current watchlist: {watchlist}"
        return f"{ticker} is not in the watchlist."
    else:
        return f"Invalid action: {action}. Use 'add' or 'remove'."

def update_user_profile(capital: float, risk_percent: float, ctx: ToolContext) -> str:
    """
    Updates the user's capital and risk preferences in the session state.
    
    Args:
        capital: Total account capital.
        risk_percent: Percentage of capital willing to risk.
        ctx: The tool context for state management.
    """
    profile = {"capital": capital, "risk_percent": risk_percent}
    ctx.set_state("user_profile", profile)
    return f"User profile updated: Capital = {capital}, Risk = {risk_percent}%"

def get_session_state(ctx: ToolContext) -> dict:
    """
    Retrieves the entire current session state including watchlist, user profile, and last analyses.
    Useful for checking context when a user returns.
    
    Args:
        ctx: The tool context for state management.
    """
    return {
        "watchlist": ctx.get_state("watchlist", []),
        "user_profile": ctx.get_state("user_profile", {}),
        "last_analyses": ctx.get_state("last_analyses", {})
    }

