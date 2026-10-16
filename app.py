import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import feedparser
import io
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any

# ============================================================
# 1. INITIAL SETUP & MATRIX INTERFACE STYLE SYSTEM
# ============================================================
nltk.download("vader_lexicon", quiet=True)

st.set_page_config(
    page_title="OneStop-InfoSpot | Asset Intelligence Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
    <style>
        html, body, .stApp, p, label, .stMarkdown p, .stCaption, 
        h1, h2, h3, h4, h5, h6,
        .stTextInput input, .stSelectbox div, .stSlider div, .stCheckbox label,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        .stDataFrame, table, th, td {
            font-family: 'Playfair Display', serif !important;
        }
        span[data-testid="stIconVisibility"], 
        button div[data-testid="stMarkdownContainer"] p,
        [class*="st-emotion-cache"] svg,
        [class*="st-emotion-cache"] i,
        span[class*="icon"],
        button div,
        .stExpander h2 span,
        [data-testid="collapsedControlSidebar"] button {
            font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }
        .stApp { background-color: #0e1117; color: #fafafa; }
        section[data-testid="stSidebar"] { background-color: #1a1d24; }
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown { color: #fafafa !important; }
        [data-testid="stMetricValue"] { color: #00FF7F !important; }
        .block-container {
            padding-top: 1.2rem !important;
        }
        .silver-title {
            font-family: 'Playfair Display', serif !important;
            font-size: 3.8rem;
            font-weight: 700;
            text-align: center;
            margin: -0.6rem 0 0.15rem 0;
            letter-spacing: 2px;
            background: linear-gradient(
                to bottom,
                #ffffff 0%,
                #e8e8e8 25%,
                #c0c0c0 50%,
                #a0a0a0 75%,
                #d0d0d0 100%
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            filter: drop-shadow(0 0 6px rgba(192, 192, 192, 0.7))
                    drop-shadow(0 0 12px rgba(255, 255, 255, 0.3));
        }
        .silver-subtitle {
            font-family: 'Playfair Display', serif !important;
            font-size: 2.1rem;
            font-weight: 600;
            text-align: center;
            margin: 0 0 0.9rem 0;
            letter-spacing: 1px;
            background: linear-gradient(
                to bottom,
                #ffffff 0%,
                #e8e8e8 25%,
                #c0c0c0 50%,
                #a0a0a0 75%,
                #d0d0d0 100%
            );
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            filter: drop-shadow(0 0 4px rgba(192, 192, 192, 0.6))
                    drop-shadow(0 0 8px rgba(255, 255, 255, 0.25));
        }
        .news-box {
            display: flex;
            flex-direction: column;
            gap: 12px;
            padding: 10px 0;
            width: 100%;
        }
        
        .news-meta {
            font-size: 0.9rem;
            color: #a0a0a0;
            margin: 0;
        }
        .news-summary {
            font-size: 1.05rem;
            color: #fafafa;
            line-height: 1.6;
            margin: 0;
        }
        
        .news-action-link {
            align-self: flex-start;
            background-color: #1f242d;
            color: #00FFFF !important;
            padding: 6px 14px;
            border-radius: 4px;
            text-decoration: none !important;
            font-size: 0.95rem;
            font-weight: bold;
            border: 1px solid #3b4252;
            transition: all 0.2s ease-in-out;
        }
        
        .news-action-link:hover {
            background-color: #00FF7F;
            color: #0e1117 !important;
            border-color: #00FF7F;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ============================================================
# 2. CONFIGURATION & COMPREHENSIVE TIME WINDOWS
# ============================================================
RESOLUTION_MAP = {
    "15min": {"interval": "15m", "lookback": 96, "period": "7d", "is_6h": False},
    "1hr": {"interval": "1h", "lookback": 72, "period": "30d", "is_6h": False},
    "6hr": {"interval": "1h", "lookback": 84, "period": "60d", "is_6h": True},
    "Daily (1d)": {"interval": "1d", "lookback": 90, "period": "2y", "is_6h": False},
    "Weekly (1w)": {"interval": "1wk", "lookback": 52, "period": "2y", "is_6h": False},
    # --- 15-YEAR HISTORICAL OPTIONS ---
    "Daily (5y)": {"interval": "1d", "lookback": 1260, "period": "5y", "is_6h": False},
    "Daily (10y)": {"interval": "1d", "lookback": 2520, "period": "10y", "is_6h": False},
    "Daily (15y)": {"interval": "1d", "lookback": 3800, "period": "15y", "is_6h": False},
    "Weekly (15y)": {"interval": "1wk", "lookback": 780, "period": "15y", "is_6h": False},
    "Monthly (15y)": {"interval": "1mo", "lookback": 180, "period": "15y", "is_6h": False},
}

def safe_float(value, default=None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def normalize_columns(df: pd.DataFrame, target_ticker: Optional[str] = None) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    if isinstance(df.columns, pd.MultiIndex):
        try:
            if target_ticker and target_ticker in df.columns.get_level_values(1):
                df = df.xs(target_ticker, axis=1, level=1)
            elif target_ticker and target_ticker in df.columns.get_level_values(0):
                df = df.xs(target_ticker, axis=1, level=0)
            else:
                df.columns = df.columns.get_level_values(0)
        except IndexError:
            df.columns = [col for col in df.columns]
            
    df.columns = [str(c).lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

# ============================================================
# 2b. ACTIVE USERS TRACKER
# ============================================================
@st.cache_resource
def get_active_users_store() -> Dict[str, float]:
    """Shared in-memory store: session_id → last_seen timestamp"""
    return {}

def get_active_user_count(timeout_seconds: int = 90) -> int:
    store = get_active_users_store()
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    session_id = st.session_state.session_id
    now = time.time()
    
    store[session_id] = now
    
    stale = [sid for sid, last in store.items() if now - last > timeout_seconds]
    for sid in stale:
        del store[sid]
    
    return len(store)


# ============================================================
# 2c. WATCHLIST & PORTFOLIO HELPERS
# ============================================================
import json
import os

WATCHLIST_FILE = "user_watchlist.json"
PORTFOLIO_FILE = "user_portfolio.json"

def load_json_file(path: str, default=None):
    if default is None:
        default = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json_file(path: str, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

def load_watchlist() -> list:
    return load_json_file(WATCHLIST_FILE, [])

def save_watchlist(tickers: list):
    save_json_file(WATCHLIST_FILE, tickers)

def load_portfolio() -> list:
    """List of dicts: {ticker, qty, entry_price}"""
    return load_json_file(PORTFOLIO_FILE, [])

def save_portfolio(portfolio: list):
    save_json_file(PORTFOLIO_FILE, portfolio)


# ============================================================
# 2d. ALERTS SYSTEM
# ============================================================
ALERTS_FILE = "user_alerts.json"

def load_alerts() -> list:
    """List of alert dicts"""
    return load_json_file(ALERTS_FILE, [])

def save_alerts(alerts: list):
    save_json_file(ALERTS_FILE, alerts)

def check_alerts(alerts: list, data: dict, liquidity_profiles: dict, patterns_by_ticker: dict) -> list:
    """
    Evaluate all alerts against current data.
    Returns list of triggered alert messages.
    """
    triggered = []
    for alert in alerts:
        try:
            t = alert.get("ticker", "").upper()
            atype = alert.get("type")
            condition = alert.get("condition")
            value = float(alert.get("value", 0))
            
            if atype == "price" and t in data and not data[t].empty:
                price = float(data[t]["close"].iloc[-1])
                if condition == "above" and price >= value:
                    triggered.append(f"🚨 {t} price ${price:,.2f} is ABOVE ${value:,.2f}")
                elif condition == "below" and price <= value:
                    triggered.append(f"🚨 {t} price ${price:,.2f} is BELOW ${value:,.2f}")
            
            elif atype == "rsi" and t in data and "rsi" in data[t].columns:
                rsi = data[t]["rsi"].iloc[-1]
                if not pd.isna(rsi):
                    if condition == "above" and rsi >= value:
                        triggered.append(f"🚨 {t} RSI {rsi:.1f} is ABOVE {value}")
                    elif condition == "below" and rsi <= value:
                        triggered.append(f"🚨 {t} RSI {rsi:.1f} is BELOW {value}")
            
            elif atype == "funding" and t in liquidity_profiles:
                fr = liquidity_profiles[t].get("funding")
                if fr is not None:
                    if condition == "above" and fr >= value:
                        triggered.append(f"🚨 {t} Funding {fr:.4f}% is ABOVE {value}%")
                    elif condition == "below" and fr <= value:
                        triggered.append(f"🚨 {t} Funding {fr:.4f}% is BELOW {value}%")
            
            elif atype == "pattern" and t in patterns_by_ticker:
                pats = patterns_by_ticker.get(t, [])
                target = alert.get("pattern_type", "").lower()
                for p in pats:
                    if target in p["type"].lower():
                        triggered.append(f"🚨 {t} Pattern detected: {p['type']} ({p['direction']})")
                        break
        except Exception:
            continue
    return triggered



# ============================================================
# 2e. ANALYSIS TOOLS HELPERS
# ============================================================
def compute_correlation_matrix(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a close-price returns correlation matrix for the given tickers."""
    closes = {}
    for t, df in data.items():
        if df is not None and not df.empty and "close" in df.columns:
            closes[t] = df["close"]
    if len(closes) < 2:
        return pd.DataFrame()
    combined = pd.DataFrame(closes).dropna(how="all")
    returns = combined.pct_change().dropna()
    if returns.empty:
        return pd.DataFrame()
    return returns.corr()


def compute_volatility_stats(data: Dict[str, pd.DataFrame], window: int = 30) -> pd.DataFrame:
    """Compute recent volatility and ATR stats for each ticker."""
    rows = []
    for t, df in data.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        rets = df["close"].pct_change().dropna()
        vol_30 = rets.tail(window).std() * np.sqrt(365) * 100 if len(rets) >= 5 else np.nan  # annualized approx
        vol_7 = rets.tail(7).std() * np.sqrt(365) * 100 if len(rets) >= 5 else np.nan
        atr_val = df["atr"].iloc[-1] if "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]) else np.nan
        price = df["close"].iloc[-1]
        atr_pct = (atr_val / price) * 100 if price and atr_val and not pd.isna(atr_val) else np.nan
        rows.append({
            "Ticker": t,
            "Price": price,
            "Vol 7d (ann%)": vol_7,
            "Vol 30d (ann%)": vol_30,
            "ATR": atr_val,
            "ATR %": atr_pct,
        })
    return pd.DataFrame(rows)


def compute_multi_period_performance(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Performance over multiple lookback periods."""
    rows = []
    for t, df in data.items():
        if df is None or len(df) < 5 or "close" not in df.columns:
            continue
        close = df["close"]
        current = close.iloc[-1]
        def ret(n):
            if len(close) > n:
                return ((current / close.iloc[-n-1]) - 1) * 100
            return np.nan
        rows.append({
            "Ticker": t,
            "1D %": ret(1),
            "7D %": ret(7),
            "30D %": ret(30),
            "90D %": ret(90),
            "YTD %": ret(min(len(close)-1, 252)),
        })
    return pd.DataFrame(rows)

# ============================================================
# 3. ADVANCED TECHNICAL INDICATORS ALGORITHMS
# ============================================================
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return sma, upper, lower

def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Compute Stochastic Oscillator %K and %D"""
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    k = 100 * (df["close"] - low_min) / (high_max - low_min)
    d = k.rolling(window=d_period).mean()
    return k, d

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range"""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "close" not in df.columns:
        return df
    df = df.copy()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = compute_bollinger(df["close"])
    # New indicators
    if all(c in df.columns for c in ["high", "low"]):
        df["stoch_k"], df["stoch_d"] = compute_stochastic(df)
        df["atr"] = compute_atr(df)
    return df

# ============================================================
# 3b. CHART PATTERN RECOGNITION ENGINE
# ============================================================
def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict]:
    """Detect common candlestick patterns. Returns list of dicts with pattern info."""
    patterns = []
    if df is None or len(df) < 5 or not all(c in df.columns for c in ["open", "high", "low", "close"]):
        return patterns
    
    df = df.copy()
    body = (df["close"] - df["open"]).abs()
    range_ = df["high"] - df["low"]
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]
    
    # Avoid division by zero
    range_ = range_.replace(0, np.nan)
    
    for i in range(2, len(df)):
        idx = df.index[i]
        o, h, l, c = df["open"].iloc[i], df["high"].iloc[i], df["low"].iloc[i], df["close"].iloc[i]
        prev_o, prev_c = df["open"].iloc[i-1], df["close"].iloc[i-1]
        body_i = body.iloc[i]
        range_i = range_.iloc[i]
        
        if pd.isna(range_i) or range_i == 0:
            continue
            
        # Doji
        if body_i / range_i < 0.1:
            patterns.append({
                "type": "Doji",
                "direction": "Neutral",
                "index": idx,
                "price": c,
                "confidence": "Medium",
                "description": "Indecision / potential reversal"
            })
        
        # Hammer (bullish)
        if (lower_wick.iloc[i] > 2 * body_i) and (upper_wick.iloc[i] < body_i * 0.3) and (c > o):
            patterns.append({
                "type": "Hammer",
                "direction": "Bullish",
                "index": idx,
                "price": c,
                "confidence": "Medium",
                "description": "Potential bullish reversal"
            })
        
        # Shooting Star (bearish)
        if (upper_wick.iloc[i] > 2 * body_i) and (lower_wick.iloc[i] < body_i * 0.3) and (c < o):
            patterns.append({
                "type": "Shooting Star",
                "direction": "Bearish",
                "index": idx,
                "price": c,
                "confidence": "Medium",
                "description": "Potential bearish reversal"
            })
        
        # Bullish Engulfing
        if (prev_c < prev_o) and (c > o) and (c >= prev_o) and (o <= prev_c) and (body_i > body.iloc[i-1] * 1.1):
            patterns.append({
                "type": "Bullish Engulfing",
                "direction": "Bullish",
                "index": idx,
                "price": c,
                "confidence": "High",
                "description": "Strong bullish reversal signal"
            })
        
        # Bearish Engulfing
        if (prev_c > prev_o) and (c < o) and (c <= prev_o) and (o >= prev_c) and (body_i > body.iloc[i-1] * 1.1):
            patterns.append({
                "type": "Bearish Engulfing",
                "direction": "Bearish",
                "index": idx,
                "price": c,
                "confidence": "High",
                "description": "Strong bearish reversal signal"
            })
    
    # Keep only the most recent patterns (last 30 bars worth)
    recent_cutoff = df.index[-min(40, len(df))]
    patterns = [p for p in patterns if p["index"] >= recent_cutoff]
    return patterns[-12:]  # limit to last 12 detections


def detect_chart_patterns(df: pd.DataFrame) -> List[Dict]:
    """Detect higher-level chart patterns (Double Top/Bottom, simple S/R)."""
    patterns = []
    if df is None or len(df) < 20 or "close" not in df.columns:
        return patterns
    
    closes = df["close"].values
    highs = df["high"].values if "high" in df.columns else closes
    lows = df["low"].values if "low" in df.columns else closes
    idx = df.index
    
    # --- Simple Support & Resistance (local peaks/troughs) ---
    window = 5
    for i in range(window, len(df) - window):
        # Local high (resistance)
        if highs[i] == max(highs[i-window:i+window+1]):
            patterns.append({
                "type": "Resistance",
                "direction": "Neutral",
                "index": idx[i],
                "price": highs[i],
                "confidence": "Medium",
                "description": f"Local resistance @ {highs[i]:.2f}"
            })
        # Local low (support)
        if lows[i] == min(lows[i-window:i+window+1]):
            patterns.append({
                "type": "Support",
                "direction": "Neutral",
                "index": idx[i],
                "price": lows[i],
                "confidence": "Medium",
                "description": f"Local support @ {lows[i]:.2f}"
            })
    
    # --- Double Top / Double Bottom (simplified) ---
    lookback = min(60, len(df) - 10)
    recent_highs = []
    recent_lows = []
    
    for i in range(len(df) - lookback, len(df) - 3):
        if highs[i] == max(highs[max(0, i-4):i+5]):
            recent_highs.append((i, highs[i]))
        if lows[i] == min(lows[max(0, i-4):i+5]):
            recent_lows.append((i, lows[i]))
    
    # Double Top
    if len(recent_highs) >= 2:
        for j in range(len(recent_highs) - 1):
            i1, p1 = recent_highs[j]
            i2, p2 = recent_highs[j+1]
            if abs(p1 - p2) / p1 < 0.025 and (i2 - i1) > 5:  # within 2.5% and separated
                patterns.append({
                    "type": "Double Top",
                    "direction": "Bearish",
                    "index": idx[i2],
                    "price": p2,
                    "confidence": "Medium",
                    "description": f"Double Top near {p2:.2f}"
                })
                break
    
    # Double Bottom
    if len(recent_lows) >= 2:
        for j in range(len(recent_lows) - 1):
            i1, p1 = recent_lows[j]
            i2, p2 = recent_lows[j+1]
            if abs(p1 - p2) / p1 < 0.025 and (i2 - i1) > 5:
                patterns.append({
                    "type": "Double Bottom",
                    "direction": "Bullish",
                    "index": idx[i2],
                    "price": p2,
                    "confidence": "Medium",
                    "description": f"Double Bottom near {p2:.2f}"
                })
                break
    
    # Keep only reasonably recent
    if len(df) > 50:
        cutoff = df.index[-50]
        patterns = [p for p in patterns if p["index"] >= cutoff]
    
    return patterns


def detect_all_patterns(df: pd.DataFrame) -> List[Dict]:
    """Run all pattern detectors and return combined list."""
    candle = detect_candlestick_patterns(df)
    chart = detect_chart_patterns(df)
    return candle + chart


def add_patterns_to_chart(fig: go.Figure, patterns: List[Dict], row: int = 1) -> go.Figure:
    """Overlay detected patterns on the price chart as annotations and markers."""
    if not patterns:
        return fig
    
    color_map = {
        "Bullish": "#00FF7F",
        "Bearish": "#FF4500",
        "Neutral": "#FFD700",
    }
    
    symbol_map = {
        "Doji": "diamond",
        "Hammer": "triangle-up",
        "Shooting Star": "triangle-down",
        "Bullish Engulfing": "star",
        "Bearish Engulfing": "star",
        "Double Top": "x",
        "Double Bottom": "x",
        "Support": "circle",
        "Resistance": "circle",
    }
    
    for p in patterns:
        color = color_map.get(p["direction"], "#FFFFFF")
        symbol = symbol_map.get(p["type"], "circle")
        
        # Marker
        fig.add_trace(
            go.Scatter(
                x=[p["index"]],
                y=[p["price"]],
                mode="markers",
                marker=dict(
                    symbol=symbol,
                    size=12,
                    color=color,
                    line=dict(width=1, color="white"),
                ),
                name=p["type"],
                showlegend=False,
                hovertemplate=f"<b>{p['type']}</b><br>{p['description']}<br>Price: {p['price']:.2f}<extra></extra>",
            ),
            row=row, col=1,
        )
        
        # Annotation label (only for high-confidence or key patterns)
        if p["confidence"] == "High" or p["type"] in ["Double Top", "Double Bottom", "Support", "Resistance"]:
            fig.add_annotation(
                x=p["index"],
                y=p["price"],
                text=p["type"],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor=color,
                ax=0,
                ay=-30 if p["direction"] != "Bearish" else 30,
                font=dict(size=10, color=color, family="Playfair Display, serif"),
                bgcolor="rgba(14,17,23,0.8)",
                bordercolor=color,
                borderwidth=1,
                row=row, col=1,
            )
    
    return fig

# ============================================================
# 4. DATA EXTRACTION CORES
# ============================================================
@st.cache_data(ttl=60, show_spinner="Fetching market data...")
def load_ticker_data(
    tickers: Tuple[str, ...],
    interval: str,
    lookback: int,
    period: str,
    is_6h: bool,
) -> Dict[str, pd.DataFrame]:
    results = {}
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            hist = ticker_obj.history(period=period, interval=interval, auto_adjust=True)
            if hist is None or hist.empty:
                # One retry with a shorter period as fallback
                hist = ticker_obj.history(period="1y", interval=interval, auto_adjust=True)
                if hist is None or hist.empty:
                    continue
            hist = normalize_columns(hist, target_ticker=t)
            if is_6h and not hist.empty:
                try:
                    hist = (
                        hist.resample("6h")
                        .agg({
                            "open": "first",
                            "high": "max",
                            "low": "min",
                            "close": "last",
                            "volume": "sum",
                        })
                        .dropna()
                    )
                except Exception:
                    pass
            if not hist.empty:
                hist = add_indicators(hist)
                display = hist.iloc[-lookback:] if len(hist) >= lookback else hist
                results[t] = display
        except Exception as e:
            # Fail silently for individual tickers so one bad ticker doesn't break the app
            pass
    return results

@st.cache_data(ttl=3600)
def get_long_term_avg(ticker: str, years: int = 15) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(period=f"{years}y", interval="1d", auto_adjust=True)
        hist = normalize_columns(hist, target_ticker=ticker)
        if hist.empty or "close" not in hist.columns:
            return None
        return float(hist["close"].mean())
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_daily_pivots(tickers: Tuple[str, ...]) -> Dict[str, Dict[str, float]]:
    pivots = {}
    for t in tickers:
        try:
            daily = yf.Ticker(t).history(period="10d", interval="1d", auto_adjust=True)
            daily = normalize_columns(daily, target_ticker=t)
            if len(daily) < 2:
                continue
            prev = daily.iloc[-2]
            high, low, close = float(prev["high"]), float(prev["low"]), float(prev["close"])
            pivot = (high + low + close) / 3
            pivots[t] = {
                "Pivot": pivot,
                "R1": (2 * pivot) - low,
                "R2": pivot + (high - low),
                "S1": (2 * pivot) - high,
                "S2": pivot - (high - low),
            }
        except Exception:
            continue
    return pivots

@st.cache_data(ttl=300)
def get_fear_greed() -> Tuple[str, str]:
    try:
        res = requests.get("https://api.alternative.me/fng/", headers=HEADERS, timeout=8).json()
        data = res["data"]
        if isinstance(data, list) and len(data) > 0:
            return data[0]["value"], data[0]["value_classification"]
        return "N/A", "Unavailable"
    except Exception:
        return "N/A", "Unavailable"

@st.cache_data(ttl=90, show_spinner="Fetching fresh crypto news...")
def get_crypto_news(cache_buster: str, max_items: int = 12) -> List[Dict]:
    feeds = {
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "Decrypt": "https://decrypt.co/feed",
        "The Block": "https://www.theblock.co/rss.xml",
        "CryptoSlate": "https://cryptoslate.com/feed/",
    }
    articles = []
    for source, url in feeds.items():
        try:
            resp = requests.get(url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                continue
            feed = feedparser.parse(io.BytesIO(resp.content))
            for entry in feed.entries[:6]:
                published_struct = entry.get("published_parsed") or entry.get("updated_parsed")
                if published_struct:
                    try:
                        pub_dt = pd.Timestamp(*published_struct[:6], tz="UTC")
                    except Exception:
                        pub_dt = pd.Timestamp.utcnow()
                else:
                    raw = entry.get("published") or entry.get("updated") or ""
                    try:
                        pub_dt = pd.to_datetime(raw, utc=True)
                    except Exception:
                        pub_dt = pd.Timestamp.utcnow()
                summary = entry.get("summary", "") or entry.get("description", "")
                if summary:
                    summary = (
                        summary.replace("<p>", "")
                        .replace("</p>", "")
                        .replace("<br>", " ")
                        .replace("<br/>", " ")
                    )
                    summary = summary[:280] + ("..." if len(summary) > 280 else "")
                articles.append({
                    "title": entry.get("title", "No title").strip(),
                    "link": entry.get("link", "#"),
                    "source": source,
                    "published": pub_dt.strftime("%Y-%m-%d %H:%M UTC"),
                    "published_dt": pub_dt,
                    "summary": summary,
                })
        except Exception:
            continue
    articles = sorted(articles, key=lambda x: x["published_dt"], reverse=True)
    return articles[:max_items]

@st.cache_data(ttl=90)
def get_real_onchain(ticker: str) -> Dict[str, Any]:
    t = ticker.upper().replace("-USD", "").replace("USDT", "").replace("USD", "").strip()
    if t in ["BTC", "BITCOIN"]:
        try:
            stats = requests.get("https://api.blockchain.info/stats", headers=HEADERS, timeout=10).json()
            addr_resp = requests.get(
                "https://api.blockchain.info/charts/n-unique-addresses?timespan=5days&format=json",
                headers=HEADERS, timeout=10,
            ).json()
            latest_addr = int(addr_resp["values"][-1]["y"]) if addr_resp.get("values") else "N/A"
            hash_rate_ehs = stats.get("hash_rate", 0) / 1e9
            return {
                "Network": "Bitcoin",
                "Hash Rate (EH/s)": f"{hash_rate_ehs:,.1f}",
                "Difficulty": f"{stats.get('difficulty', 0):,.0f}",
                "24h Transactions": f"{stats.get('n_tx', 0):,}",
                "Blocks Mined (24h)": stats.get("n_blocks_mined", "N/A"),
                "Avg Minutes/Block": f"{stats.get('minutes_between_blocks', 0):.2f}",
                "Unique Addresses (latest)": f"{latest_addr:,}" if isinstance(latest_addr, int) else latest_addr,
                "Est. Tx Volume (USD)": f"${stats.get('estimated_transaction_volume_usd', 0):,.0f}",
                "Total BTC in Circulation": f"{stats.get('totalbc', 0) / 1e8:,.0f}",
                "Source": "Blockchain.com (free)",
            }
        except Exception as e:
            return {"Error": f"BTC failed: {str(e)[:100]}"}
    try:
        r = requests.get("https://api.llama.fi/v2/chains", headers=HEADERS, timeout=12)
        chains = r.json()
        matched = None
        for c in chains:
            name = (c.get("name") or "").lower()
            symbol = (c.get("tokenSymbol") or "").upper()
            if (
                symbol == t
                or (t == "XRP" and "xrpl" in name)
                or (t == "ETH" and name == "ethereum")
                or (t == "SOL" and name == "solana")
            ):
                matched = c
                break
        if matched:
            return {
                "Network / Chain": matched.get("name"),
                "TVL": f"${matched.get('tvl', 0):,.0f}",
                "Token Symbol": matched.get("tokenSymbol") or t,
                "Source": "DefiLlama (free)",
            }
    except Exception:
        pass
    return {"Note": f"No free on-chain data found for {t}", "Source": "Free endpoints only"}

@st.cache_data(ttl=60)
def get_futures_metrics(ticker: str, current_price: float = 0.0) -> Dict[str, Any]:
    t = ticker.upper().replace("-USD", "").replace("USDT", "").replace("USD", "").strip()
    symbol_okx = f"{t}-USDT-SWAP"
    ccy = t
    result = {
        "source": "OKX",
        "long_pct": None,
        "short_pct": None,
        "oi": None,
        "funding_rate": None,
        "errors": [],
    }
    try:
        r = requests.get(
            f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
            f"?ccy={ccy}&period=1H",
            headers=HEADERS,
            timeout=8,
        )
        data = r.json()
        if data.get("code") == "0" and data.get("data"):
            latest = data["data"][-1]
            ratio = safe_float(latest[1])
            if ratio and ratio > 0:
                long_pct = (ratio / (1 + ratio)) * 100
                result["long_pct"] = round(long_pct, 1)
                result["short_pct"] = round(100 - long_pct, 1)
        r = requests.get(
            f"https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId={symbol_okx}",
            headers=HEADERS,
            timeout=6,
        )
        data = r.json()
        if data.get("code") == "0" and data.get("data"):
            oi = safe_float(data["data"][0].get("oiCcy") or data["data"][0].get("oi"))
            if oi:
                result["oi"] = oi
        r = requests.get(
            f"https://www.okx.com/api/v5/public/funding-rate?instId={symbol_okx}",
            headers=HEADERS,
            timeout=6,
        )
        data = r.json()
        if data.get("code") == "0" and data.get("data"):
            fr = safe_float(data["data"][0].get("fundingRate"))
            if fr is not None:
                result["funding_rate"] = fr * 100
    except Exception as e:
        result["errors"].append(str(e)[:100])
        result["source"] = "OKX (partial failure)"
    if result["long_pct"] is None:
        result["long_pct"] = 50.0
        result["short_pct"] = 50.0
        result["source"] += " (L/S fallback)"
    return result

@st.cache_data(ttl=300, show_spinner="Gathering cross-asset macro indicators...")
def load_macro_data() -> Dict[str, pd.DataFrame]:
    macro_tickers = {
        "DXY (U.S. Dollar Index)": "DX-Y.NYB",
        "S&P 500 Index": "^GSPC",
        "NASDAQ 100 Index": "^NDX",
        "Gold Futures": "GC=F",
        "Crude Oil Futures": "CL=F",
        "U.S. 10Y Treasury Yield": "^TNX"
    }
    macro_results = {}
    for label, sym in macro_tickers.items():
        try:
            m_hist = yf.Ticker(sym).history(period="3mo", interval="1d", auto_adjust=True)
            if not m_hist.empty:
                macro_results[label] = normalize_columns(m_hist, target_ticker=sym)
        except Exception:
            continue
    return macro_results

# ============================================================
# 4b. PREDICTIVE ANALYTICS ENGINE
# ============================================================
def compute_technical_score(hist: pd.DataFrame) -> Dict[str, float]:
    if hist is None or hist.empty or "close" not in hist.columns:
        return {"score": 0.0, "details": {}}
    
    score = 0.0
    details = {}
    last = hist.iloc[-1]
    
    if "rsi" in hist.columns and not pd.isna(last["rsi"]):
        rsi = last["rsi"]
        if rsi < 30:
            rsi_score = 40
        elif rsi < 40:
            rsi_score = 20
        elif rsi > 70:
            rsi_score = -40
        elif rsi > 60:
            rsi_score = -20
        else:
            rsi_score = 0
        score += rsi_score * 0.25
        details["RSI"] = round(rsi_score, 1)
    
    if "macd" in hist.columns and "macd_signal" in hist.columns:
        macd = last["macd"]
        signal = last["macd_signal"]
        hist_macd = last.get("macd_hist", 0)
        if macd > signal and hist_macd > 0:
            macd_score = 30
        elif macd < signal and hist_macd < 0:
            macd_score = -30
        else:
            macd_score = 0
        score += macd_score * 0.20
        details["MACD"] = round(macd_score, 1)
    
    if all(c in hist.columns for c in ["bb_upper", "bb_lower", "close"]):
        bb_range = last["bb_upper"] - last["bb_lower"]
        if bb_range > 0:
            pct_b = (last["close"] - last["bb_lower"]) / bb_range
            if pct_b < 0.2:
                bb_score = 25
            elif pct_b > 0.8:
                bb_score = -25
            else:
                bb_score = 0
            score += bb_score * 0.15
            details["Bollinger %B"] = round(bb_score, 1)
    
    if "sma50" in hist.columns and "sma200" in hist.columns:
        if last["close"] > last["sma50"] > last["sma200"]:
            sma_score = 30
        elif last["close"] < last["sma50"] < last["sma200"]:
            sma_score = -30
        elif last["close"] > last["sma50"]:
            sma_score = 10
        elif last["close"] < last["sma50"]:
            sma_score = -10
        else:
            sma_score = 0
        score += sma_score * 0.25
        details["SMA Trend"] = round(sma_score, 1)
    
    period_avg = hist["close"].mean()
    dist = ((last["close"] / period_avg) - 1) * 100
    if dist < -8:
        mean_score = 15
    elif dist > 12:
        mean_score = -15
    else:
        mean_score = 0
    score += mean_score * 0.15
    details["Mean Reversion"] = round(mean_score, 1)
    
    return {"score": max(-100, min(100, score)), "details": details}


def compute_positioning_score(futures: Dict) -> Dict[str, float]:
    score = 0.0
    details = {}
    
    long_pct = futures.get("long_pct", 50.0)
    funding = futures.get("funding_rate")
    
    if long_pct > 65:
        pos_score = -30
    elif long_pct > 55:
        pos_score = -10
    elif long_pct < 35:
        pos_score = 30
    elif long_pct < 45:
        pos_score = 10
    else:
        pos_score = 0
    score += pos_score * 0.6
    details["Long/Short Positioning"] = round(pos_score, 1)
    
    if funding is not None:
        if funding > 0.05:
            fund_score = -25
        elif funding > 0.02:
            fund_score = -10
        elif funding < -0.03:
            fund_score = 20
        else:
            fund_score = 0
        score += fund_score * 0.4
        details["Funding Rate"] = round(fund_score, 1)
    
    return {"score": max(-100, min(100, score)), "details": details}


def compute_sentiment_score(fear_greed_value: str, news_articles: List[Dict]) -> Dict[str, float]:
    score = 0.0
    details = {}
    
    try:
        fng = float(fear_greed_value)
        if fng < 20:
            fng_score = 40
        elif fng < 35:
            fng_score = 20
        elif fng > 80:
            fng_score = -40
        elif fng > 65:
            fng_score = -20
        else:
            fng_score = 0
        score += fng_score * 0.55
        details["Fear & Greed"] = round(fng_score, 1)
    except (ValueError, TypeError):
        details["Fear & Greed"] = 0.0
    
    if news_articles:
        sia = SIA()
        scores = [sia.polarity_scores(a["title"])["compound"] for a in news_articles]
        avg_sent = np.mean(scores) if scores else 0.0
        news_score = avg_sent * 50
        score += news_score * 0.45
        details["News Sentiment"] = round(news_score, 1)
    else:
        details["News Sentiment"] = 0.0
    
    return {"score": max(-100, min(100, score)), "details": details}


def compute_macro_score(macro_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    score = 0.0
    details = {}
    
    if not macro_data:
        return {"score": 0.0, "details": {}}
    
    for name in ["S&P 500 Index", "NASDAQ 100 Index"]:
        if name in macro_data and not macro_data[name].empty:
            df = macro_data[name]
            if len(df) > 5:
                ret = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
                if ret > 2:
                    score += 15
                elif ret < -2:
                    score -= 15
    
    if "DXY (U.S. Dollar Index)" in macro_data:
        df = macro_data["DXY (U.S. Dollar Index)"]
        if len(df) > 5:
            ret = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
            if ret > 1.5:
                score -= 20
                details["DXY Pressure"] = -20
            elif ret < -1.5:
                score += 15
                details["DXY Tail-wind"] = 15
    
    if "U.S. 10Y Treasury Yield" in macro_data:
        df = macro_data["U.S. 10Y Treasury Yield"]
        if len(df) > 5:
            ret = (df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100
            if ret > 5:
                score -= 15
                details["Yield Rise"] = -15
    
    details["Overall Macro"] = round(score, 1)
    return {"score": max(-100, min(100, score)), "details": details}


def generate_prediction(
    ticker: str,
    hist: pd.DataFrame,
    futures: Dict,
    fear_greed_value: str,
    news: List[Dict],
    macro_data: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    tech = compute_technical_score(hist)
    pos = compute_positioning_score(futures)
    sent = compute_sentiment_score(fear_greed_value, news)
    macro = compute_macro_score(macro_data)
    
    weights = {
        "Technical": 0.40,
        "Positioning": 0.25,
        "Sentiment": 0.25,
        "Macro": 0.10,
    }
    
    overall = (
        tech["score"] * weights["Technical"]
        + pos["score"] * weights["Positioning"]
        + sent["score"] * weights["Sentiment"]
        + macro["score"] * weights["Macro"]
    )
    
    if overall >= 35:
        bias = "Strong Bullish"
        color = "#00FF7F"
    elif overall >= 15:
        bias = "Bullish"
        color = "#90EE90"
    elif overall <= -35:
        bias = "Strong Bearish"
        color = "#FF4500"
    elif overall <= -15:
        bias = "Bearish"
        color = "#FF7F50"
    else:
        bias = "Neutral / Mixed"
        color = "#00FFFF"
    
    last_close = float(hist["close"].iloc[-1])
    recent_vol = hist["close"].pct_change().std() * 100
    if pd.isna(recent_vol) or recent_vol == 0:
        recent_vol = 2.5
    
    upside = last_close * (1 + (recent_vol * 1.8) / 100)
    downside = last_close * (1 - (recent_vol * 1.8) / 100)
    
    return {
        "overall_score": round(overall, 1),
        "bias": bias,
        "color": color,
        "components": {
            "Technical": tech,
            "Positioning": pos,
            "Sentiment": sent,
            "Macro": macro,
        },
        "projected_upside": upside,
        "projected_downside": downside,
        "current_price": last_close,
        "recent_volatility_pct": round(recent_vol, 2),
    }

# ============================================================
# 5. CHART BUILDING ENGINE
# ============================================================
def build_asset_chart(
    h_data: pd.DataFrame,
    show_candles: bool,
    show_sma50: bool,
    show_sma200: bool,
    show_bb: bool,
    show_rsi: bool,
    show_macd: bool,
    show_stoch: bool,
    show_atr: bool,
    show_vol_profile: bool,
    show_baseline: bool,
    show_long_baseline: bool,
    show_vwap: bool,
    show_vs_btc: bool,
    btc_data: Optional[pd.DataFrame] = None,
    patterns: Optional[List[Dict]] = None,
) -> go.Figure:
    h_data = h_data.copy()
    if show_baseline:
        h_data["baseline"] = h_data["close"].mean()
    if show_vwap and "volume" in h_data.columns:
        h_data["vwap"] = (h_data["close"] * h_data["volume"]).cumsum() / h_data["volume"].cumsum()
    if show_vs_btc and btc_data is not None:
        common_idx = h_data.index.intersection(btc_data.index)
        if len(common_idx) > 5:
            asset_norm = h_data.loc[common_idx, "close"] / h_data.loc[common_idx, "close"].iloc[0]
            btc_norm = btc_data.loc[common_idx, "close"] / btc_data.loc[common_idx, "close"].iloc[0]
            h_data.loc[common_idx, "vs_btc"] = (asset_norm / btc_norm - 1) * 100
    rows = 1
    if show_rsi: rows += 1
    if show_macd: rows += 1
    if show_stoch: rows += 1
    if show_atr: rows += 1
    if show_vol_profile: rows += 1
    if show_vs_btc and "vs_btc" in h_data.columns: rows += 1
    row_heights = [0.45] + [0.11] * (rows - 1) if rows > 1 else [1.0]
    subplot_titles = ["Price"]
    if show_rsi: subplot_titles.append("RSI (14)")
    if show_macd: subplot_titles.append("MACD")
    if show_stoch: subplot_titles.append("Stochastic")
    if show_atr: subplot_titles.append("ATR (14)")
    if show_vol_profile: subplot_titles.append("Volume Profile")
    if show_vs_btc and "vs_btc" in h_data.columns: subplot_titles.append("vs BTC (%)")
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=row_heights,
        subplot_titles=subplot_titles,
    )
    current_row = 1
    has_ohlc = all(col in h_data.columns for col in ["open", "high", "low", "close"])
    if show_candles and has_ohlc:
        fig.add_trace(
            go.Candlestick(
                x=h_data.index,
                open=h_data["open"], high=h_data["high"],
                low=h_data["low"], close=h_data["close"],
                name="Price",
                increasing_line_color="#00FF7F",
                decreasing_line_color="#FF4500",
                increasing_fillcolor="#00FF7F",
                decreasing_fillcolor="#FF4500",
                line=dict(width=1),
            ),
            row=current_row, col=1,
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=h_data.index, y=h_data["close"],
                name="Price", line=dict(color="#00FFFF", width=2),
            ),
            row=current_row, col=1,
        )
    if show_bb and "bb_upper" in h_data.columns:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["bb_upper"], name="BB Upper",
                                 line=dict(color="rgba(255,255,255,0.3)", width=1), showlegend=False), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["bb_lower"], name="BB Lower",
                                 line=dict(color="rgba(255,255,255,0.3)", width=1),
                                 fill="tonexty", fillcolor="rgba(0,255,127,0.08)"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["bb_mid"], name="BB Mid",
                                 line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dot")), row=current_row, col=1)
    if show_sma50 and "sma50" in h_data.columns:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["sma50"], name="SMA 50",
                                 line=dict(color="#00FF7F", width=1.5, dash="dash")), row=current_row, col=1)
    if show_sma200 and "sma200" in h_data.columns:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["sma200"], name="SMA 200",
                                 line=dict(color="#FF4500", width=1.5)), row=current_row, col=1)
    if show_baseline and "baseline" in h_data.columns:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["baseline"], name="Hist. Baseline (Avg)",
                                 line=dict(color="#FFD700", width=1.8, dash="dot")), row=current_row, col=1)
    if show_vwap and "vwap" in h_data.columns:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["vwap"], name="VWAP",
                                 line=dict(color="#FF00FF", width=1.6)), row=current_row, col=1)
    
    # --- PATTERN OVERLAY ---
    if patterns:
        fig = add_patterns_to_chart(fig, patterns, row=current_row)
    
    if show_rsi and "rsi" in h_data.columns:
        current_row += 1
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["rsi"], name="RSI",
                                 line=dict(color="#FFD700", width=1.5)), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="lime", row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
    if show_macd and "macd_hist" in h_data.columns:
        current_row += 1
        colors = ["#00FF7F" if v >= 0 else "#FF4500" for v in h_data["macd_hist"]]
        fig.add_trace(go.Bar(x=h_data.index, y=h_data["macd_hist"], name="Histogram",
                             marker_color=colors, showlegend=False), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["macd"], name="MACD",
                                 line=dict(color="#00BFFF", width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["macd_signal"], name="Signal",
                                 line=dict(color="#FF69B4", width=1.5)), row=current_row, col=1)
    if show_stoch and "stoch_k" in h_data.columns:
        current_row += 1
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["stoch_k"], name="%K",
                                 line=dict(color="#00FFFF", width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["stoch_d"], name="%D",
                                 line=dict(color="#FFD700", width=1.5)), row=current_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="lime", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
    if show_atr and "atr" in h_data.columns:
        current_row += 1
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["atr"], name="ATR",
                                 line=dict(color="#FF8C00", width=1.8)), row=current_row, col=1)
    if show_vol_profile and "volume" in h_data.columns:
        current_row += 1
        try:
            price_bins = pd.cut(h_data["close"], bins=24)
            vol_profile = h_data.groupby(price_bins, observed=True)["volume"].sum()
            bin_centers = [interval.mid for interval in vol_profile.index]
            fig.add_trace(go.Bar(x=vol_profile.values, y=bin_centers, orientation="h",
                                 name="Vol Profile", marker_color="rgba(0, 191, 255, 0.45)", showlegend=False),
                          row=current_row, col=1)
        except Exception:
            pass
    if show_vs_btc and "vs_btc" in h_data.columns:
        current_row += 1
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["vs_btc"], name="vs BTC %",
                                 line=dict(color="#00FFAA", width=1.8)), row=current_row, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=current_row, col=1)
    fig.update_layout(
        height=220 + (rows * 150),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.02),
        xaxis_rangeslider_visible=False,
        font=dict(family="Playfair Display, serif"),
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(rangeslider_visible=False, row=i, col=1)
    return fig

# ============================================================
# 6. SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Controls")

# --- WATCHLIST ---
st.sidebar.subheader("⭐ Watchlist")
if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

wl_input = st.sidebar.text_input("Add to Watchlist (ticker)", placeholder="e.g. BTC-USD")
if st.sidebar.button("➕ Add to Watchlist") and wl_input.strip():
    t = wl_input.strip().upper()
    if not t.endswith("-USD") and t.isalpha():
        t = t + "-USD"
    if t not in st.session_state.watchlist:
        st.session_state.watchlist.append(t)
        save_watchlist(st.session_state.watchlist)
        st.sidebar.success(f"Added {t}")

if st.session_state.watchlist:
    st.sidebar.write("**Your Watchlist:**")
    for i, t in enumerate(st.session_state.watchlist):
        cols = st.sidebar.columns([3, 1])
        cols[0].write(t)
        if cols[1].button("x", key=f"rm_wl_{i}"):
            st.session_state.watchlist.pop(i)
            save_watchlist(st.session_state.watchlist)
            st.rerun()
    if st.sidebar.button("Use Watchlist as Tickers"):
        t_raw = ", ".join(st.session_state.watchlist)
else:
    st.sidebar.caption("No tickers in watchlist yet.")

# --- MAIN TICKER INPUT ---
t_raw = st.sidebar.text_input("Enter Tickers (commas):", value="BTC-USD, ETH-USD, SOL-USD, XRP-USD", key="main_tickers")
tkrs = [t.strip().upper() for t in t_raw.split(",") if t.strip()]
res_disp = st.sidebar.selectbox("Resolution", list(RESOLUTION_MAP.keys()))
rf_rate = st.sidebar.slider("Refresh Rate (Seconds)", 15, 120, 45)

st.sidebar.subheader("Chart Type & Indicators")
show_candles = st.sidebar.checkbox("Candlestick Chart", value=True)
show_sma50 = st.sidebar.checkbox("50 SMA", value=True)
show_sma200 = st.sidebar.checkbox("200 SMA", value=True)
show_bb = st.sidebar.checkbox("Bollinger Bands (20,2)", value=True)
show_rsi = st.sidebar.checkbox("RSI (14)", value=True)
show_macd = st.sidebar.checkbox("MACD (12,26,9)", value=True)
show_stoch = st.sidebar.checkbox("Stochastic %K/%D", value=False)
show_atr = st.sidebar.checkbox("ATR (14)", value=False)
show_vol_profile = st.sidebar.checkbox("Volume Profile", value=True)

st.sidebar.subheader("Pattern Recognition")
show_patterns = st.sidebar.checkbox("Chart Pattern Recognition", value=False)
st.sidebar.caption("Doji • Hammer • Engulfing • Double Top/Bottom • Support/Resistance")

st.sidebar.subheader("Overlays & On-chain")
show_baseline = st.sidebar.checkbox("Historical Baseline (Period Avg)", value=True)
show_long_baseline = st.sidebar.checkbox("Long-term Baseline (1Y Avg)", value=False)
show_vwap = st.sidebar.checkbox("VWAP", value=False)
show_vs_btc = st.sidebar.checkbox("Relative Strength vs BTC", value=False)
show_onchain = st.sidebar.checkbox("Show On-chain Metrics", value=True)

# --- PORTFOLIO ---
st.sidebar.subheader("💼 Simple Portfolio")
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()

with st.sidebar.expander("Add Holding"):
    p_ticker = st.text_input("Ticker", placeholder="BTC-USD", key="p_ticker")
    p_qty = st.number_input("Quantity", min_value=0.0, value=0.0, step=0.01, key="p_qty")
    p_entry = st.number_input("Entry Price ($)", min_value=0.0, value=0.0, step=0.01, key="p_entry")
    if st.button("Add to Portfolio") and p_ticker and p_qty > 0 and p_entry > 0:
        t = p_ticker.strip().upper()
        if not t.endswith("-USD") and t.isalpha():
            t = t + "-USD"
        st.session_state.portfolio.append({"ticker": t, "qty": p_qty, "entry_price": p_entry})
        save_portfolio(st.session_state.portfolio)
        st.success(f"Added {p_qty} {t}")
        st.rerun()

if st.session_state.portfolio:
    st.sidebar.write("**Holdings:**")
    for i, h in enumerate(st.session_state.portfolio):
        st.sidebar.caption(f"{h['qty']} {h['ticker']} @ ${h['entry_price']:,.2f}")
        if st.sidebar.button(f"Remove {h['ticker']}", key=f"rm_pf_{i}"):
            st.session_state.portfolio.pop(i)
            save_portfolio(st.session_state.portfolio)
            st.rerun()

# --- ALERTS ---
st.sidebar.subheader("🔔 Alerts")
if "alerts" not in st.session_state:
    st.session_state.alerts = load_alerts()

with st.sidebar.expander("Create New Alert"):
    a_ticker = st.text_input("Ticker", placeholder="BTC-USD", key="a_ticker")
    a_type = st.selectbox("Alert Type", ["price", "rsi", "funding", "pattern"], key="a_type")
    
    if a_type == "pattern":
        a_pattern = st.selectbox("Pattern", ["Hammer", "Doji", "Bullish Engulfing", "Bearish Engulfing", "Double Top", "Double Bottom", "Support", "Resistance"], key="a_pattern")
        a_condition = "detected"
        a_value = 0.0
    else:
        a_condition = st.selectbox("Condition", ["above", "below"], key="a_cond")
        a_value = st.number_input("Value", value=0.0, key="a_value")
        a_pattern = ""
    
    if st.button("Add Alert") and a_ticker.strip():
        t = a_ticker.strip().upper()
        if not t.endswith("-USD") and t.isalpha():
            t = t + "-USD"
        new_alert = {
            "ticker": t,
            "type": a_type,
            "condition": a_condition,
            "value": a_value,
            "pattern_type": a_pattern,
        }
        st.session_state.alerts.append(new_alert)
        save_alerts(st.session_state.alerts)
        st.success("Alert added")
        st.rerun()

if st.session_state.alerts:
    st.sidebar.write(f"**Active Alerts ({len(st.session_state.alerts)}):**")
    for i, a in enumerate(st.session_state.alerts):
        if a["type"] == "pattern":
            desc = f"{a['ticker']} → {a['pattern_type']}"
        else:
            desc = f"{a['ticker']} {a['type']} {a['condition']} {a['value']}"
        cols = st.sidebar.columns([4, 1])
        cols[0].caption(desc)
        if cols[1].button("x", key=f"rm_alert_{i}"):
            st.session_state.alerts.pop(i)
            save_alerts(st.session_state.alerts)
            st.rerun()
else:
    st.sidebar.caption("No active alerts")
res_cfg = RESOLUTION_MAP[res_disp]
interval = res_cfg["interval"]
lookback_bars = res_cfg["lookback"]
period = res_cfg["period"]
is_6h = res_cfg["is_6h"]
st_autorefresh(interval=rf_rate * 1000, key="datarefresh")

# ============================================================
# 7. MAIN LAYOUT
# ============================================================
active_users = get_active_user_count(timeout_seconds=90)

col_title, col_users = st.columns([5.5, 1])
with col_title:
    st.markdown('<div class="silver-title">OneStop-InfoSpot</div>', unsafe_allow_html=True)
    st.markdown('<div class="silver-subtitle">Asset Intelligence Terminal</div>', unsafe_allow_html=True)
with col_users:
    st.metric("🟢 Active Users", active_users)

st.caption("Dark mode • Candlesticks • Pattern Recognition • Long/Short Allocation • Open Interest • Funding • RSI • MACD • Bollinger • On-chain • Global Macro • 15Y Historical • Predictive Analytics")

tab_crypto, tab_macro, tab_predict, tab_analysis = st.tabs([
    "⚡ Crypto Intelligence Terminal",
    "🌐 Global Macro Insights",
    "🔮 Predictive Analytics",
    "📊 Analysis Tools"
])

with tab_crypto:
    if not tkrs:
        st.info("Enter at least one ticker in the sidebar.")
    else:
        data = load_ticker_data(tuple(tkrs), interval, lookback_bars, period, is_6h)
        if not data:
            st.error("No valid data loaded. Check tickers or try a different resolution. Some data sources may be temporarily unavailable.")
        else:
            # ----- Evaluate Alerts -----
            patterns_by_ticker = {}
            if show_patterns:
                for t, hist in data.items():
                    try:
                        patterns_by_ticker[t] = detect_all_patterns(hist)
                    except Exception:
                        patterns_by_ticker[t] = []
            
            # Build temporary liquidity for alert checking
            temp_liquidity = {}
            for t, hist in data.items():
                if not hist.empty and "close" in hist.columns:
                    try:
                        cpx = float(hist["close"].iloc[-1])
                        fut = get_futures_metrics(t, current_price=cpx)
                        temp_liquidity[t] = {
                            "longs": fut.get("long_pct", 50),
                            "shorts": fut.get("short_pct", 50),
                            "funding": fut.get("funding_rate"),
                        }
                    except Exception:
                        pass
            
            if "alerts" in st.session_state and st.session_state.alerts:
                triggered = check_alerts(
                    st.session_state.alerts,
                    data,
                    temp_liquidity,
                    patterns_by_ticker
                )
                if triggered:
                    for msg in triggered:
                        st.warning(msg)

            daily_pivots = get_daily_pivots(tuple(tkrs))
            btc_data = None
            if show_vs_btc:
                if "BTC-USD" in data:
                    btc_data = data["BTC-USD"]
                else:
                    try:
                        btc_raw = load_ticker_data(("BTC-USD",), interval, lookback_bars, period, is_6h)
                        btc_data = btc_raw.get("BTC-USD")
                    except Exception:
                        pass
            combined_df = pd.DataFrame()
            matrix_rows = []
            liquidity_profiles = {}
            onchain_profiles = {}
            futures_table_rows = []
            for t, hist in data.items():
                if hist.empty or "close" not in hist.columns:
                    continue
                s_px = float(hist["close"].iloc[0])
                combined_df[t] = ((hist["close"] / s_px) - 1) * 100
                c_px = float(hist["close"].iloc[-1])
                p_px = float(hist["close"].iloc[-2]) if len(hist) > 1 else c_px
                delta = ((c_px - p_px) / p_px) * 100
                fut = get_futures_metrics(t, current_price=c_px)
                long_r = fut["long_pct"] if fut["long_pct"] is not None else 50.0
                short_r = fut["short_pct"] if fut["short_pct"] is not None else 50.0
                oi_val = fut["oi"]
                funding = fut["funding_rate"]
                source_label = fut["source"]
                liquidity_profiles[t] = {
                    "price": c_px,
                    "100x_l": c_px * 0.99, "50x_l": c_px * 0.98, "25x_l": c_px * 0.96,
                    "100x_s": c_px * 1.01, "50x_s": c_px * 1.02, "25x_s": c_px * 1.04,
                    "longs": long_r, "shorts": short_r,
                    "oi": oi_val, "funding": funding, "source": source_label,
                }
                futures_table_rows.append({
                    "Ticker": t,
                    "Long %": f"{long_r:.1f}%",
                    "Short %": f"{short_r:.1f}%",
                    "Open Interest": f"{oi_val:,.0f}" if oi_val else "N/A",
                    "Funding Rate": f"{funding:+.4f}%" if funding is not None else "N/A",
                    "Source": source_label,
                })
                if show_onchain and ("-USD" in t or t.replace("-", "").isalpha()):
                    onchain_profiles[t] = get_real_onchain(t)
                if t in daily_pivots:
                    p = daily_pivots[t]
                    matrix_rows.append({
                        "Ticker": t, "Price": f"${c_px:,.2f}", "Change": f"{delta:+.2f}%",
                        "Long %": f"{long_r:.1f}%", "Short %": f"{short_r:.1f}%",
                        "Pivot": f"${p['Pivot']:,.2f}", "S1": f"${p['S1']:,.2f}", "S2": f"${p['S2']:,.2f}",
                        "R1": f"${p['R1']:,.2f}", "R2": f"${p['R2']:,.2f}",
                    })
                else:
                    matrix_rows.append({
                        "Ticker": t, "Price": f"${c_px:,.2f}", "Change": f"{delta:+.2f}%",
                        "Long %": f"{long_r:.1f}%", "Short %": f"{short_r:.1f}%",
                        "Pivot": "N/A", "S1": "N/A", "S2": "N/A", "R1": "N/A", "R2": "N/A",
                    })
            st.subheader("📊 Comparative Relative Performance")
            if not combined_df.empty:
                fig = go.Figure()
                for col in combined_df.columns:
                    fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df[col], name=col))
                fig.update_layout(height=320, template="plotly_dark",
                                  paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                  margin=dict(l=10, r=10, t=30, b=10),
                                  font=dict(family="Playfair Display, serif"))
                st.plotly_chart(fig, use_container_width=True)
            
            # ========== PORTFOLIO SUMMARY ==========
            if st.session_state.get("portfolio"):
                st.subheader("💼 Portfolio Performance")
                pf_rows = []
                total_cost = 0.0
                total_value = 0.0
                for h in st.session_state.portfolio:
                    t = h["ticker"]
                    qty = float(h["qty"])
                    entry = float(h["entry_price"])
                    cost = qty * entry
                    total_cost += cost
                    current = None
                    if t in data and not data[t].empty and "close" in data[t].columns:
                        current = float(data[t]["close"].iloc[-1])
                    if current is not None:
                        value = qty * current
                        pnl = value - cost
                        pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
                        total_value += value
                        pf_rows.append({
                            "Ticker": t,
                            "Qty": qty,
                            "Entry": f"${entry:,.2f}",
                            "Current": f"${current:,.2f}",
                            "Value": f"${value:,.2f}",
                            "P&L": f"${pnl:+,.2f}",
                            "P&L %": f"{pnl_pct:+.2f}%",
                        })
                    else:
                        pf_rows.append({
                            "Ticker": t,
                            "Qty": qty,
                            "Entry": f"${entry:,.2f}",
                            "Current": "N/A",
                            "Value": "N/A",
                            "P&L": "N/A",
                            "P&L %": "N/A",
                        })
                if pf_rows:
                    st.dataframe(pd.DataFrame(pf_rows), use_container_width=True, hide_index=True)
                    if total_cost > 0 and total_value > 0:
                        total_pnl = total_value - total_cost
                        total_pnl_pct = (total_pnl / total_cost) * 100
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Total Cost", f"${total_cost:,.2f}")
                        c2.metric("Current Value", f"${total_value:,.2f}")
                        c3.metric("Total P&L", f"${total_pnl:+,.2f}", f"{total_pnl_pct:+.2f}%")

            st.subheader("🔥 Asset Analysis Frameworks")
            m_tgt = st.selectbox("Select Target Asset:", list(liquidity_profiles.keys()))
            if m_tgt and m_tgt in data:
                h_data = data[m_tgt].copy()
                if show_long_baseline:
                    long_avg = get_long_term_avg(m_tgt)
                    if long_avg is not None:
                        h_data["long_baseline"] = long_avg
                
                # --- PATTERN DETECTION ---
                detected_patterns = []
                if show_patterns:
                    detected_patterns = detect_all_patterns(h_data)
                
                fig = build_asset_chart(
                    h_data=h_data,
                    show_candles=show_candles,
                    show_sma50=show_sma50,
                    show_sma200=show_sma200,
                    show_bb=show_bb,
                    show_rsi=show_rsi,
                    show_macd=show_macd,
                    show_stoch=show_stoch,
                    show_atr=show_atr,
                    show_vol_profile=show_vol_profile,
                    show_baseline=show_baseline,
                    show_long_baseline=show_long_baseline,
                    show_vwap=show_vwap,
                    show_vs_btc=show_vs_btc,
                    btc_data=btc_data,
                    patterns=detected_patterns if show_patterns else None,
                )
                if show_long_baseline and "long_baseline" in h_data.columns:
                    fig.add_trace(
                        go.Scatter(x=h_data.index, y=h_data["long_baseline"], name="1Y Baseline",
                                   line=dict(color="#FF8C00", width=1.8, dash="dashdot")),
                        row=1, col=1,
                    )
                st.plotly_chart(fig, use_container_width=True)
                
                # --- PATTERN SUMMARY BOX ---
                if show_patterns:
                    if detected_patterns:
                        st.markdown("#### 🔎 Detected Chart Patterns")
                        pattern_df = pd.DataFrame([
                            {
                                "Pattern": p["type"],
                                "Direction": p["direction"],
                                "Confidence": p["confidence"],
                                "Price": f"${p['price']:,.2f}",
                                "Description": p["description"],
                            }
                            for p in detected_patterns
                        ])
                        st.dataframe(pattern_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No significant chart patterns detected in the current lookback window.")
                
                cols = st.columns(5)
                if show_rsi and "rsi" in h_data.columns and not h_data["rsi"].isna().all():
                    rsi_val = h_data["rsi"].iloc[-1]
                    cols[0].metric("RSI", f"{rsi_val:.1f}",
                                   "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral")
                if show_macd and "macd" in h_data.columns:
                    cols[1].metric("MACD", f"{h_data['macd'].iloc[-1]:.2f}")
                if show_bb and "bb_upper" in h_data.columns:
                    bb_range = h_data["bb_upper"].iloc[-1] - h_data["bb_lower"].iloc[-1]
                    if bb_range > 0:
                        bb_pos = (h_data["close"].iloc[-1] - h_data["bb_lower"].iloc[-1]) / bb_range
                        cols[2].metric("%B (BB)", f"{bb_pos * 100:.1f}%")
                if show_baseline:
                    baseline = h_data["close"].mean()
                    dist = ((h_data["close"].iloc[-1] / baseline) - 1) * 100
                    cols[3].metric("vs Baseline", f"{dist:+.1f}%")
                if show_vs_btc and "vs_btc" in h_data.columns:
                    cols[4].metric("vs BTC", f"{h_data['vs_btc'].iloc[-1]:+.1f}%")
                st.markdown("#### 🔥 Liquidity Map (Estimated Liquidation Levels)")
                p = liquidity_profiles[m_tgt]
                st.caption(f"Source: **{p['source']}** | Order book distribution values are calculated continuously")
                levels = [p["25x_l"], p["50x_l"], p["100x_l"], p["price"], p["100x_s"], p["50x_s"], p["25x_s"]]
                labels = ["25x Long Liq", "50x Long Liq", "100x Long Liq", "Current Price",
                          "100x Short Liq", "50x Short Liq", "25x Short Liq"]
                fig_map = go.Figure(go.Bar(x=labels, y=levels, text=[f"${v:,.2f}" for v in levels], textposition="auto",
                                           marker=dict(color=levels, colorscale="Portland")))
                fig_map.update_layout(height=340, template="plotly_dark",
                                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                      font=dict(family="Playfair Display, serif"))
                st.plotly_chart(fig_map, use_container_width=True)
            st.subheader("📊 Real-Time Futures Metrics")
            st.caption("Long/Short Account Ratio • Open Interest • Funding Rate  —  Source: OKX")
            if futures_table_rows:
                st.dataframe(pd.DataFrame(futures_table_rows).set_index("Ticker"), use_container_width=True)
            st.subheader("📊 Long vs Short Allocation")
            if liquidity_profiles:
                c_as = list(liquidity_profiles.keys())
                fig_c = go.Figure()
                fig_c.add_trace(go.Bar(x=c_as, y=[liquidity_profiles[k]["longs"] for k in c_as],
                                       name="Long %", marker_color="#00FF7F",
                                       text=[f"{liquidity_profiles[k]['longs']:.1f}%" for k in c_as], textposition="auto"))
                fig_c.add_trace(go.Bar(x=c_as, y=[liquidity_profiles[k]["shorts"] for k in c_as],
                                       name="Short %", marker_color="#FF4500",
                                       text=[f"{liquidity_profiles[k]['shorts']:.1f}%" for k in c_as], textposition="auto"))
                fig_c.update_layout(height=320, barmode="group", template="plotly_dark",
                                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", yaxis_title="%",
                                    font=dict(family="Playfair Display, serif"))
                st.plotly_chart(fig_c, use_container_width=True)
            if show_onchain and onchain_profiles:
                st.subheader("🔗 Real On-chain Metrics")
                for ticker, metrics in onchain_profiles.items():
                    st.markdown(f"**{ticker}**")
                    st.dataframe(pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]),
                                 use_container_width=True, hide_index=True)
            if matrix_rows:
                st.subheader("🧮 Side-by-Side Pivot Matrix (Daily Pivots)")
                st.caption("Classic Floor Pivots calculated from the previous daily candle close parameters")
                st.dataframe(pd.DataFrame(matrix_rows).set_index("Ticker"), use_container_width=True)
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📰 Live Crypto News")
                pulse_key = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                news = get_crypto_news(cache_buster=pulse_key)
                if news:
                    sia = SIA()
                    for art in news:
                        with st.expander(art["title"]):
                            score = sia.polarity_scores(art["title"])["compound"]
                            if score >= 0.05:
                                status_badge = '<span style="color:#00FF7F; font-weight:bold;">Bullish</span>'
                            elif score <= -0.05:
                                status_badge = '<span style="color:#FF4500; font-weight:bold;">Bearish</span>'
                            else:
                                status_badge = '<span style="color:#00FFFF; font-weight:bold;">Neutral</span>'
                            
                            html_body = f"""
                            <div class="news-box">
                                <p class="news-meta">
                                    <strong>{art['source']}</strong> • {art['published']} • Sentiment: {status_badge} ({score:+.2f})
                                </p>
                                <p class="news-summary">{art['summary'] if art['summary'] else ''}</p>
                                <a href="{art['link']}" target="_blank" class="news-action-link">Open Article ↗</a>
                            </div>
                            """
                            st.markdown(html_body, unsafe_allow_html=True)
                else:
                    st.info("No news articles available right now. Try refreshing.")
            with col2:
                st.subheader("📊 Sentiment")
                fng_v, fng_c = get_fear_greed()
                st.metric("Fear & Greed Index", f"{fng_v}/100", fng_c)

with tab_macro:
    st.subheader("🌐 Broad Market Telemetry Framework")
    st.caption("Asynchronous tracking layer pulling multi-asset global correlations over a rolling 90-day window.")
    
    macro_data = load_macro_data()
    
    if not macro_data:
        st.info("Macro matrix data endpoints temporarily unavailable. Retrying query package updates...")
    else:
        m_cols = st.columns(3)
        m_index = 0
        
        for name, m_df in macro_data.items():
            if not m_df.empty and "close" in m_df.columns:
                current_p = float(m_df["close"].iloc[-1])
                prior_p = float(m_df["close"].iloc[-2]) if len(m_df) > 1 else current_p
                pct_delta = ((current_p - prior_p) / prior_p) * 100
                
                col_target = m_cols[m_index % 3]
                if "Yield" in name:
                    col_target.metric(name, f"{current_p:.2f}%", f"{pct_delta:+.2f}%")
                else:
                    col_target.metric(name, f"${current_p:,.2f}", f"{pct_delta:+.2f}%")
                m_index += 1
        st.markdown("#### 📉 90-Day Normalized Comparative Performance Matrix")
        st.caption("All indices normalized to zero base from original baseline inception interval to pinpoint risk-on/risk-off decoupling triggers.")
        
        macro_combined = pd.DataFrame()
        for name, m_df in macro_data.items():
            if not m_df.empty and "close" in m_df.columns:
                base_p = float(m_df["close"].iloc[0])
                macro_combined[name] = ((m_df["close"] / base_p) - 1) * 100
                
        if not macro_combined.empty:
            fig_macro = go.Figure()
            for col in macro_combined.columns:
                fig_macro.add_trace(go.Scatter(x=macro_combined.index, y=macro_combined[col], name=col, line=dict(width=1.8)))
            
            fig_macro.update_layout(
                height=450,
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", y=1.12),
                font=dict(family="Playfair Display, serif")
            )
            st.plotly_chart(fig_macro, use_container_width=True)

# ============================================================
# PREDICTIVE ANALYTICS TAB
# ============================================================
with tab_predict:
    st.subheader("🔮 Multi-Factor Predictive Signal Engine")
    st.caption("Synthesizes Technicals • Futures Positioning • Sentiment • Macro into a single Bias Score. This is a decision-support tool, not financial advice.")
    
    if not tkrs:
        st.info("Enter at least one ticker in the sidebar to generate predictions.")
    else:
        pred_data = load_ticker_data(tuple(tkrs), interval, lookback_bars, period, is_6h)
        fng_v, fng_c = get_fear_greed()
        news = get_crypto_news(cache_buster=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
        macro_data = load_macro_data()
        
        if not pred_data:
            st.error("Unable to load market data for prediction.")
        else:
            pred_ticker = st.selectbox("Select asset for prediction:", list(pred_data.keys()), key="pred_select")
            
            if pred_ticker and pred_ticker in pred_data:
                hist = pred_data[pred_ticker]
                c_px = float(hist["close"].iloc[-1])
                futures = get_futures_metrics(pred_ticker, current_price=c_px)
                
                prediction = generate_prediction(
                    ticker=pred_ticker,
                    hist=hist,
                    futures=futures,
                    fear_greed_value=fng_v,
                    news=news,
                    macro_data=macro_data,
                )
                
                st.markdown("### Overall Market Bias")
                score_col1, score_col2, score_col3 = st.columns([1.5, 1, 1])
                
                with score_col1:
                    st.markdown(
                        f"""
                        <div style="background: linear-gradient(135deg, #1a1d24 0%, #0e1117 100%);
                                    border: 2px solid {prediction['color']};
                                    border-radius: 12px; padding: 20px; text-align: center;">
                            <div style="font-size: 0.9rem; color: #a0a0a0;">Bias Score</div>
                            <div style="font-size: 3.2rem; font-weight: 700; color: {prediction['color']};">
                                {prediction['overall_score']:+.1f}
                            </div>
                            <div style="font-size: 1.4rem; font-weight: 600; color: {prediction['color']};">
                                {prediction['bias']}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                
                with score_col2:
                    st.metric("Current Price", f"${prediction['current_price']:,.2f}")
                    st.metric("Recent Volatility", f"{prediction['recent_volatility_pct']:.2f}%")
                
                with score_col3:
                    st.metric("Projected Upside", f"${prediction['projected_upside']:,.2f}")
                    st.metric("Projected Downside", f"${prediction['projected_downside']:,.2f}")
                
                st.markdown("---")
                
                st.markdown("### Factor Breakdown")
                
                comp_cols = st.columns(4)
                component_names = ["Technical", "Positioning", "Sentiment", "Macro"]
                component_icons = ["📈", "⚖️", "📰", "🌐"]
                
                for i, name in enumerate(component_names):
                    comp = prediction["components"][name]
                    with comp_cols[i]:
                        color = "#00FF7F" if comp["score"] > 10 else "#FF4500" if comp["score"] < -10 else "#00FFFF"
                        st.markdown(
                            f"""
                            <div style="background:#1a1d24; border-radius:8px; padding:14px; text-align:center; border:1px solid #3b4252;">
                                <div style="font-size:1.5rem;">{component_icons[i]}</div>
                                <div style="font-size:0.85rem; color:#a0a0a0;">{name}</div>
                                <div style="font-size:1.8rem; font-weight:700; color:{color};">{comp['score']:+.1f}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                
                st.markdown("#### Detailed Signal Contributions")
                detail_rows = []
                for name in component_names:
                    for k, v in prediction["components"][name]["details"].items():
                        detail_rows.append({
                            "Factor Group": name,
                            "Signal": k,
                            "Contribution": f"{v:+.1f}"
                        })
                
                if detail_rows:
                    st.dataframe(
                        pd.DataFrame(detail_rows),
                        use_container_width=True,
                        hide_index=True,
                    )
                
                st.markdown("### Score Visualization")
                fig_score = go.Figure()
                
                fig_score.add_trace(go.Bar(
                    x=["Technical", "Positioning", "Sentiment", "Macro", "OVERALL"],
                    y=[
                        prediction["components"]["Technical"]["score"],
                        prediction["components"]["Positioning"]["score"],
                        prediction["components"]["Sentiment"]["score"],
                        prediction["components"]["Macro"]["score"],
                        prediction["overall_score"],
                    ],
                    marker_color=[
                        "#00BFFF", "#FFD700", "#FF69B4", "#90EE90", prediction["color"]
                    ],
                    text=[
                        f"{prediction['components']['Technical']['score']:+.1f}",
                        f"{prediction['components']['Positioning']['score']:+.1f}",
                        f"{prediction['components']['Sentiment']['score']:+.1f}",
                        f"{prediction['components']['Macro']['score']:+.1f}",
                        f"{prediction['overall_score']:+.1f}",
                    ],
                    textposition="auto",
                ))
                
                fig_score.update_layout(
                    height=340,
                    template="plotly_dark",
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#0e1117",
                    yaxis_title="Score (-100 to +100)",
                    yaxis=dict(range=[-100, 100]),
                    font=dict(family="Playfair Display, serif"),
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                fig_score.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig_score, use_container_width=True)
                
                st.info(
                    "⚠️ **Disclaimer**: This Predictive Signal Engine aggregates existing technical, positioning, "
                    "sentiment, and macro data already present in the app. It is a decision-support tool only and "
                    "does **not** constitute financial, investment, or trading advice. Markets can remain irrational "
                    "longer than models can stay solvent. Always do your own research."
                )

# ============================================================
# ANALYSIS TOOLS TAB
# ============================================================
with tab_analysis:
    st.subheader("📊 Advanced Analysis Tools")
    st.caption("Correlation • Volatility • Multi-period Performance — all derived from the currently loaded tickers.")

    if not tkrs:
        st.info("Enter at least one ticker in the sidebar to use the analysis tools.")
    else:
        analysis_data = load_ticker_data(tuple(tkrs), interval, lookback_bars, period, is_6h)

        if not analysis_data or len(analysis_data) == 0:
            st.warning("No valid data available for analysis. Try different tickers or a different resolution.")
        else:
            # ----- 1. Multi-Period Performance -----
            st.markdown("### 📈 Multi-Period Performance")
            perf_df = compute_multi_period_performance(analysis_data)
            if not perf_df.empty:
                # Format percentages
                display_perf = perf_df.copy()
                for col in ["1D %", "7D %", "30D %", "90D %", "YTD %"]:
                    if col in display_perf.columns:
                        display_perf[col] = display_perf[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                st.dataframe(display_perf.set_index("Ticker"), use_container_width=True)
            else:
                st.info("Not enough history to compute performance.")

            st.markdown("---")

            # ----- 2. Volatility Dashboard -----
            st.markdown("### 📉 Volatility Dashboard")
            vol_df = compute_volatility_stats(analysis_data)
            if not vol_df.empty:
                display_vol = vol_df.copy()
                display_vol["Price"] = display_vol["Price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
                display_vol["Vol 7d (ann%)"] = display_vol["Vol 7d (ann%)"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                display_vol["Vol 30d (ann%)"] = display_vol["Vol 30d (ann%)"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                display_vol["ATR"] = display_vol["ATR"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
                display_vol["ATR %"] = display_vol["ATR %"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                st.dataframe(display_vol.set_index("Ticker"), use_container_width=True)

                # Simple bar chart of 30d volatility
                try:
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(
                        x=vol_df["Ticker"],
                        y=vol_df["Vol 30d (ann%)"],
                        marker_color="#00BFFF",
                        text=vol_df["Vol 30d (ann%)"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else ""),
                        textposition="auto"
                    ))
                    fig_vol.update_layout(
                        title="30-Day Annualized Volatility",
                        height=320,
                        template="plotly_dark",
                        paper_bgcolor="#0e1117",
                        plot_bgcolor="#0e1117",
                        font=dict(family="Playfair Display, serif"),
                        margin=dict(l=10, r=10, t=40, b=10),
                        yaxis_title="Volatility %"
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)
                except Exception:
                    pass
            else:
                st.info("Not enough data for volatility stats.")

            st.markdown("---")

            # ----- 3. Correlation Matrix -----
            st.markdown("### 🔗 Correlation Matrix (Returns)")
            st.caption("Based on daily (or current resolution) returns of the loaded tickers. Values near +1 move together; near -1 move inversely.")
            corr = compute_correlation_matrix(analysis_data)
            if not corr.empty and len(corr) >= 2:
                try:
                    fig_corr = go.Figure(data=go.Heatmap(
                        z=corr.values,
                        x=corr.columns.tolist(),
                        y=corr.index.tolist(),
                        colorscale="RdBu",
                        zmid=0,
                        text=np.round(corr.values, 2),
                        texttemplate="%{text}",
                        textfont={"size": 12},
                        colorbar=dict(title="Corr")
                    ))
                    fig_corr.update_layout(
                        height=420,
                        template="plotly_dark",
                        paper_bgcolor="#0e1117",
                        plot_bgcolor="#0e1117",
                        font=dict(family="Playfair Display, serif"),
                        margin=dict(l=10, r=10, t=20, b=10),
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
                except Exception as e:
                    st.write(corr.round(2))
            else:
                st.info("Need at least 2 tickers with overlapping history to build a correlation matrix.")


st.sidebar.success(f"Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S')}")
