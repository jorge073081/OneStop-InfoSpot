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
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# 1. INITIAL SETUP & MATRIX INTERFACE STYLE SYSTEM
# ============================================================
try:
    nltk.download("vader_lexicon", quiet=True)
except Exception:
    pass  # non-fatal; sentiment features degrade gracefully

st.set_page_config(
    page_title="OneStop-InfoSpot | Asset Intelligence Terminal",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0b0e11;
            --surface: #12161c;
            --surface-2: #1a2030;
            --border: #2a3344;
            --text: #e8edf5;
            --muted: #8b97a8;
            --accent: #3b82f6;
            --bull: #22c55e;
            --bear: #ef4444;
        }

        html, body, .stApp,
        p, label, .stMarkdown, .stMarkdown p, .stCaption,
        h1, h2, h3, h4, h5, h6,
        .stTextInput input, .stSelectbox div, .stSlider div, .stCheckbox label,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        .stDataFrame, table, th, td,
        button, .stButton button, .stDownloadButton button,
        .stRadio label, .stTabs [data-baseweb="tab"] {
            font-family: 'IBM Plex Sans', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }

        .stApp { background-color: var(--bg) !important; color: var(--text) !important; }
        section[data-testid="stSidebar"] {
            background-color: var(--surface) !important;
            border-right: 1px solid var(--border);
        }
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown { color: var(--text) !important; }
        .stCaption, [data-testid="stCaption"] { color: var(--muted) !important; }

        .block-container {
            padding-top: 1.4rem !important;
            padding-bottom: 2rem !important;
            padding-left: 1.4rem !important;
            padding-right: 1.4rem !important;
            max-width: 1440px;
            overflow: visible !important;
        }

        .pulse-hero {
            text-align: center;
            padding: 1rem 1.25rem 1.15rem 1.25rem;
            margin: 0.5rem 0 1rem 0;
            border-radius: 12px;
            border: 1px solid var(--border);
            background: linear-gradient(180deg, #141a24 0%, #0b0e11 100%);
        }
        .pulse-label {
            font-size: 0.75rem;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.35rem;
        }
        .pulse-sym {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.15rem;
        }
        .pulse-chg {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 2.6rem;
            font-weight: 600;
            line-height: 1.15;
            margin: 0.2rem 0;
        }
        .pulse-chg.up { color: #22c55e; }
        .pulse-chg.down { color: #ef4444; }
        .pulse-sub {
            font-size: 0.85rem;
            color: var(--muted);
        }
        .pulse-tape {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            justify-content: center;
            margin-top: 0.75rem;
        }
        .pulse-chip {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.8rem;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: #12161c;
        }
        .pulse-chip.up { color: #22c55e; border-color: rgba(34,197,94,0.35); }
        .pulse-chip.down { color: #ef4444; border-color: rgba(239,68,68,0.35); }

        .app-header {

            text-align: center !important;
            padding: 0.35rem 0.5rem 0.6rem 0.5rem !important;
            margin: 0 0 0.35rem 0 !important;
            overflow: visible !important;
        }
        .silver-title {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-size: 1.85rem !important;
            font-weight: 600 !important;
            text-align: center !important;
            margin: 0.25rem 0 0.25rem 0 !important;
            padding: 0.15rem 0 !important;
            line-height: 1.35 !important;
            letter-spacing: -0.02em;
            color: var(--text) !important;
            background: none !important;
            -webkit-text-fill-color: unset !important;
            overflow: visible !important;
            white-space: normal !important;
        }
        .silver-subtitle {
            font-family: 'IBM Plex Sans', sans-serif !important;
            font-size: 0.9rem !important;
            font-weight: 400 !important;
            text-align: center !important;
            color: var(--muted) !important;
            margin: 0 0 0.5rem 0 !important;
            padding: 0 !important;
            line-height: 1.4 !important;
            letter-spacing: 0.02em;
            overflow: visible !important;
        }

        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.65rem 0.85rem !important;
        }
        [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 1.25rem !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            font-size: 0.72rem !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            font-weight: 500 !important;
        }
        [data-testid="stMetricDelta"] {
            font-family: 'IBM Plex Mono', monospace !important;
            font-size: 0.8rem !important;
        }

        .stTextInput input, .stSelectbox > div > div, .stNumberInput input {
            background-color: var(--surface-2) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            color: var(--text) !important;
        }
        .stTextInput input:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 1px var(--accent) !important;
        }

        .stButton > button, .stDownloadButton > button {
            background: var(--surface-2) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            font-weight: 500 !important;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            border-color: var(--accent) !important;
            background: #1e293b !important;
        }

        div[role="radiogroup"] {
            gap: 0.25rem !important;
            flex-wrap: wrap !important;
            padding: 0.25rem !important;
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            margin-bottom: 0.75rem !important;
        }
        div[role="radiogroup"] label {
            background: transparent !important;
            border-radius: 6px !important;
            padding: 0.35rem 0.65rem !important;
            color: var(--muted) !important;
            font-size: 0.82rem !important;
            font-weight: 500 !important;
        }
        div[role="radiogroup"] label:hover {
            color: var(--text) !important;
            background: var(--surface-2) !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }
        .stDataFrame, table, th, td { font-size: 0.85rem !important; }

        .streamlit-expanderHeader {
            font-weight: 500 !important;
            color: var(--text) !important;
            background: var(--surface) !important;
            border-radius: 6px !important;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-size: 0.8rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted) !important;
            font-weight: 600 !important;
        }

        hr { border-color: var(--border) !important; opacity: 0.7; }
        [data-testid="stAlert"] {
            border-radius: 8px !important;
            border: 1px solid var(--border) !important;
        }

        @media (max-width: 768px) {
            .silver-title { font-size: 1.4rem !important; line-height: 1.35 !important; }
            .silver-subtitle { font-size: 0.8rem !important; }
            .block-container {
                padding-top: 0.6rem !important;
                padding-left: 0.65rem !important;
                padding-right: 0.65rem !important;
                padding-bottom: 3.5rem !important;
            }
            [data-testid="stMetricValue"] { font-size: 1.05rem !important; }
            [data-testid="stMetricLabel"] { font-size: 0.65rem !important; }
            .stButton > button, .stDownloadButton > button {
                width: 100%;
                min-height: 2.6rem;
            }
            [data-testid="stDataFrame"], [data-testid="stTable"] {
                max-width: 100%;
                overflow-x: auto !important;
            }
            h2, h3 { font-size: 1.05rem !important; margin-top: 0.35rem !important; }
            div[role="radiogroup"] label {
                font-size: 0.72rem !important;
                padding: 0.3rem 0.45rem !important;
            }
        }

        @media (max-width: 480px) {
            .silver-title { font-size: 1.25rem !important; line-height: 1.35 !important; }
            [data-testid="stMetricValue"] { font-size: 0.95rem !important; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_sia():
    """Single shared VADER analyzer."""
    try:
        return SIA()
    except Exception:
        return None

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

def format_price(px) -> str:
    """Adaptive price precision so XRP etc. are not rounded to 2 dp."""
    try:
        if px is None or (isinstance(px, float) and np.isnan(px)):
            return "—"
        px = float(px)
    except (TypeError, ValueError):
        return "—"
    ax = abs(px)
    if ax >= 1000:
        return f"${px:,.2f}"
    if ax >= 50:
        return f"${px:,.2f}"
    if ax >= 1:
        return f"${px:,.4f}"
    if ax >= 0.01:
        return f"${px:,.5f}"
    return f"${px:,.8f}"


def safe_float(value, default=None) -> Optional[float]:

    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_user_tickers(raw_tickers: List[str]) -> List[str]:
    """
    Let users type BTC, ETH, SOL without -USD.
    Stocks/ETFs (AAPL, SPY, …) stay unchanged.
    """
    # Liquid equities / ETFs that must NOT get -USD
    equity = {
        "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "EFA", "EEM", "TLT", "HYG",
        "GLD", "SLV", "USO", "UNG", "ARKK", "ARKG", "SMH", "XBI", "XLF", "XLE",
        "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "KRE", "IYR", "VNQ",
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "BRK-B", "BRK.B",
        "JPM", "V", "MA", "UNH", "XOM", "JNJ", "WMT", "PG", "HD", "COST", "ORCL", "CRM",
        "AMD", "NFLX", "ADBE", "PEP", "KO", "MRK", "ABBV", "CVX", "BAC", "WFC", "GS",
        "IBM", "CAT", "BA", "GE", "DIS", "MCD", "NKE", "INTC", "QCOM", "TXN", "AMAT",
        "MU", "ARM", "TSM", "NOW", "INTU", "PANW", "UBER", "PLTR", "COIN", "HOOD",
        "SHOP", "SNOW", "CRWD", "NET", "AVGO", "LLY", "T", "VZ", "PFE",
    }
    out = []
    for t in raw_tickers:
        t = (t or "").strip().upper().replace(" ", "")
        if not t:
            continue
        # Yahoo futures / forex leave as-is
        if t.endswith("=F") or t.endswith("=X") or "^" in t:
            out.append(t)
            continue
        # Already pair-like
        if t.endswith("-USD") or t.endswith("-USDT") or t.endswith("-USDC"):
            out.append(t.replace("-USDT", "-USD").replace("-USDC", "-USD"))
            continue
        # BTCUSD / ETHUSDT glued forms
        for suf in ("USDT", "USD", "USDC"):
            if t.endswith(suf) and len(t) > len(suf):
                base = t[: -len(suf)]
                if base.isalpha() and 2 <= len(base) <= 10:
                    t = base + "-USD"
                    break
        if t.endswith("-USD"):
            out.append(t)
            continue
        # Known equity / ETF
        if t in equity or t.replace(".", "-") in equity:
            out.append(t.replace(".", "-"))
            continue
        # Plain symbol → treat as crypto pair for this app
        if t.replace("-", "").isalnum() and "-" not in t and t.isalpha() and 2 <= len(t) <= 10:
            out.append(t + "-USD")
            continue
        out.append(t)
    # dedupe preserve order
    seen = set()
    final = []
    for t in out:
        if t not in seen:
            seen.add(t)
            final.append(t)
    return final


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
from datetime import datetime


def load_json_file(path: str, default=None):
    if default is None:
        default = []
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                return data if data is not None else default
    except Exception:
        pass
    return default

def save_json_file(path: str, data):
    """Local disk fallback (ephemeral on many hosts)."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


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
# 2f. PHASE 4 HELPERS – Backtesting, Multi-TF, Export, Alert History
# ============================================================
import io as _io
from datetime import datetime

def run_rsi_backtest(df: pd.DataFrame, low=30, high=70):
    """Simple RSI mean-reversion backtest. Returns stats + equity curve."""
    if df is None or len(df) < 30 or "rsi" not in df.columns or "close" not in df.columns:
        return None
    data = df[["close", "rsi"]].dropna().copy()
    data["signal"] = 0
    data.loc[data["rsi"] < low, "signal"] = 1   # buy
    data.loc[data["rsi"] > high, "signal"] = -1  # sell
    data["position"] = data["signal"].replace(0, pd.NA).ffill().fillna(0)
    data["returns"] = data["close"].pct_change()
    data["strategy"] = data["position"].shift(1) * data["returns"]
    data["equity"] = (1 + data["strategy"].fillna(0)).cumprod()
    
    total_return = (data["equity"].iloc[-1] - 1) * 100
    buy_hold = (data["close"].iloc[-1] / data["close"].iloc[0] - 1) * 100
    max_dd = ((data["equity"] / data["equity"].cummax()) - 1).min() * 100
    trades = data["position"].diff().abs().sum() / 2
    win_rate = (data.loc[data["strategy"] > 0, "strategy"].count() / max(1, data.loc[data["strategy"] != 0, "strategy"].count())) * 100
    
    return {
        "equity": data["equity"],
        "total_return": total_return,
        "buy_hold": buy_hold,
        "max_drawdown": max_dd,
        "approx_trades": int(trades),
        "win_rate": win_rate,
    }


def run_sma_cross_backtest(df: pd.DataFrame):
    """Simple 50/200 SMA crossover backtest."""
    if df is None or len(df) < 210 or "sma50" not in df.columns or "sma200" not in df.columns:
        return None
    data = df[["close", "sma50", "sma200"]].dropna().copy()
    data["signal"] = 0
    data.loc[data["sma50"] > data["sma200"], "signal"] = 1
    data.loc[data["sma50"] < data["sma200"], "signal"] = -1
    data["position"] = data["signal"].replace(0, pd.NA).ffill().fillna(0)
    data["returns"] = data["close"].pct_change()
    data["strategy"] = data["position"].shift(1) * data["returns"]
    data["equity"] = (1 + data["strategy"].fillna(0)).cumprod()
    
    total_return = (data["equity"].iloc[-1] - 1) * 100
    buy_hold = (data["close"].iloc[-1] / data["close"].iloc[0] - 1) * 100
    max_dd = ((data["equity"] / data["equity"].cummax()) - 1).min() * 100
    trades = data["position"].diff().abs().sum() / 2
    
    return {
        "equity": data["equity"],
        "total_return": total_return,
        "buy_hold": buy_hold,
        "max_drawdown": max_dd,
        "approx_trades": int(trades),
    }


def get_multi_tf_summary(ticker: str) -> Dict[str, Dict]:
    """Fetch key metrics across several timeframes for one ticker."""
    tfs = {
        "15m": {"interval": "15m", "period": "5d"},
        "1h":  {"interval": "1h",  "period": "30d"},
        "1d":  {"interval": "1d",  "period": "1y"},
        "1wk": {"interval": "1wk", "period": "5y"},
    }
    results = {}
    for label, cfg in tfs.items():
        try:
            hist = yf.Ticker(ticker).history(period=cfg["period"], interval=cfg["interval"], auto_adjust=True)
            hist = normalize_columns(hist, target_ticker=ticker)
            if hist.empty or "close" not in hist.columns:
                continue
            hist = add_indicators(hist)
            last = hist.iloc[-1]
            change = 0.0
            if len(hist) > 1:
                change = ((last["close"] / hist["close"].iloc[-2]) - 1) * 100
            results[label] = {
                "price": float(last["close"]),
                "change": change,
                "rsi": float(last["rsi"]) if "rsi" in hist.columns and not pd.isna(last["rsi"]) else None,
                "sma50": float(last["sma50"]) if "sma50" in hist.columns and not pd.isna(last["sma50"]) else None,
            }
        except Exception:
            continue
    return results


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return CSV bytes for Streamlit download button."""
    buf = _io.StringIO()
    df.to_csv(buf, index=True)
    return buf.getvalue().encode()


# ============================================================
# FRED MACRO DATA (optional free API key from https://fred.stlouisfed.org/docs/api/api_key.html)
# ============================================================
FRED_SERIES = {
    "CPI (All Urban Consumers)": "CPIAUCSL",
    "Fed Funds Effective Rate": "FEDFUNDS",
    "M2 Money Supply": "M2SL",
    "Unemployment Rate": "UNRATE",
    "10Y Real Yield (TIPS)": "DFII10",
    "Industrial Production": "INDPRO",
}

@st.cache_data(ttl=86400, show_spinner="Fetching FRED macro series...")
def load_fred_series(api_key: str, series_id: str, limit: int = 120) -> Optional[pd.DataFrame]:
    """Pull a single FRED series. Returns DataFrame with date index and 'value' column."""
    if not api_key or not api_key.strip():
        return None
    try:
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}&api_key={api_key.strip()}&file_type=json"
            f"&sort_order=desc&limit={limit}"
        )
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        obs = data.get("observations") or []
        if not obs:
            return None
        rows = []
        for o in obs:
            try:
                val = float(o["value"])
                rows.append({"date": o["date"], "value": val})
            except (ValueError, TypeError, KeyError):
                continue
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        return df
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner="Loading FRED dashboard...")
def load_fred_dashboard(api_key: str) -> Dict[str, pd.DataFrame]:
    """Load all configured FRED series. Empty dict if no key or all fail."""
    if not api_key or not str(api_key).strip():
        return {}
    out = {}
    for label, sid in FRED_SERIES.items():
        df = load_fred_series(api_key, sid)
        if df is not None and not df.empty:
            out[label] = df
    return out


# ============================================================
# SHARED / LAZY DATA LAYER (one fetch per ticker+resolution combo)
# ============================================================

@st.cache_data(ttl=45, show_spinner=False)
def load_live_board(symbols: Tuple[str, ...] = None) -> pd.DataFrame:
    """Quick last price + session change for a fixed board of stocks & crypto."""
    if symbols is None:
        symbols = (
            "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA",
            "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD",
        )
    rows = []
    def _one(sym: str):
        try:
            h = yf.Ticker(sym).history(period="5d", interval="1d", auto_adjust=True)
            if h is None or h.empty or "Close" not in h.columns:
                return None
            px = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2]) if len(h) > 1 else px
            ch = ((px / prev) - 1) * 100 if prev else 0.0
            return {"Symbol": sym, "Price": px, "Chg%": ch}
        except Exception:
            return None
    workers = min(8, len(symbols))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed({ex.submit(_one, s): s for s in symbols}):
            try:
                r = fut.result()
                if r:
                    rows.append(r)
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Keep a stable display order
    order = {s: i for i, s in enumerate(symbols)}
    df["_ord"] = df["Symbol"].map(lambda x: order.get(x, 999))
    df = df.sort_values("_ord").drop(columns=["_ord"])
    return df




def update_and_diff_last_visit(prices: Dict[str, float], min_age_sec: int = 120) -> List[str]:
    """Compare current prices to last snapshot; only refresh baseline after min_age (survives auto-refresh)."""
    if not prices:
        return []
    prev = st.session_state.get("_last_visit_prices") or {}
    prev_ts = st.session_state.get("_last_visit_ts")
    lines = []
    now = time.time()
    age = (now - float(prev_ts)) if prev_ts else None
    if prev and isinstance(prev, dict) and age is not None and age >= min_age_sec:
        if age < 3600:
            ago = f" · {int(age // 60)}m ago"
        else:
            ago = f" · {int(age // 3600)}h ago"
        for sym, px in prices.items():
            try:
                px = float(px)
                old = prev.get(sym)
                if old is None:
                    continue
                old = float(old)
                if old <= 0:
                    continue
                ch = (px / old - 1) * 100
                if abs(ch) < 0.05:
                    lines.append(f"{sym.replace('-USD', '')} flat")
                else:
                    lines.append(f"{sym.replace('-USD', '')} {ch:+.2f}%")
            except Exception:
                continue
        st.session_state["_last_visit_ago"] = ago
    # Establish or roll baseline: first time, or after 15 minutes
    if not prev or age is None or age >= 900:
        st.session_state["_last_visit_prices"] = {
            k: float(v) for k, v in prices.items() if v is not None
        }
        st.session_state["_last_visit_ts"] = now
    return lines[:8]


def get_us_session_label(now_utc: Optional[pd.Timestamp] = None) -> Dict[str, str]:
    """Rough US equity session label in America/New_York (no exchange holiday calendar)."""
    try:
        now = now_utc or pd.Timestamp.utcnow()
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        et = now.tz_convert("America/New_York")
    except Exception:
        et = pd.Timestamp.now(tz="America/New_York")
    wd = et.dayofweek  # Mon=0
    mins = et.hour * 60 + et.minute
    pre_s, rth_s, rth_e, ah_e = 4 * 60, 9 * 60 + 30, 16 * 60, 20 * 60
    if wd >= 5:
        session = "Weekend"
        focus = "Plan levels · light liquidity — prefer higher timeframes"
    elif mins < pre_s:
        session = "Overnight"
        focus = "Futures & crypto leading · watch overnight gaps into the open"
    elif mins < rth_s:
        session = "Pre-market"
        focus = "Gaps & news · size smaller until cash open (9:30 ET)"
    elif mins < rth_e:
        session = "RTH (cash session)"
        focus = "Best liquidity window · respect open range and VWAP context"
    elif mins < ah_e:
        session = "After-hours"
        focus = "Thin books · earnings reactions · avoid chasing prints"
    else:
        session = "Evening"
        focus = "Debrief · mark levels for tomorrow’s pre-market"
    return {
        "session": session,
        "focus": focus,
        "et_clock": et.strftime("%Y-%m-%d %H:%M ET"),
    }


def compute_risk_score(macro_snap: Dict, fng_v, btc_chg: Optional[float] = None, spy_chg: Optional[float] = None) -> Dict[str, Any]:
    """Simple 0–100 risk appetite score from free macro prints."""
    score = 50
    details = []
    try:
        fv = float(fng_v)
        if fv <= 25:
            score -= 15
            details.append("Fear&Greed low")
        elif fv >= 75:
            score += 12
            details.append("Fear&Greed high")
    except Exception:
        pass
    try:
        if "VIX" in macro_snap and not macro_snap["VIX"].empty:
            vix = float(macro_snap["VIX"]["close"].iloc[-1])
            if vix >= 25:
                score -= 18
                details.append(f"VIX {vix:.1f}")
            elif vix <= 15:
                score += 10
                details.append(f"VIX {vix:.1f}")
    except Exception:
        pass
    try:
        dxy_k = "DXY (U.S. Dollar Index)"
        if dxy_k in macro_snap and not macro_snap[dxy_k].empty:
            dxy = macro_snap[dxy_k]
            ch = (float(dxy["close"].iloc[-1]) / float(dxy["close"].iloc[-2]) - 1) * 100 if len(dxy) > 1 else 0
            if ch > 0.3:
                score -= 8
                details.append("USD up")
            elif ch < -0.3:
                score += 6
                details.append("USD soft")
    except Exception:
        pass
    try:
        y2, y10 = macro_snap.get("U.S. 2Y Yield"), macro_snap.get("U.S. 10Y Treasury Yield")
        if y2 is not None and y10 is not None and not y2.empty and not y10.empty:
            common = y2["close"].index.intersection(y10["close"].index)
            sp = float(y10["close"].loc[common].iloc[-1] - y2["close"].loc[common].iloc[-1])
            if sp < 0:
                score -= 10
                details.append("2s10s inverted")
            elif sp > 0.5:
                score += 5
                details.append("curve positive")
    except Exception:
        pass
    if btc_chg is not None:
        if btc_chg > 2:
            score += 8
            details.append(f"BTC {btc_chg:+.1f}%")
        elif btc_chg < -2:
            score -= 8
            details.append(f"BTC {btc_chg:+.1f}%")
    if spy_chg is not None:
        if spy_chg > 0.5:
            score += 6
            details.append(f"SPY {spy_chg:+.1f}%")
        elif spy_chg < -0.5:
            score -= 6
            details.append(f"SPY {spy_chg:+.1f}%")
    score = int(max(0, min(100, score)))
    if score >= 65:
        label, color = "Risk-on", "🟢"
    elif score <= 35:
        label, color = "Risk-off", "🔴"
    else:
        label, color = "Mixed / neutral", "🟡"
    return {"score": score, "label": label, "color": color, "details": details}


@st.cache_data(ttl=180, show_spinner=False)
def load_gap_and_levels(symbols: Tuple[str, ...]) -> pd.DataFrame:
    """Daily gaps + basic levels for a small symbol list (Yahoo)."""
    rows = []
    syms = list(symbols)[:12]

    def _one(sym: str):
        try:
            h = yf.Ticker(sym).history(period="1y", interval="1d", auto_adjust=True)
            if h is None or len(h) < 5 or "Close" not in h.columns:
                return None
            h = h.dropna(subset=["Close"])
            last = h.iloc[-1]
            prev = h.iloc[-2]
            px = float(last["Close"])
            prev_c = float(prev["Close"])
            prev_h = float(prev["High"]) if "High" in h.columns else prev_c
            prev_l = float(prev["Low"]) if "Low" in h.columns else prev_c
            gap = (px / prev_c - 1) * 100 if prev_c else 0.0
            # classic floor pivot from prior day
            pivot = (prev_h + prev_l + prev_c) / 3.0
            r1 = 2 * pivot - prev_l
            s1 = 2 * pivot - prev_h
            # SMA50
            sma50 = float(h["Close"].tail(50).mean()) if len(h) >= 50 else float(h["Close"].mean())
            vs_sma = (px / sma50 - 1) * 100 if sma50 else 0.0
            # relative volume
            rel_vol = np.nan
            if "Volume" in h.columns and len(h) > 5:
                avg_v = float(h["Volume"].tail(20).mean())
                last_v = float(last["Volume"]) if pd.notna(last["Volume"]) else np.nan
                if avg_v and avg_v > 0 and pd.notna(last_v):
                    rel_vol = last_v / avg_v
            y_hi = float(h["High"].max()) if "High" in h.columns else px
            y_lo = float(h["Low"].min()) if "Low" in h.columns else px
            return {
                "Symbol": sym,
                "Price": px,
                "Gap %": gap,
                "Prev C": prev_c,
                "Prev H": prev_h,
                "Prev L": prev_l,
                "Pivot": pivot,
                "R1": r1,
                "S1": s1,
                "vs SMA50 %": vs_sma,
                "Rel Vol": rel_vol,
                "52W High": y_hi,
                "52W Low": y_lo,
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(_one, s): s for s in syms}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r:
                    rows.append(r)
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def get_economic_calendar(days_ahead: int = 45) -> pd.DataFrame:
    """
    Static high-impact US macro calendar (free, no API).
    Dates are known scheduled releases; confirm near the day on official sites.
    """
    # (date, event, typical_impact, notes)
    # Verified against Fed Board FOMC calendar + BLS CPI/NFP schedules (as of Oct 2026).
    # FOMC date = statement day (2nd day of meeting). Always confirm on federalreserve.gov / bls.gov.
    events = [
        ("2026-10-28", "FOMC Rate Decision", "High", "Meeting Oct 27–28 · statement ~2pm ET"),
        ("2026-11-06", "US Employment (NFP)", "High", "Jobs + unemployment · 8:30am ET"),
        ("2026-11-10", "US CPI (Inflation)", "High", "October CPI · 8:30am ET"),
        ("2026-12-04", "US Employment (NFP)", "High", "Jobs + unemployment · 8:30am ET"),
        ("2026-12-09", "FOMC Rate Decision", "High", "Meeting Dec 8–9 · SEP · statement ~2pm ET"),
        ("2026-12-10", "US CPI (Inflation)", "High", "November CPI · 8:30am ET"),
        ("2027-01-13", "US CPI (Inflation)", "High", "December CPI · 8:30am ET"),
        ("2027-01-27", "FOMC Rate Decision", "High", "Meeting Jan 26–27 · statement ~2pm ET"),
        ("2027-02-11", "US CPI (Inflation)", "High", "January CPI · confirm on BLS"),
        ("2027-03-10", "US CPI (Inflation)", "High", "February CPI · confirm on BLS"),
        ("2027-03-17", "FOMC Rate Decision", "High", "Meeting Mar 16–17 · SEP · statement ~2pm ET"),
        ("2027-04-28", "FOMC Rate Decision", "High", "Meeting Apr 27–28 · statement ~2pm ET"),
        ("2027-06-09", "FOMC Rate Decision", "High", "Meeting Jun 8–9 · SEP · statement ~2pm ET"),
        ("2027-07-28", "FOMC Rate Decision", "High", "Meeting Jul 27–28 · statement ~2pm ET"),
        ("2027-09-15", "FOMC Rate Decision", "High", "Meeting Sep 14–15 · SEP · statement ~2pm ET"),
        ("2027-10-27", "FOMC Rate Decision", "High", "Meeting Oct 26–27 · statement ~2pm ET"),
        ("2027-12-08", "FOMC Rate Decision", "High", "Meeting Dec 7–8 · SEP · statement ~2pm ET"),
    ]
    today = pd.Timestamp.now(tz="UTC").normalize()
    rows = []
    for d, name, impact, notes in events:
        try:
            dt = pd.Timestamp(d).tz_localize("UTC").normalize()
        except Exception:
            dt = pd.Timestamp(d, tz="UTC").normalize()
        if dt < today - pd.Timedelta(days=1):
            continue
        if dt > today + pd.Timedelta(days=int(days_ahead)):
            continue
        rows.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "Event": name,
            "Impact": impact,
            "Notes": notes,
            "Days": int((dt - today).days),
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=120, show_spinner=False)
def load_sector_board() -> pd.DataFrame:
    """Sector ETF snapshot — 1D and ~1W change via Yahoo."""
    sectors = {
        "XLK": "Tech",
        "XLF": "Financials",
        "XLE": "Energy",
        "XLV": "Healthcare",
        "XLI": "Industrials",
        "XLY": "Consumer Disc.",
        "XLP": "Consumer Staples",
        "XLU": "Utilities",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "SMH": "Semiconductors",
        "XBI": "Biotech",
    }

    def _one(sym: str, name: str):
        try:
            h = yf.Ticker(sym).history(period="1mo", interval="1d", auto_adjust=True)
            if h is None or h.empty or "Close" not in h.columns:
                return None
            px = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2]) if len(h) > 1 else px
            d1 = ((px / prev) - 1) * 100 if prev else 0.0
            week_ago = float(h["Close"].iloc[-6]) if len(h) > 6 else prev
            w1 = ((px / week_ago) - 1) * 100 if week_ago else 0.0
            return {"Sector": name, "ETF": sym, "Price": px, "1D %": d1, "1W %": w1}
        except Exception:
            return None

    rows = []
    syms = list(sectors.items())
    with ThreadPoolExecutor(max_workers=min(8, len(syms))) as ex:
        futs = {ex.submit(_one, s, n): s for s, n in syms}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r:
                    rows.append(r)
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("1D %", ascending=False)
    return df




# ============================================================
# STOCK SCREENER — top liquid stocks + crypto ETFs (timeout-safe)
# ============================================================
# Keep this list modest so Yahoo + Streamlit Cloud stay responsive.
TOP_STOCKS = [
    # ETFs / beta
    "SPY", "QQQ", "IWM", "DIA", "SMH", "XLF", "XLE", "XLK", "XLV", "ARKK",
    # Mag7 + mega
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "AVGO", "LLY",
    # Large liquid names
    "JPM", "V", "MA", "UNH", "XOM", "JNJ", "WMT", "PG", "HD", "COST",
    "ORCL", "CRM", "AMD", "NFLX", "ADBE", "PEP", "KO", "MRK", "ABBV", "CVX",
    "BAC", "WFC", "GS", "IBM", "CAT", "BA", "GE", "DIS", "MCD", "NKE",
    "INTC", "QCOM", "TXN", "AMAT", "MU", "ARM", "TSM", "NOW", "INTU", "PANW",
    "UBER", "PLTR", "COIN", "HOOD", "SHOP", "SNOW", "CRWD", "NET",
]

CRYPTO_ETFS = [
    "IBIT", "FBTC", "ARKB", "BITB", "BITO", "HODL", "BTCO", "EZBC",
    "ETHA", "ETHE", "ETHW", "EZET",
    "BITQ", "BKCH", "DAPP", "BLOK", "WGMI",
]

def build_screener_universe(mode: str) -> List[str]:
    """Small, liquid universes only — avoids Yahoo timeouts."""
    mode = (mode or "").lower()
    stocks = list(TOP_STOCKS)
    crypto = list(CRYPTO_ETFS)
    if "crypto" in mode and "stock" not in mode and "combined" not in mode:
        base = crypto
    elif "stock" in mode and "crypto" not in mode:
        base = stocks
    else:
        base = stocks + crypto
    seen = set()
    out = []
    for s in base:
        s = str(s).strip().upper()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


@st.cache_data(ttl=180, show_spinner="Screening top names…")
def run_stock_screener(symbols: Tuple[str, ...]) -> pd.DataFrame:
    """Fetch daily metrics for a compact liquid list (parallel, capped)."""
    def _one(sym: str):
        try:
            h = yf.Ticker(sym).history(period="6mo", interval="1d", auto_adjust=True)
            if h is None or len(h) < 30 or "Close" not in h.columns:
                return None
            close = h["Close"]
            vol = h["Volume"] if "Volume" in h.columns else None
            px = float(close.iloc[-1])
            prev = float(close.iloc[-2])
            d1 = (px / prev - 1) * 100
            d5 = (px / float(close.iloc[-6]) - 1) * 100 if len(close) > 6 else np.nan
            d21 = (px / float(close.iloc[-22]) - 1) * 100 if len(close) > 22 else np.nan
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_g = gain.ewm(alpha=1 / 14, min_periods=14).mean()
            avg_l = loss.ewm(alpha=1 / 14, min_periods=14).mean()
            rs = avg_g / avg_l.replace(0, np.nan)
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])
            sma50 = float(close.tail(50).mean()) if len(close) >= 50 else float(close.mean())
            sma200 = float(close.tail(200).mean()) if len(close) >= 200 else np.nan
            avg_vol = float(vol.tail(20).mean()) if vol is not None and len(vol) >= 5 else np.nan
            last_vol = float(vol.iloc[-1]) if vol is not None else np.nan
            rel_vol = (last_vol / avg_vol) if avg_vol and avg_vol > 0 else np.nan
            kind = "Crypto ETF" if sym in set(CRYPTO_ETFS) else "Stock/ETF"
            return {
                "Symbol": sym,
                "Type": kind,
                "Price": px,
                "1D %": d1,
                "5D %": d5,
                "1M %": d21,
                "RSI": rsi,
                "vs SMA50 %": (px / sma50 - 1) * 100 if sma50 else np.nan,
                "vs SMA200 %": (px / sma200 - 1) * 100 if sma200 and not np.isnan(sma200) else np.nan,
                "Rel Vol": rel_vol,
                "Above SMA50": px > sma50,
                "Above SMA200": px > sma200 if sma200 and not np.isnan(sma200) else False,
            }
        except Exception:
            return None

    rows = []
    sym_list = list(symbols)[:120]  # hard cap — timeout safety
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_one, s): s for s in sym_list}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                if r:
                    rows.append(r)
            except Exception:
                pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def get_shared_market_data(tickers, interval, lookback, period, is_6h):
    """Fetch once per unique config; reuse across tabs in the same run via session_state."""
    key = ("mkt", tuple(tickers), interval, lookback, period, is_6h)
    cache = st.session_state.setdefault("_data_cache", {})
    # Soft TTL using timestamp
    now = time.time()
    entry = cache.get(key)
    if entry and (now - entry.get("ts", 0)) < 60:
        return entry["data"]
    data = load_ticker_data(tuple(tickers), interval, lookback, period, is_6h) if tickers else {}
    cache[key] = {"ts": now, "data": data}
    # Prevent unbounded growth
    if len(cache) > 12:
        oldest = sorted(cache.items(), key=lambda x: x[1].get("ts", 0))[: len(cache) - 12]
        for k, _ in oldest:
            cache.pop(k, None)
    return data


def build_daily_briefing(tickers, data, fear_greed, macro_data, defi, alerts_triggered):
    """Compact text/metrics package for the Briefing tab."""
    lines = []
    fng_v, fng_c = fear_greed if fear_greed else ("N/A", "N/A")
    lines.append(f"Fear & Greed: {fng_v}/100 ({fng_c})")
    if macro_data:
        for name in ["VIX", "U.S. 10Y Treasury Yield", "DXY (U.S. Dollar Index)"]:
            if name in macro_data and not macro_data[name].empty:
                p = float(macro_data[name]["close"].iloc[-1])
                lines.append(f"{name}: {p:.2f}")
    if defi and defi.get("total_tvl"):
        lines.append(f"Total Crypto TVL: ${defi['total_tvl']:,.0f}")
    movers = []
    for t, df in (data or {}).items():
        if df is not None and len(df) > 1 and "close" in df.columns:
            ch = (float(df["close"].iloc[-1]) / float(df["close"].iloc[-2]) - 1) * 100
            movers.append((t, ch, float(df["close"].iloc[-1])))
    movers.sort(key=lambda x: abs(x[1]), reverse=True)
    return {"lines": lines, "movers": movers[:8], "alerts": alerts_triggered or []}

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
# 3b. CHART PATTERN RECOGNITION ENGINE (upgraded)
# ============================================================
def _trend_slope(closes: np.ndarray, look: int = 8) -> float:
    """Simple % slope over last `look` bars. Positive = uptrend context."""
    if len(closes) < look + 1:
        return 0.0
    a, b = float(closes[-look - 1]), float(closes[-1])
    if a == 0:
        return 0.0
    return (b - a) / a


def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict]:
    """
    Detect candlestick patterns with trend context and stricter geometry.
    Confidence: High / Medium / Low based on quality + context fit.
    """
    patterns: List[Dict] = []
    need = ["open", "high", "low", "close"]
    if df is None or len(df) < 8 or not all(c in df.columns for c in need):
        return patterns

    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    idx = df.index
    n = len(df)

    body = np.abs(c - o)
    full = h - l
    full_safe = np.where(full < 1e-12, np.nan, full)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    avg_body = pd.Series(body).rolling(14, min_periods=3).mean().values

    def add(i, typ, direction, conf, desc, price=None):
        patterns.append({
            "type": typ,
            "direction": direction,
            "index": idx[i],
            "price": float(price if price is not None else c[i]),
            "confidence": conf,
            "description": desc,
        })

    # Only scan a recent window for responsiveness / less noise
    start_i = max(3, n - 80)

    for i in range(start_i, n):
        if np.isnan(full_safe[i]) or full[i] <= 0:
            continue
        bi, fi = body[i], full[i]
        ui, li = upper[i], lower[i]
        bull = c[i] > o[i]
        bear = c[i] < o[i]
        ab = avg_body[i] if not np.isnan(avg_body[i]) and avg_body[i] > 0 else bi
        slope = _trend_slope(c[: i + 1], 8)

        # --- Single-candle ---
        # Doji (tight body)
        if bi / fi < 0.08:
            conf = "High" if bi / fi < 0.04 else "Medium"
            add(i, "Doji", "Neutral", conf, "Indecision — possible pause/reversal")

        # Spinning top
        elif bi / fi < 0.25 and ui > 0.2 * fi and li > 0.2 * fi:
            add(i, "Spinning Top", "Neutral", "Low", "Indecision with upper & lower wicks")

        # Marubozu (little/no wick)
        if bi / fi > 0.88:
            if bull:
                add(i, "Bullish Marubozu", "Bullish", "Medium", "Strong buying, full-range close")
            elif bear:
                add(i, "Bearish Marubozu", "Bearish", "Medium", "Strong selling, full-range close")

        # Hammer — prefer after decline
        if li >= 2.0 * max(bi, 1e-12) and ui <= 0.35 * max(bi, 1e-12) and bi / fi < 0.4:
            conf = "High" if slope < -0.01 else ("Medium" if slope <= 0 else "Low")
            if conf != "Low" or slope < 0:
                add(i, "Hammer", "Bullish", conf if slope < 0 else "Low",
                    "Long lower wick — bullish reversal if after a decline")

        # Inverted Hammer
        if ui >= 2.0 * max(bi, 1e-12) and li <= 0.35 * max(bi, 1e-12) and bi / fi < 0.4 and slope < -0.005:
            add(i, "Inverted Hammer", "Bullish", "Medium", "Long upper wick after decline — possible bottom")

        # Hanging Man (same shape as hammer, after rise)
        if li >= 2.0 * max(bi, 1e-12) and ui <= 0.35 * max(bi, 1e-12) and bi / fi < 0.4 and slope > 0.01:
            add(i, "Hanging Man", "Bearish", "Medium", "Long lower wick after advance — possible top")

        # Shooting Star — prefer after advance
        if ui >= 2.0 * max(bi, 1e-12) and li <= 0.35 * max(bi, 1e-12) and bi / fi < 0.4:
            conf = "High" if slope > 0.01 else ("Medium" if slope >= 0 else "Low")
            if conf != "Low" or slope > 0:
                add(i, "Shooting Star", "Bearish", conf if slope > 0 else "Low",
                    "Long upper wick — bearish reversal if after a rally")

        # --- Two-candle ---
        if i < 1:
            continue
        prev_bull = c[i - 1] > o[i - 1]
        prev_bear = c[i - 1] < o[i - 1]
        prev_bi = body[i - 1]

        # Bullish Engulfing
        if prev_bear and bull and c[i] >= o[i - 1] and o[i] <= c[i - 1] and bi > prev_bi * 1.05:
            conf = "High" if slope < 0 else "Medium"
            add(i, "Bullish Engulfing", "Bullish", conf, "Bullish candle engulfs prior bearish body")

        # Bearish Engulfing
        if prev_bull and bear and c[i] <= o[i - 1] and o[i] >= c[i - 1] and bi > prev_bi * 1.05:
            conf = "High" if slope > 0 else "Medium"
            add(i, "Bearish Engulfing", "Bearish", conf, "Bearish candle engulfs prior bullish body")

        # Piercing Line
        if prev_bear and bull and o[i] < l[i - 1] and c[i] > (o[i - 1] + c[i - 1]) / 2 and c[i] < o[i - 1]:
            add(i, "Piercing Line", "Bullish", "Medium", "Opens below prior low, closes above midpoint")

        # Dark Cloud Cover
        if prev_bull and bear and o[i] > h[i - 1] and c[i] < (o[i - 1] + c[i - 1]) / 2 and c[i] > o[i - 1]:
            add(i, "Dark Cloud Cover", "Bearish", "Medium", "Opens above prior high, closes below midpoint")

        # Tweezer Bottom
        if i >= 1 and abs(l[i] - l[i - 1]) / max(abs(l[i]), 1e-12) < 0.0025 and prev_bear and bull:
            add(i, "Tweezer Bottom", "Bullish", "Medium", "Matching lows — possible support")

        # Tweezer Top
        if i >= 1 and abs(h[i] - h[i - 1]) / max(abs(h[i]), 1e-12) < 0.0025 and prev_bull and bear:
            add(i, "Tweezer Top", "Bearish", "Medium", "Matching highs — possible resistance")

        # --- Three-candle ---
        if i < 2:
            continue
        # Morning Star
        mid_small = body[i - 1] < ab * 0.6
        if (c[i - 2] < o[i - 2] and mid_small and bull
                and c[i] > (o[i - 2] + c[i - 2]) / 2 and body[i - 2] > ab * 0.5):
            add(i, "Morning Star", "Bullish", "High", "Three-bar bullish reversal sequence")

        # Evening Star
        if (c[i - 2] > o[i - 2] and mid_small and bear
                and c[i] < (o[i - 2] + c[i - 2]) / 2 and body[i - 2] > ab * 0.5):
            add(i, "Evening Star", "Bearish", "High", "Three-bar bearish reversal sequence")

        # Three White Soldiers
        if (c[i] > o[i] and c[i - 1] > o[i - 1] and c[i - 2] > o[i - 2]
                and c[i] > c[i - 1] > c[i - 2]
                and o[i] > o[i - 1] > o[i - 2]
                and body[i] > ab * 0.5 and body[i - 1] > ab * 0.5):
            add(i, "Three White Soldiers", "Bullish", "High", "Three strong consecutive advances")

        # Three Black Crows
        if (c[i] < o[i] and c[i - 1] < o[i - 1] and c[i - 2] < o[i - 2]
                and c[i] < c[i - 1] < c[i - 2]
                and o[i] < o[i - 1] < o[i - 2]
                and body[i] > ab * 0.5 and body[i - 1] > ab * 0.5):
            add(i, "Three Black Crows", "Bearish", "High", "Three strong consecutive declines")

    # Keep last ~40 bars of signals, prefer higher confidence, cap count
    if n > 40:
        cutoff = idx[-40]
        patterns = [p for p in patterns if p["index"] >= cutoff]
    rank = {"High": 0, "Medium": 1, "Low": 2}
    patterns.sort(key=lambda p: (rank.get(p["confidence"], 9),))
    # Dedupe same type on same bar
    seen = set()
    deduped = []
    for p in patterns:
        key = (p["type"], str(p["index"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped[:18]


def detect_chart_patterns(df: pd.DataFrame) -> List[Dict]:
    """Swing-based chart patterns: S/R, double top/bottom, simple H&S / inv H&S."""
    patterns: List[Dict] = []
    if df is None or len(df) < 30 or "high" not in df.columns:
        return patterns

    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    idx = df.index
    n = len(df)
    win = 4  # local swing window

    def is_swing_high(i):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        return highs[i] >= np.max(highs[lo:hi]) - 1e-12

    def is_swing_low(i):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        return lows[i] <= np.min(lows[lo:hi]) + 1e-12

    swing_h = [(i, highs[i]) for i in range(win, n - win) if is_swing_high(i)]
    swing_l = [(i, lows[i]) for i in range(win, n - win) if is_swing_low(i)]

    # Recent support / resistance (last few swings only)
    for i, px in swing_h[-3:]:
        patterns.append({
            "type": "Resistance",
            "direction": "Neutral",
            "index": idx[i],
            "price": float(px),
            "confidence": "Medium",
            "description": f"Swing resistance @ {px:.4g}",
        })
    for i, px in swing_l[-3:]:
        patterns.append({
            "type": "Support",
            "direction": "Neutral",
            "index": idx[i],
            "price": float(px),
            "confidence": "Medium",
            "description": f"Swing support @ {px:.4g}",
        })

    # Double Top: two highs within 1.5%, separated, with lower low between, prefer second near end
    if len(swing_h) >= 2:
        for j in range(len(swing_h) - 1, 0, -1):
            i2, p2 = swing_h[j]
            i1, p1 = swing_h[j - 1]
            if i2 - i1 < 6:
                continue
            if abs(p1 - p2) / max(p1, 1e-12) > 0.015:
                continue
            neck = float(np.min(lows[i1:i2 + 1]))
            if neck >= min(p1, p2) * 0.995:
                continue
            # Prefer when price has left the second peak
            conf = "High" if closes[-1] < (p2 + neck) / 2 else "Medium"
            if i2 >= n - 45:
                patterns.append({
                    "type": "Double Top",
                    "direction": "Bearish",
                    "index": idx[i2],
                    "price": float(p2),
                    "confidence": conf,
                    "description": f"Double top ~{p2:.4g}, neckline ~{neck:.4g}",
                })
                break

    # Double Bottom
    if len(swing_l) >= 2:
        for j in range(len(swing_l) - 1, 0, -1):
            i2, p2 = swing_l[j]
            i1, p1 = swing_l[j - 1]
            if i2 - i1 < 6:
                continue
            if abs(p1 - p2) / max(abs(p1), 1e-12) > 0.015:
                continue
            neck = float(np.max(highs[i1:i2 + 1]))
            if neck <= max(p1, p2) * 1.005:
                continue
            conf = "High" if closes[-1] > (p2 + neck) / 2 else "Medium"
            if i2 >= n - 45:
                patterns.append({
                    "type": "Double Bottom",
                    "direction": "Bullish",
                    "index": idx[i2],
                    "price": float(p2),
                    "confidence": conf,
                    "description": f"Double bottom ~{p2:.4g}, neckline ~{neck:.4g}",
                })
                break

    # Very simple Head & Shoulders: 3 swing highs, middle highest
    if len(swing_h) >= 3:
        a, b, c_ = swing_h[-3], swing_h[-2], swing_h[-1]
        if b[1] > a[1] and b[1] > c_[1] and abs(a[1] - c_[1]) / max(b[1], 1e-12) < 0.03:
            if c_[0] - a[0] >= 10 and c_[0] >= n - 50:
                patterns.append({
                    "type": "Head & Shoulders",
                    "direction": "Bearish",
                    "index": idx[c_[0]],
                    "price": float(c_[1]),
                    "confidence": "Medium",
                    "description": f"H&S structure, head ~{b[1]:.4g}",
                })

    # Inverse H&S: 3 swing lows, middle lowest
    if len(swing_l) >= 3:
        a, b, c_ = swing_l[-3], swing_l[-2], swing_l[-1]
        if b[1] < a[1] and b[1] < c_[1] and abs(a[1] - c_[1]) / max(abs(b[1]), 1e-12) < 0.03:
            if c_[0] - a[0] >= 10 and c_[0] >= n - 50:
                patterns.append({
                    "type": "Inverse Head & Shoulders",
                    "direction": "Bullish",
                    "index": idx[c_[0]],
                    "price": float(c_[1]),
                    "confidence": "Medium",
                    "description": f"Inv H&S structure, head ~{b[1]:.4g}",
                })

    return patterns


def detect_all_patterns(df: pd.DataFrame) -> List[Dict]:
    """Run all pattern detectors and return combined, confidence-sorted list."""
    candle = detect_candlestick_patterns(df)
    chart = detect_chart_patterns(df)
    combined = candle + chart
    rank = {"High": 0, "Medium": 1, "Low": 2}
    combined.sort(key=lambda p: (rank.get(p.get("confidence"), 9), str(p.get("index"))))
    return combined


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
        "Spinning Top": "diamond",
        "Hammer": "triangle-up",
        "Inverted Hammer": "triangle-up",
        "Hanging Man": "triangle-down",
        "Shooting Star": "triangle-down",
        "Bullish Engulfing": "star",
        "Bearish Engulfing": "star",
        "Piercing Line": "star",
        "Dark Cloud Cover": "star",
        "Morning Star": "star",
        "Evening Star": "star",
        "Three White Soldiers": "triangle-up",
        "Three Black Crows": "triangle-down",
        "Bullish Marubozu": "square",
        "Bearish Marubozu": "square",
        "Tweezer Bottom": "circle",
        "Tweezer Top": "circle",
        "Double Top": "x",
        "Double Bottom": "x",
        "Head & Shoulders": "x",
        "Inverse Head & Shoulders": "x",
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
def _fetch_one_ticker(t: str, interval: str, lookback: int, period: str, is_6h: bool) -> Tuple[str, Optional[pd.DataFrame]]:
    """Worker for concurrent ticker fetch."""
    try:
        ticker_obj = yf.Ticker(t)
        hist = ticker_obj.history(period=period, interval=interval, auto_adjust=True)
        if hist is None or hist.empty:
            hist = ticker_obj.history(period="1y", interval=interval, auto_adjust=True)
            if hist is None or hist.empty:
                return t, None
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
            # Trim empty trailing closes so charts don't pad blank space
            if "close" in display.columns:
                display = display.dropna(subset=["close"])
            return t, display
    except Exception:
        pass
    return t, None


@st.cache_data(ttl=45, show_spinner="Fetching market data (parallel)...")
def load_ticker_data(
    tickers: Tuple[str, ...],
    interval: str,
    lookback: int,
    period: str,
    is_6h: bool,
) -> Dict[str, pd.DataFrame]:
    """Parallel fetch for faster multi-ticker loads."""
    results = {}
    if not tickers:
        return results
    # Cap workers to avoid hammering Yahoo
    workers = min(6, max(1, len(tickers)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_fetch_one_ticker, t, interval, lookback, period, is_6h): t
            for t in tickers
        }
        for fut in as_completed(futures):
            try:
                t, df = fut.result()
                if df is not None and not df.empty:
                    results[t] = df
            except Exception:
                continue
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

@st.cache_data(ttl=600)
def get_fear_greed() -> Tuple[str, str]:
    try:
        res = requests.get("https://api.alternative.me/fng/", headers=HEADERS, timeout=8).json()
        data = res["data"]
        if isinstance(data, list) and len(data) > 0:
            return data[0]["value"], data[0]["value_classification"]
        return "N/A", "Unavailable"
    except Exception:
        return "N/A", "Unavailable"

@st.cache_data(ttl=300, show_spinner=False)
def get_equity_snapshot(symbol: str) -> Dict[str, Any]:
    """Free Yahoo snapshot for stocks/ETFs (best-effort fields)."""
    sym = (symbol or "").strip().upper()
    out = {"Symbol": sym, "Error": None}
    if not sym:
        out["Error"] = "No symbol"
        return out
    try:
        t = yf.Ticker(sym)
        info = {}
        try:
            info = t.fast_info or {}
        except Exception:
            info = {}
        # fast_info may be object-like
        def _get(obj, *keys):
            for k in keys:
                try:
                    if isinstance(obj, dict) and k in obj and obj[k] is not None:
                        return obj[k]
                    v = getattr(obj, k, None)
                    if v is not None:
                        return v
                except Exception:
                    pass
            return None
        price = _get(info, "last_price", "lastPrice", "regularMarketPrice")
        prev = _get(info, "previous_close", "previousClose")
        day_h = _get(info, "day_high", "dayHigh")
        day_l = _get(info, "day_low", "dayLow")
        yr_h = _get(info, "year_high", "fiftyTwoWeekHigh")
        yr_l = _get(info, "year_low", "fiftyTwoWeekLow")
        mcap = _get(info, "market_cap", "marketCap")
        vol = _get(info, "last_volume", "lastVolume", "volume")
        # richer info (slower / sometimes empty on free)
        try:
            inf = t.info or {}
        except Exception:
            inf = {}
        out.update({
            "Name": inf.get("shortName") or inf.get("longName") or sym,
            "Price": price,
            "Prev Close": prev,
            "Day Range": f"{day_l} – {day_h}" if day_l and day_h else None,
            "52W Range": f"{yr_l} – {yr_h}" if yr_l and yr_h else None,
            "Market Cap": mcap,
            "Volume": vol,
            "PE": inf.get("trailingPE") or inf.get("forwardPE"),
            "Forward PE": inf.get("forwardPE"),
            "EPS": inf.get("trailingEps"),
            "Sector": inf.get("sector"),
            "Industry": inf.get("industry"),
            "Avg Volume": inf.get("averageVolume") or inf.get("averageVolume10days"),
            "Dividend Yield": inf.get("dividendYield"),
            "As Of": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        })
        return out
    except Exception as e:
        out["Error"] = str(e)[:120]
        return out


def filter_news_for_tickers(news: List[Dict], tickers: List[str]) -> List[Dict]:
    """Return headlines that mention any loaded ticker (symbol or common name)."""
    if not news or not tickers:
        return []
    aliases = {
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", "ether", "eth"],
        "SOL": ["solana", "sol"],
        "XRP": ["xrp", "ripple"],
        "AAPL": ["apple", "aapl"],
        "MSFT": ["microsoft", "msft"],
        "NVDA": ["nvidia", "nvda"],
        "AMZN": ["amazon", "amzn"],
        "GOOGL": ["google", "alphabet", "googl"],
        "GOOG": ["google", "alphabet", "goog"],
        "META": ["meta", "facebook"],
        "TSLA": ["tesla", "tsla"],
        "SPY": ["s&p", "sp500", "s&p 500", "spy"],
        "QQQ": ["nasdaq", "qqq"],
    }
    keys = []
    for t in tickers:
        base = t.upper().replace("-USD", "").replace("USDT", "").strip()
        keys.append(base.lower())
        for a in aliases.get(base, []):
            keys.append(a.lower())
        keys.append(t.lower())
    keys = list(set(keys))
    hits = []
    for art in news:
        blob = f"{art.get('title','')} {art.get('summary','')}".lower()
        matched = [k for k in keys if k and k in blob]
        if matched:
            a = dict(art)
            a["matched"] = ", ".join(sorted(set(matched))[:4])
            hits.append(a)
    return hits


@st.cache_data(ttl=900, show_spinner="Fetching market news…")
def get_market_news(cache_buster: str = "", max_items: int = 40) -> List[Dict]:
    """
    Free multi-source news: Fed, SEC, Stocks/Macro, Crypto.
    cache_buster only exists to optionally bust Streamlit cache from the caller.
    """
    feeds = [
        # Fed / policy
        ("Fed", "Federal Reserve", "https://www.federalreserve.gov/feeds/press_all.xml"),
        ("Fed", "Fed Speeches", "https://www.federalreserve.gov/feeds/speeches.xml"),
        # SEC / regulation
        ("SEC", "SEC Press", "https://www.sec.gov/news/pressreleases.rss"),
        # Stocks / equities
        ("Stocks", "Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
        ("Stocks", "CNBC Top News", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
        ("Stocks", "CNBC Markets", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258"),
        ("Stocks", "MarketWatch", "https://feeds.marketwatch.com/marketwatch/topstories/"),
        ("Stocks", "MarketWatch Markets", "https://feeds.marketwatch.com/marketwatch/marketpulse/"),
        # Crypto
        ("Crypto", "CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Crypto", "CoinTelegraph", "https://cointelegraph.com/rss"),
        ("Crypto", "Decrypt", "https://decrypt.co/feed"),
        ("Crypto", "The Block", "https://www.theblock.co/rss.xml"),
    ]
    articles: List[Dict] = []
    for category, source, url in feeds:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                continue
            feed = feedparser.parse(io.BytesIO(resp.content))
            for entry in feed.entries[:5]:
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
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.tz_localize("UTC")
                summary = entry.get("summary", "") or entry.get("description", "") or ""
                if summary:
                    # strip simple HTML
                    for tag in ("<p>", "</p>", "<br>", "<br/>", "<br />"):
                        summary = summary.replace(tag, " ")
                    while "<" in summary and ">" in summary:
                        a, b = summary.find("<"), summary.find(">")
                        if a == -1 or b == -1 or b < a:
                            break
                        summary = summary[:a] + " " + summary[b + 1 :]
                    summary = " ".join(summary.split())[:240]
                    if len(summary) == 240:
                        summary += "..."
                articles.append({
                    "title": (entry.get("title") or "No title").strip(),
                    "link": entry.get("link") or "#",
                    "source": source,
                    "category": category,
                    "published": pub_dt.strftime("%Y-%m-%d %H:%M UTC"),
                    "published_dt": pub_dt,
                    "summary": summary,
                })
        except Exception:
            continue
    articles = sorted(articles, key=lambda x: x["published_dt"], reverse=True)
    # Light dedupe by title
    seen = set()
    out = []
    for a in articles:
        key = a["title"].lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
        if len(out) >= max_items:
            break
    return out


def get_crypto_news(cache_buster: str, max_items: int = 12) -> List[Dict]:
    """Backward-compatible wrapper — crypto slice of market news."""
    all_news = get_market_news(cache_buster=cache_buster, max_items=60)
    crypto = [a for a in all_news if a.get("category") == "Crypto"]
    return crypto[:max_items] if crypto else all_news[:max_items]

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

@st.cache_data(ttl=30)
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

@st.cache_data(ttl=60, show_spinner="Gathering cross-asset macro indicators...")
def load_macro_data() -> Dict[str, pd.DataFrame]:
    """Expanded free macro set via Yahoo Finance (no API key)."""
    macro_tickers = {
        "DXY (U.S. Dollar Index)": "DX-Y.NYB",
        "S&P 500 Index": "^GSPC",
        "NASDAQ 100 Index": "^NDX",
        "Dow Jones": "^DJI",
        "Russell 2000": "^RUT",
        "Gold Futures": "GC=F",
        "Silver Futures": "SI=F",
        "Crude Oil Futures": "CL=F",
        "U.S. 2Y Yield": "^IRX",
        "U.S. 10Y Treasury Yield": "^TNX",
        "U.S. 30Y Yield": "^TYX",
        "VIX": "^VIX",
        "Bitcoin": "BTC-USD",
    }
    macro_results = {}
    for label, sym in macro_tickers.items():
        try:
            m_hist = yf.Ticker(sym).history(period="6mo", interval="1d", auto_adjust=True)
            if m_hist is not None and not m_hist.empty:
                macro_results[label] = normalize_columns(m_hist, target_ticker=sym)
        except Exception:
            continue
    return macro_results


@st.cache_data(ttl=180, show_spinner="Fetching DeFi / stablecoin landscape...")
def load_defi_overview() -> Dict[str, Any]:
    """Free DefiLlama aggregates: total TVL + stablecoin mcap snapshot."""
    out = {"total_tvl": None, "top_chains": [], "stablecoin_mcap": None, "error": None}
    try:
        r = requests.get("https://api.llama.fi/v2/chains", headers=HEADERS, timeout=12)
        chains = r.json()
        if isinstance(chains, list) and chains:
            # sort by TVL
            ranked = sorted(
                [c for c in chains if c.get("tvl")],
                key=lambda x: x.get("tvl") or 0,
                reverse=True
            )
            out["total_tvl"] = sum((c.get("tvl") or 0) for c in ranked)
            out["top_chains"] = [
                {"name": c.get("name"), "tvl": c.get("tvl"), "symbol": c.get("tokenSymbol")}
                for c in ranked[:15]
            ]
    except Exception as e:
        out["error"] = str(e)[:120]
    try:
        r2 = requests.get("https://stablecoins.llama.fi/stablecoins?includePrices=true", headers=HEADERS, timeout=12)
        data = r2.json()
        # Approximate total from listed pegged assets
        total = 0.0
        for s in data.get("peggedAssets", [])[:50]:
            circ = (s.get("circulating") or {}).get("peggedUSD") or 0
            total += float(circ)
        out["stablecoin_mcap"] = total if total > 0 else None
    except Exception:
        pass
    return out

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
        sia = get_sia() or SIA()
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


def build_volume_by_price_chart(h_data: pd.DataFrame, bins: int = 28) -> Optional[go.Figure]:
    """Proper volume-at-price profile: price on Y, volume on X (standalone, not shared axis)."""
    if h_data is None or h_data.empty:
        return None
    if "close" not in h_data.columns or "volume" not in h_data.columns:
        return None
    df = h_data.dropna(subset=["close", "volume"]).copy()
    if len(df) < 5:
        return None
    try:
        price_bins = pd.cut(df["close"], bins=min(bins, max(8, len(df) // 3)))
        grouped = df.groupby(price_bins, observed=True)["volume"].sum()
        if grouped.empty:
            return None
        mids = [interval.mid for interval in grouped.index]
        vols = grouped.values.astype(float)
        # Color by whether bin is above/below last price
        last_px = float(df["close"].iloc[-1])
        colors = ["rgba(0,255,127,0.75)" if m >= last_px else "rgba(255,69,0,0.75)" for m in mids]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=vols,
            y=mids,
            orientation="h",
            marker_color=colors,
            name="Volume",
            hovertemplate="Price ~ %{y:.4g}<br>Volume %{x:,.0f}<extra></extra>",
        ))
        # Mark last price
        fig.add_hline(
            y=last_px, line_dash="dot", line_color="#00FFFF", line_width=1,
            annotation_text=f"Last {last_px:,.4g}", annotation_position="top right",
        )
        fig.update_layout(
            title="Volume by Price (profile)",
            height=320,
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            margin=dict(l=10, r=20, t=40, b=10),
            font=dict(family="IBM Plex Sans, sans-serif"),
            xaxis_title="Volume",
            yaxis_title="Price",
            showlegend=False,
            bargap=0.15,
        )
        return fig
    except Exception:
        return None


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
    revision_key: str = "asset-chart",
) -> go.Figure:
    h_data = h_data.copy()
    # --- Chart cutoff fixes ---
    if "close" in h_data.columns:
        h_data = h_data.dropna(subset=["close"])
    if len(h_data) > 0:
        h_data = h_data.loc[: h_data.last_valid_index()]
    # Drop trailing bars separated by a large time gap (orphan candle on the right)
    if len(h_data) >= 5:
        try:
            idx = pd.DatetimeIndex(h_data.index)
            deltas = idx.to_series().diff().dropna()
            med = deltas.median()
            if pd.notna(med) and med > pd.Timedelta(0):
                keep_until = len(h_data)
                for i in range(len(h_data) - 1, 0, -1):
                    gap = idx[i] - idx[i - 1]
                    if gap > med * 4:
                        keep_until = i
                    else:
                        break
                if keep_until < len(h_data):
                    h_data = h_data.iloc[:keep_until]
        except Exception:
            pass
    # SMA200 is meaningless / flat on short intraday windows — auto-disable unless enough bars
    effective_sma200 = show_sma200 and "sma200" in h_data.columns and h_data["sma200"].notna().sum() >= 30
    # Equal-width bars (TradingView-style) — eliminates visual time gaps on the right
    time_index = h_data.index
    x_pos = list(range(len(h_data)))
    if show_baseline and "close" in h_data.columns and len(h_data):
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
    if show_vol_profile: subplot_titles.append("Volume")
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
                x=x_pos,
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
                x=x_pos, y=h_data["close"],
                name="Price", line=dict(color="#00FFFF", width=2),
            ),
            row=current_row, col=1,
        )
    if show_bb and "bb_upper" in h_data.columns:
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["bb_upper"], name="BB Upper",
                                 line=dict(color="rgba(255,255,255,0.3)", width=1), showlegend=False), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["bb_lower"], name="BB Lower",
                                 line=dict(color="rgba(255,255,255,0.3)", width=1),
                                 fill="tonexty", fillcolor="rgba(0,255,127,0.08)"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["bb_mid"], name="BB Mid",
                                 line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dot")), row=current_row, col=1)
    if show_sma50 and "sma50" in h_data.columns:
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["sma50"], name="SMA 50",
                                 line=dict(color="#00FF7F", width=1.5, dash="dash")), row=current_row, col=1)
    if effective_sma200:
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["sma200"], name="SMA 200",
                                 line=dict(color="#FF4500", width=1.5)), row=current_row, col=1)
    if show_baseline and "baseline" in h_data.columns:
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["baseline"], name="Hist. Baseline (Avg)",
                                 line=dict(color="#FFD700", width=1.8, dash="dot")), row=current_row, col=1)
    if show_vwap and "vwap" in h_data.columns:
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["vwap"], name="VWAP",
                                 line=dict(color="#FF00FF", width=1.6)), row=current_row, col=1)
    
    # --- PATTERN OVERLAY ---
    if patterns:
        # Map pattern timestamps → equal-width integer positions
        mapped_patterns = []
        for _p in (patterns or []):
            _p = dict(_p)
            try:
                loc = time_index.get_loc(_p["index"])
                if isinstance(loc, slice):
                    loc = loc.start
                if isinstance(loc, (int, np.integer)):
                    _p["index"] = int(loc)
                    mapped_patterns.append(_p)
            except Exception:
                try:
                    loc = int(time_index.get_indexer([_p["index"]], method="nearest")[0])
                    if loc >= 0:
                        _p["index"] = loc
                        mapped_patterns.append(_p)
                except Exception:
                    pass
        fig = add_patterns_to_chart(fig, mapped_patterns, row=current_row)
    
    if show_rsi and "rsi" in h_data.columns:
        current_row += 1
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["rsi"], name="RSI",
                                 line=dict(color="#FFD700", width=1.5)), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="lime", row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
    if show_macd and "macd_hist" in h_data.columns:
        current_row += 1
        colors = ["#00FF7F" if v >= 0 else "#FF4500" for v in h_data["macd_hist"]]
        fig.add_trace(go.Bar(x=x_pos, y=h_data["macd_hist"], name="Histogram",
                             marker_color=colors, showlegend=False), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["macd"], name="MACD",
                                 line=dict(color="#00BFFF", width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["macd_signal"], name="Signal",
                                 line=dict(color="#FF69B4", width=1.5)), row=current_row, col=1)
    if show_stoch and "stoch_k" in h_data.columns:
        current_row += 1
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["stoch_k"], name="%K",
                                 line=dict(color="#00FFFF", width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["stoch_d"], name="%D",
                                 line=dict(color="#FFD700", width=1.5)), row=current_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="lime", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
    if show_atr and "atr" in h_data.columns:
        current_row += 1
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["atr"], name="ATR",
                                 line=dict(color="#FF8C00", width=1.8)), row=current_row, col=1)
    if show_vol_profile and "volume" in h_data.columns:
        current_row += 1
        # Classic volume-by-bar (works with equal-width x-axis). Horizontal "volume profile"
        # was broken under shared_xaxes and is intentionally replaced.
        try:
            vol_colors = [
                "rgba(0,255,127,0.55)" if c >= o else "rgba(255,69,0,0.55)"
                for o, c in zip(
                    h_data["open"] if "open" in h_data.columns else h_data["close"],
                    h_data["close"],
                )
            ]
            fig.add_trace(
                go.Bar(
                    x=x_pos,
                    y=h_data["volume"],
                    name="Volume",
                    marker_color=vol_colors,
                    showlegend=False,
                ),
                row=current_row,
                col=1,
            )
        except Exception:
            fig.add_trace(
                go.Bar(
                    x=x_pos,
                    y=h_data["volume"],
                    name="Volume",
                    marker_color="rgba(0, 191, 255, 0.5)",
                    showlegend=False,
                ),
                row=current_row,
                col=1,
            )
    if show_vs_btc and "vs_btc" in h_data.columns:
        current_row += 1
        fig.add_trace(go.Scatter(x=x_pos, y=h_data["vs_btc"], name="vs BTC %",
                                 line=dict(color="#00FFAA", width=1.8)), row=current_row, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=current_row, col=1)
    fig.update_layout(
        height=220 + (rows * 150),
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=10, r=12, t=40, b=10),
        legend=dict(orientation="h", y=1.02),
        xaxis_rangeslider_visible=False,
        font=dict(family="IBM Plex Sans, sans-serif"),
        # Drawing tools (trend lines, zones, freehand) — used with Plotly modebar
        dragmode="zoom",
        # Keep zoom/pan/drawn shapes when data refreshes
        uirevision=revision_key,
        newshape=dict(
            line=dict(color="#00FFFF", width=2),
            fillcolor="rgba(0, 255, 255, 0.12)",
            opacity=1,
            label=dict(texttemplate="", font=dict(size=11, color="#fafafa")),
        ),
    )
    # Equal-width x-axis: no time gaps, chart always fills the panel
    n = len(h_data)
    tick_step = max(1, n // 6) if n else 1
    tickvals = list(range(0, n, tick_step)) if n else []
    ticktext = []
    for i in tickvals:
        try:
            ticktext.append(pd.Timestamp(time_index[i]).strftime("%m/%d %H:%M"))
        except Exception:
            ticktext.append(str(time_index[i])[:16])
    # IMPORTANT: do not force xaxis.range on every rebuild — that resets pan/zoom.
    # uirevision keeps the user's view; only set ticks + disable rangeslider.
    for i in range(1, rows + 1):
        fig.update_xaxes(
            rangeslider_visible=False,
            tickvals=tickvals,
            ticktext=ticktext,
            uirevision=revision_key,
            row=i, col=1,
        )
        fig.update_yaxes(uirevision=revision_key, row=i, col=1)
    return fig

# ============================================================
# 6. SIDEBAR CONTROLS
# ============================================================
st.sidebar.header("Controls")

# --- MAIN TICKER INPUT (session-state driven so Watchlist can update it) ---
if "main_tickers_value" not in st.session_state:
    st.session_state.main_tickers_value = "BTC-USD, ETH-USD, SOL-USD, XRP-USD"

# Shareable URLs: ?tickers=BTC-USD,ETH-USD  (also accepts ?ticker=)
if "url_tickers_applied" not in st.session_state:
    st.session_state.url_tickers_applied = False
if not st.session_state.url_tickers_applied:
    try:
        qp = st.query_params
        raw = qp.get("tickers", None) or qp.get("ticker", None)
        if raw:
            if isinstance(raw, list):
                raw = ",".join(raw)
            parsed = [t.strip().upper() for t in str(raw).replace(" ", ",").split(",") if t.strip()]
            norm = normalize_user_tickers(parsed)
            if norm:
                st.session_state.main_tickers_value = ", ".join(norm[:6])
        st.session_state.url_tickers_applied = True
    except Exception:
        st.session_state.url_tickers_applied = True

t_raw = st.sidebar.text_input(
    "Enter Tickers (commas):",
    value=st.session_state.main_tickers_value,
    key="main_tickers_input",
    help="Crypto: type BTC, ETH, SOL (auto -USD). Stocks/ETFs: AAPL, SPY, etc.",
)
# Keep session state in sync when user types
if t_raw != st.session_state.main_tickers_value:
    st.session_state.main_tickers_value = t_raw

tkrs = [t.strip().upper() for t in st.session_state.main_tickers_value.split(",") if t.strip()]
tkrs = normalize_user_tickers(tkrs)
# Performance cap — free Yahoo endpoints stay responsive
MAX_TICKERS = 6
if len(tkrs) > MAX_TICKERS:
    st.sidebar.warning(f"Using first {MAX_TICKERS} tickers (max for speed).")
    tkrs = tkrs[:MAX_TICKERS]
# Reflect normalized symbols in the sidebar field (BTC -> BTC-USD) without fighting mid-typing too hard
_norm_display = ", ".join(tkrs)
if tkrs and _norm_display != st.session_state.main_tickers_value and all(
    len(x) > 2 for x in st.session_state.main_tickers_value.replace(" ", "").split(",") if x
):
    # Only sync when user finished a token (avoid rewriting while typing "BT")
    if not st.session_state.main_tickers_value.strip().endswith(","):
        st.session_state.main_tickers_value = _norm_display

# Keep URL in sync so the address bar is shareable
try:
    if tkrs:
        st.query_params["tickers"] = ",".join(tkrs)
except Exception:
    pass

with st.sidebar.expander("🔗 Share this setup"):
    if tkrs:
        share_q = ",".join(tkrs)
        st.caption("Copy your app URL from the browser, or build a link:")
        st.code(f"?tickers={share_q}", language=None)
        st.caption("Example: https://your-app.streamlit.app/?tickers=BTC-USD,ETH-USD")
    else:
        st.caption("Add tickers to generate a share link.")

res_disp = st.sidebar.selectbox("Resolution", list(RESOLUTION_MAP.keys()))
# Fixed refresh for live prices / candles / macro. Heavy blocks use longer cache TTLs (~3 min).
FIXED_REFRESH_SEC = 60
rf_rate = FIXED_REFRESH_SEC
st.sidebar.caption("Live data ~60s · Chart tab has Analyze/Live modes")
chart_mode = st.sidebar.radio(
    "Chart mode",
    ["Analyze (pan/zoom stays)", "Live (auto candles)"],
    index=0,
    help="Analyze pauses auto-refresh on the Chart tab so your view stays put. Live updates candles ~45s but may reset pan/zoom.",
)

st.sidebar.subheader("Charts")
show_candles = st.sidebar.checkbox("Candlesticks", value=True)
show_sma50 = st.sidebar.checkbox("50 SMA", value=True)
show_sma200 = st.sidebar.checkbox("200 SMA", value=False)
show_rsi = st.sidebar.checkbox("RSI", value=True)
show_vol_profile = st.sidebar.checkbox("Volume", value=True)

with st.sidebar.expander("More indicators", expanded=False):
    show_bb = st.checkbox("Bollinger Bands", value=False)
    show_macd = st.checkbox("MACD", value=False)
    show_stoch = st.checkbox("Stochastic", value=False)
    show_atr = st.checkbox("ATR", value=False)
    show_baseline = st.checkbox("Period baseline", value=False)
    show_long_baseline = st.checkbox("Long-term baseline", value=False)
    show_vwap = st.checkbox("VWAP", value=False)
    show_vs_btc = st.checkbox("vs BTC", value=False)
    # On-chain metrics always on under Crypto tab (no sidebar toggle)
    show_onchain = True
    show_patterns = st.checkbox("Pattern recognition", value=False)
    st.caption("Doji, Hammer, Stars, Engulfing, Morning/Evening Star, Soldiers/Crows, H&S, Double Top/Bottom, S/R")


st.sidebar.subheader("🔑 Optional API Keys")
# Prefer secrets only — never put the real key into a widget (Streamlit password fields can be revealed via the eye icon)
fred_api_key = ""
try:
    fred_api_key = (st.secrets.get("fred_api_key", "") or "").strip()
except Exception:
    fred_api_key = ""

if fred_api_key:
    st.sidebar.caption("FRED key loaded from secrets ✓ (not shown in UI)")
else:
    st.sidebar.caption(
        "Add `fred_api_key` in Streamlit **Secrets** to unlock FRED data. "
        "Get a free key: https://fred.stlouisfed.org/docs/api/api_key.html"
    )
    # Optional local-only paste: stored in session, not prefilled from secrets, cleared guidance
    with st.sidebar.expander("Paste FRED key for this session only"):
        _session_fred = st.text_input(
            "FRED key (this session)",
            value="",
            type="password",
            help="Used only until the app restarts. Prefer Secrets so the key is never shown in the UI.",
            key="fred_key_session_only",
        )
        if _session_fred.strip():
            fred_api_key = _session_fred.strip()
            st.caption("Session key active (can still be revealed via the eye icon — Secrets is safer).")


res_cfg = RESOLUTION_MAP[res_disp]
interval = res_cfg["interval"]
lookback_bars = res_cfg["lookback"]
period = res_cfg["period"]
is_6h = res_cfg["is_6h"]

# ============================================================
# 7. MAIN LAYOUT
# ============================================================



st.markdown(
    "<div class='app-header'>"
    "<div class='silver-title'>OneStop-InfoSpot</div>"
    "<div class='silver-subtitle'>Market intelligence desk · stocks · crypto · macro</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.caption("Dark mode • Candlesticks • Pattern Recognition • Long/Short Allocation • Open Interest • Funding • RSI • MACD • Bollinger • On-chain • Global Macro • 15Y Historical • Predictive Analytics")

# Radio nav = only the active section runs heavy code (faster than st.tabs)
# Optional deep-link: ?view=briefing|crypto|compare|macro|screener
_view_map = {
    "briefing": "☀️ Briefing",
    "chart": "📈 Chart",
    "compare": "🔀 Compare/Analysis",
    "analysis": "🔀 Compare/Analysis",
    "crypto": "⚡ Crypto",
    "macro": "🌐 Macro/FRED",
    "fred": "🌐 Macro/FRED",
    "news": "📰 LiveMacro/Crypto/StockNews",
    "predictive": "🔮 Predictive",
    "tools": "🛠️ Tools",
}
try:
    _vq = (st.query_params.get("view") or "").lower()
    if _vq in _view_map and "main_nav" not in st.session_state:
        st.session_state.main_nav = _view_map[_vq]
except Exception:
    pass

nav = st.radio(
    "Navigate",
    ["☀️ Briefing", "📈 Chart", "🔀 Compare/Analysis", "⚡ Crypto", "🌐 Macro/FRED", "📰 LiveMacro/Crypto/StockNews", "🔮 Predictive", "🛠️ Tools"],
    horizontal=True,
    label_visibility="collapsed",
    key="main_nav",
)

# Pause auto-refresh only in Chart + Analyze mode (keeps pan/zoom)
# Live mode on Chart (and all other tabs) still auto-refresh ~45s
try:
    _cm = chart_mode
except NameError:
    _cm = "Analyze (pan/zoom stays)"
_pause_refresh = (nav == "📈 Chart" and str(_cm).startswith("Analyze"))
if not _pause_refresh:
    st_autorefresh(interval=rf_rate * 1000, key="datarefresh")




# ============================================================
# BRIEFING TAB (expanded daily command center)
# ============================================================

if nav == "☀️ Briefing":
    _asof = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    st.subheader("☀️ Morning Brief")
    st.caption(f"Cross-asset desk · stocks + crypto + policy · **as of {_asof}**")
    if not tkrs:
        st.info("Add tickers in the sidebar for movers, funding, and equity snapshots. Board, calendar, sectors & news still load below.")

    # Always load light desk data (even with no tickers)
    try:
        fng_v, fng_c = get_fear_greed()
    except Exception:
        fng_v, fng_c = "—", "—"
    try:
        macro_snap = load_macro_data()
    except Exception:
        macro_snap = {}
    try:
        defi_snap = load_defi_overview()
    except Exception:
        defi_snap = {}
    brief_data = {}
    if tkrs:
        with st.spinner("Loading briefing…"):
            brief_data = get_shared_market_data(tkrs, interval, lookback_bars, period, is_6h)

    if True:  # keep indentation structure for rest of briefing body
        # --- Regime one-liner ---
        regime = "Mixed"
        try:
            fv = float(fng_v)
            vix = float(macro_snap["VIX"]["close"].iloc[-1]) if "VIX" in macro_snap and not macro_snap["VIX"].empty else None
            if fv <= 25 and (vix is None or vix >= 20):
                regime = "Stress / Fear — historically better for patient entries"
            elif fv >= 75:
                regime = "Greed — late-cycle risk; favor discipline over FOMO"
            elif vix is not None and vix >= 25:
                regime = "Elevated volatility — size down, wait for clarity"
            else:
                regime = "Neutral / mixed — follow your levels"
        except Exception:
            pass
        st.markdown(f"**Regime read:** {regime}")

        # --- MARKET PULSE (attention hero) ---
        try:
            _pulse_board = load_live_board()
            st.session_state["_brief_board"] = _pulse_board
            _movers = []
            if _pulse_board is not None and not _pulse_board.empty and "Chg%" in _pulse_board.columns:
                for _, _r in _pulse_board.iterrows():
                    try:
                        _movers.append((str(_r["Symbol"]), float(_r["Chg%"]), _r.get("Price")))
                    except Exception:
                        continue
            # Include sidebar tickers from brief_data if available
            if brief_data:
                for _t, _df in brief_data.items():
                    if _df is None or _df.empty or "close" not in _df.columns or len(_df) < 2:
                        continue
                    try:
                        _ch = (float(_df["close"].iloc[-1]) / float(_df["close"].iloc[-2]) - 1) * 100
                        _movers.append((_t, _ch, float(_df["close"].iloc[-1])))
                    except Exception:
                        pass
            # dedupe by symbol keep first
            _seen = set()
            _uniq = []
            for _s, _c, _p in _movers:
                if _s in _seen:
                    continue
                _seen.add(_s)
                _uniq.append((_s, _c, _p))
            if _uniq:
                _uniq.sort(key=lambda x: abs(x[1]), reverse=True)
                _top = _uniq[0]
                _sym, _chg, _px = _top
                _cls = "up" if _chg >= 0 else "down"
                _dir = "leading the board" if _chg >= 0 else "heaviest pressure"
                _px_s = format_price(_px) if _px is not None else ""
                _chips = []
                for _s, _c, _ in _uniq[1:6]:
                    _ccls = "up" if _c >= 0 else "down"
                    _chips.append(
                        f"<span class='pulse-chip {_ccls}'>{_s.replace('-USD','')} {_c:+.2f}%</span>"
                    )
                _chip_html = "".join(_chips)
                st.markdown(
                    f"""
                    <div class="pulse-hero">
                      <div class="pulse-label">Market pulse</div>
                      <div class="pulse-sym">{_sym.replace("-USD", "")} {_px_s}</div>
                      <div class="pulse-chg {_cls}">{_chg:+.2f}%</div>
                      <div class="pulse-sub">{_dir} · live board + your tickers</div>
                      <div class="pulse-tape">{_chip_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

        # --- Event banner (today / tomorrow / next 2 weeks) ---
        try:
            _cal = get_economic_calendar(days_ahead=14)
            if _cal is not None and not _cal.empty:
                _hot = _cal[_cal["Days"] <= 1]
                if not _hot.empty:
                    _ev = " · ".join(
                        [
                            f"**{r.Event}** ({'today' if int(r.Days) == 0 else 'tomorrow'})"
                            for _, r in _hot.head(3).iterrows()
                        ]
                    )
                    st.warning(f"📅 Event window: {_ev} — expect volatility around the print.")
                else:
                    _next = _cal.sort_values("Days").iloc[0]
                    _when = "today" if int(_next["Days"]) == 0 else (
                        "tomorrow" if int(_next["Days"]) == 1 else f"in {int(_next['Days'])} days"
                    )
                    st.info(
                        f"📅 Next high-impact event: **{_next['Event']}** on {_next['Date']} ({_when})"
                        + (f" — {_next['Notes']}" if _next.get("Notes") else "")
                    )
        except Exception:
            pass

        # --- Since last visit (session snapshot) ---
        try:
            _snap_prices = {}
            if brief_data:
                for _t, _df in brief_data.items():
                    if _df is not None and not _df.empty and "close" in _df.columns:
                        _snap_prices[_t] = float(_df["close"].iloc[-1])
            # also board majors if present later — update again after board load
            if _snap_prices:
                _diffs = update_and_diff_last_visit(_snap_prices)
                if _diffs:
                    _ago = st.session_state.get("_last_visit_ago", "")
                    st.info("Since last visit" + str(_ago) + ": " + " · ".join(_diffs))
        except Exception:
            pass

        # --- 1) Session checklist ---
        sess = get_us_session_label()
        st.markdown(f"**Session:** {sess['session']} · {sess['et_clock']}")
        st.caption(sess["focus"])
        try:
            cal_today = get_economic_calendar(days_ahead=3)
            if cal_today is not None and not cal_today.empty:
                soon = cal_today[cal_today["Days"] <= 1]
                if not soon.empty:
                    bits = [f"{r.Event} ({r.Date})" for _, r in soon.head(3).iterrows()]
                    st.caption("Near-term calendar: " + " · ".join(bits))
        except Exception:
            pass

        # BTC / SPY day change for risk score (from board or brief)
        _btc_chg = _spy_chg = None
        try:
            if brief_data and "BTC-USD" in brief_data and not brief_data["BTC-USD"].empty:
                bdf = brief_data["BTC-USD"]
                if "close" in bdf.columns and len(bdf) > 1:
                    _btc_chg = (float(bdf["close"].iloc[-1]) / float(bdf["close"].iloc[-2]) - 1) * 100
        except Exception:
            pass
        try:
            _board_tmp = load_live_board()
            if _board_tmp is not None and not _board_tmp.empty:
                row = _board_tmp[_board_tmp["Symbol"] == "SPY"]
                if not row.empty and "Chg%" in row.columns:
                    _spy_chg = float(row["Chg%"].iloc[0])
                rowb = _board_tmp[_board_tmp["Symbol"] == "BTC-USD"]
                if _btc_chg is None and not rowb.empty and "Chg%" in rowb.columns:
                    _btc_chg = float(rowb["Chg%"].iloc[0])
        except Exception:
            _board_tmp = None
        # Reuse for live board section (avoid second fetch in same run)
        st.session_state["_brief_board"] = _board_tmp

        # --- 4) Risk-on / risk-off score ---
        risk = compute_risk_score(macro_snap, fng_v, btc_chg=_btc_chg, spy_chg=_spy_chg)
        r1, r2, r3 = st.columns([1, 1, 2])
        r1.metric("Risk appetite", f"{risk['color']} {risk['score']}/100")
        r2.metric("Bias", risk["label"])
        r3.caption("Drivers: " + (", ".join(risk["details"]) if risk["details"] else "limited inputs"))

        # --- 2 & 3) Gap scanner + levels for sidebar tickers (capped) ---
        gap_syms = []
        if tkrs:
            gap_syms.extend(tkrs[:6])
        for s in ("SPY", "QQQ", "BTC-USD"):
            if s not in gap_syms:
                gap_syms.append(s)
        try:
            levels_df = load_gap_and_levels(tuple(gap_syms[:10]))
        except Exception:
            levels_df = pd.DataFrame()

        if levels_df is not None and not levels_df.empty:
            st.markdown("##### Gap scanner")
            st.caption("Overnight/session gap vs prior close · rel volume · vs SMA50 (daily Yahoo, ~2 min cache)")
            gshow = levels_df[["Symbol", "Price", "Gap %", "Rel Vol", "vs SMA50 %"]].copy()
            gshow = gshow.sort_values("Gap %", key=lambda s: s.abs(), ascending=False)
            gshow["Price"] = gshow["Price"].map(lambda x: format_price(x))
            gshow["Gap %"] = gshow["Gap %"].map(lambda x: f"{x:+.2f}%")
            gshow["vs SMA50 %"] = gshow["vs SMA50 %"].map(lambda x: f"{x:+.2f}%")
            gshow["Rel Vol"] = gshow["Rel Vol"].map(lambda x: f"{x:.2f}x" if pd.notna(x) else "—")
            st.dataframe(gshow, use_container_width=True, hide_index=True)

            st.markdown("##### Levels cards")
            st.caption("Prior day H/L/C · floor pivot · R1/S1 · 52w range")
            show_n = min(4, len(levels_df))
            cols = st.columns(show_n)
            top = levels_df.sort_values("Gap %", key=lambda s: s.abs(), ascending=False).head(show_n)
            for i, (_, row) in enumerate(top.iterrows()):
                with cols[i]:
                    st.markdown(f"**{row['Symbol']}**")
                    st.write(format_price(row["Price"]))
                    st.caption(
                        f"Prev C {format_price(row['Prev C'])} · H {format_price(row['Prev H'])} · L {format_price(row['Prev L'])}"
                    )
                    st.caption(
                        f"Pivot {format_price(row['Pivot'])} · R1 {format_price(row['R1'])} · S1 {format_price(row['S1'])}"
                    )
                    st.caption(
                        f"52w {format_price(row['52W Low'])} – {format_price(row['52W High'])}"
                    )

        # --- Top metrics ---
        c = st.columns(6)
        c[0].metric("Fear & Greed", f"{fng_v}", str(fng_c)[:16] if fng_c else "")
        if "VIX" in macro_snap and not macro_snap["VIX"].empty:
            c[1].metric("VIX", f"{float(macro_snap['VIX']['close'].iloc[-1]):.1f}")
        else:
            c[1].metric("VIX", "—")
        if "DXY (U.S. Dollar Index)" in macro_snap and not macro_snap["DXY (U.S. Dollar Index)"].empty:
            dxy = macro_snap["DXY (U.S. Dollar Index)"]
            px = float(dxy["close"].iloc[-1])
            ch = ((px / float(dxy["close"].iloc[-2])) - 1) * 100 if len(dxy) > 1 else 0
            c[2].metric("DXY", f"{px:.1f}", f"{ch:+.1f}%")
        else:
            c[2].metric("DXY", "—")
        if "U.S. 10Y Treasury Yield" in macro_snap and not macro_snap["U.S. 10Y Treasury Yield"].empty:
            c[3].metric("US 10Y", f"{float(macro_snap['U.S. 10Y Treasury Yield']['close'].iloc[-1]):.2f}%")
        else:
            c[3].metric("US 10Y", "—")
        try:
            y2, y10 = macro_snap.get("U.S. 2Y Yield"), macro_snap.get("U.S. 10Y Treasury Yield")
            if y2 is not None and y10 is not None and not y2.empty and not y10.empty:
                common = y2["close"].index.intersection(y10["close"].index)
                sp = float(y10["close"].loc[common].iloc[-1] - y2["close"].loc[common].iloc[-1])
                c[4].metric("2s10s", f"{sp:.2f}", "Inv" if sp < 0 else "Pos")
            else:
                c[4].metric("2s10s", "—")
        except Exception:
            c[4].metric("2s10s", "—")
        if defi_snap.get("total_tvl"):
            c[5].metric("TVL", f"${defi_snap['total_tvl']/1e9:.1f}B")
        else:
            c[5].metric("TVL", "—")

        # --- Live market board (stocks + crypto) ---
        st.markdown("##### Live market board")
        st.caption("Major ETFs, mega-cap stocks, and crypto — last daily close & change (Yahoo, ~45s cache)")
        try:
            board = st.session_state.get("_brief_board")
            if board is None:
                board = load_live_board()
            if board is not None and not board.empty:
                # Split stocks vs crypto for readability
                crypto_syms = {"BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"}
                stocks = board[~board["Symbol"].isin(crypto_syms)]
                cryptos = board[board["Symbol"].isin(crypto_syms)]
                b1, b2 = st.columns(2)
                with b1:
                    st.markdown("**Stocks / ETFs**")
                    if not stocks.empty:
                        disp = stocks.copy()
                        disp["Price"] = disp["Price"].map(lambda x: format_price(x))
                        disp["Chg%"] = disp["Chg%"].map(lambda x: f"{x:+.2f}%")
                        st.dataframe(disp, use_container_width=True, hide_index=True)
                with b2:
                    st.markdown("**Crypto**")
                    if not cryptos.empty:
                        disp = cryptos.copy()
                        disp["Price"] = disp["Price"].map(lambda x: format_price(x))
                        disp["Chg%"] = disp["Chg%"].map(lambda x: f"{x:+.2f}%")
                        st.dataframe(disp, use_container_width=True, hide_index=True)
                    # Also surface user tickers live if different
                    if brief_data:
                        extra = []
                        for t, df in brief_data.items():
                            if df is None or df.empty or "close" not in df.columns:
                                continue
                            if t in crypto_syms:
                                continue
                            px = float(df["close"].iloc[-1])
                            ch = (px / float(df["close"].iloc[-2]) - 1) * 100 if len(df) > 1 else 0
                            extra.append({"Symbol": t, "Price": f"${px:,.4g}", "Chg%": f"{ch:+.2f}%"})
                        if extra:
                            st.markdown("**Your tickers**")
                            st.dataframe(pd.DataFrame(extra), use_container_width=True, hide_index=True)
            else:
                st.caption("Live board temporarily unavailable.")
        except Exception:
            st.caption("Live board temporarily unavailable.")

        # --- Economic calendar + sectors ---
        cal_col, sec_col = st.columns(2)
        with cal_col:
            st.markdown("##### Economic calendar")
            st.caption("High-impact US releases (scheduled dates — confirm on release day)")
            try:
                cal = get_economic_calendar(days_ahead=60)
                if cal is not None and not cal.empty:
                    show = cal.copy()
                    show["When"] = show["Days"].map(lambda d: "Today" if d == 0 else ("Tomorrow" if d == 1 else f"in {d}d"))
                    st.dataframe(
                        show[["Date", "When", "Event", "Impact", "Notes"]],
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.caption("No upcoming events in the next 60 days in the built-in list.")
            except Exception:
                st.caption("Calendar unavailable.")
        with sec_col:
            st.markdown("##### Sector board")
            st.caption("US sector ETFs — sorted by 1-day change")
            try:
                sec = load_sector_board()
                if sec is not None and not sec.empty:
                    disp = sec.copy()
                    disp["Price"] = disp["Price"].map(lambda x: format_price(x))
                    disp["1D %"] = disp["1D %"].map(lambda x: f"{x:+.2f}%")
                    disp["1W %"] = disp["1W %"].map(lambda x: f"{x:+.2f}%")
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.caption("Sector data unavailable.")
            except Exception:
                st.caption("Sector data unavailable.")

        st.markdown("---")
        left, right = st.columns(2)

        with left:
            st.markdown("##### Movers")
            if not brief_data:
                st.caption("Price data unavailable — try fewer tickers or another resolution.")
            else:
                rows = []
                for t, df in brief_data.items():
                    if df is None or len(df) < 2 or "close" not in df.columns:
                        continue
                    px = float(df["close"].iloc[-1])
                    ch = (px / float(df["close"].iloc[-2]) - 1) * 100
                    rsi = df["rsi"].iloc[-1] if "rsi" in df.columns else None
                    rows.append({
                        "Ticker": t.replace("-USD", ""),
                        "Price": format_price(px),
                        "Chg": f"{ch:+.2f}%",
                        "RSI": f"{float(rsi):.0f}" if rsi is not None and pd.notna(rsi) else "—",
                    })
                rows.sort(key=lambda r: float(r["Chg"].replace("%", "")), reverse=True)
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        with right:
            st.markdown("##### Funding / positioning")
            frows = []
            for t, df in (brief_data or {}).items():
                if df is None or df.empty or "close" not in df.columns:
                    continue
                try:
                    fut = get_futures_metrics(t, current_price=float(df["close"].iloc[-1]))
                    fr = fut.get("funding_rate")
                    frows.append({
                        "Ticker": t.replace("-USD", ""),
                        "Long%": f"{fut.get('long_pct', 50):.0f}",
                        "Short%": f"{fut.get('short_pct', 50):.0f}",
                        "Funding": f"{fr:+.3f}%" if fr is not None else "—",
                    })
                except Exception:
                    continue
            if frows:
                st.dataframe(pd.DataFrame(frows), use_container_width=True, hide_index=True)
            else:
                st.caption("Futures metrics unavailable right now.")

        st.markdown("##### Market-moving headlines")
        st.caption(f"Fed · SEC · Stocks · Crypto — free RSS (~15 min cache) · as of {_asof}")
        try:
            news = get_market_news(cache_buster="news-v1", max_items=40)
            if not news:
                st.caption("No headlines right now.")
            else:
                order = ["Fed", "SEC", "Stocks", "Crypto"]
                cols = st.columns(4)
                for i, cat in enumerate(order):
                    with cols[i]:
                        st.markdown(f"**{cat}**")
                        items = [a for a in news if a.get("category") == cat][:4]
                        if not items:
                            st.caption("—")
                        for art in items:
                            title = (art.get("title") or "")[:90]
                            link = art.get("link") or "#"
                            src = art.get("source") or ""
                            st.markdown(f"- [{title}]({link}) · *{src}*")

                # B) Ticker-aware news
                if tkrs:
                    hits = filter_news_for_tickers(news, tkrs)
                    st.markdown("##### Related to your tickers")
                    if hits:
                        for art in hits[:8]:
                            title = (art.get("title") or "")[:110]
                            link = art.get("link") or "#"
                            src = art.get("source") or ""
                            matched = art.get("matched") or ""
                            st.markdown(f"- [{title}]({link}) · *{src}* · matched: `{matched}`")
                    else:
                        st.caption("No headlines matched your current tickers in the latest feed pull.")
        except Exception:
            st.caption("News feed unavailable.")

        # C) Equity snapshots for stock/ETF tickers (skip pure crypto -USD pairs)
        if tkrs:
            stockish = [t for t in tkrs if not str(t).upper().endswith("-USD")]
            # also allow plain crypto symbols without snapshot spam
            stockish = [t for t in stockish if t.upper() not in {"BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"}]
            if stockish:
                st.markdown("##### Equity snapshots")
                st.caption("Yahoo free fields — PE/sector may be blank for some ETFs")
                cols = st.columns(min(3, len(stockish)))
                for i, sym in enumerate(stockish[:6]):
                    with cols[i % len(cols)]:
                        snap = get_equity_snapshot(sym)
                        if snap.get("Error"):
                            st.warning(f"{sym}: {snap['Error']}")
                            continue
                        st.markdown(f"**{snap.get('Name') or sym}** (`{sym}`)")
                        px = snap.get("Price")
                        st.write(f"Price: **{format_price(px).lstrip('$')}**" if isinstance(px, (int, float)) else f"Price: {px or '—'}")
                        mcap = snap.get("Market Cap")
                        if isinstance(mcap, (int, float)) and mcap:
                            if mcap >= 1e12:
                                mc_s = f"${mcap/1e12:.2f}T"
                            elif mcap >= 1e9:
                                mc_s = f"${mcap/1e9:.1f}B"
                            else:
                                mc_s = f"${mcap/1e6:.0f}M"
                            st.write(f"Market cap: {mc_s}")
                        pe = snap.get("PE")
                        if isinstance(pe, (int, float)):
                            st.write(f"PE: {pe:.1f}")
                        if snap.get("Sector"):
                            st.write(f"{snap.get('Sector')} · {snap.get('Industry') or ''}")
                        if snap.get("52W Range"):
                            st.caption(f"52W: {snap['52W Range']}")
                        st.caption(snap.get("As Of") or "")

        st.caption("Next: **Chart** for candles · **Compare/Analysis** for relative strength · **Crypto** for positioning & yields")


if nav == "🔀 Compare/Analysis":
    st.subheader("🔀 Compare / Analysis")
    st.caption("Returns, volatility, correlation, and relative performance for your loaded tickers.")

    if not tkrs or len(tkrs) < 1:
        st.info("Load 2+ tickers in the sidebar for the best comparison.")
    else:
        cmp_data = get_shared_market_data(tkrs, interval, lookback_bars, period, is_6h)
        if not cmp_data:
            st.warning("Could not load comparison data.")
        else:
            # Performance table
            st.markdown("### Returns snapshot")
            perf = compute_multi_period_performance(cmp_data)
            if not perf.empty:
                disp = perf.copy()
                for col in ["1D %", "7D %", "30D %", "90D %", "YTD %"]:
                    if col in disp.columns:
                        disp[col] = disp[col].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else "N/A")
                st.dataframe(disp.set_index("Ticker"), use_container_width=True)

            # Normalized chart
            st.markdown("### Normalized performance (session lookback)")
            norm = pd.DataFrame()
            for t, df in cmp_data.items():
                if df is not None and not df.empty and "close" in df.columns:
                    base = float(df["close"].iloc[0])
                    if base:
                        norm[t] = (df["close"] / base - 1) * 100
            if not norm.empty:
                fig = go.Figure()
                for col in norm.columns:
                    fig.add_trace(go.Scatter(x=norm.index, y=norm[col], name=col, line=dict(width=2)))
                fig.update_layout(
                    height=380, template="plotly_dark",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                    margin=dict(l=10, r=10, t=20, b=10),
                    legend=dict(orientation="h", y=1.1),
                    font=dict(family="IBM Plex Sans, sans-serif"),
                    yaxis_title="% from start of window",
                )
                st.plotly_chart(fig, use_container_width=True)

            # Vol + correlation strip
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Volatility")
                vol = compute_volatility_stats(cmp_data)
                if not vol.empty:
                    vdisp = vol.copy()
                    vdisp["Price"] = vdisp["Price"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A")
                    for col in ["Vol 7d (ann%)", "Vol 30d (ann%)", "ATR %"]:
                        if col in vdisp.columns:
                            vdisp[col] = vdisp[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
                    if "ATR" in vdisp.columns:
                        vdisp["ATR"] = vdisp["ATR"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
                    st.dataframe(vdisp.set_index("Ticker"), use_container_width=True)
            with c2:
                st.markdown("### Correlation")
                corr = compute_correlation_matrix(cmp_data)
                if not corr.empty and len(corr) >= 2:
                    fig_c = go.Figure(data=go.Heatmap(
                        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                        colorscale="RdBu", zmid=0,
                        text=np.round(corr.values, 2), texttemplate="%{text}",
                    ))
                    fig_c.update_layout(
                        height=320, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        margin=dict(l=10, r=10, t=10, b=10),
                        font=dict(family="IBM Plex Sans, sans-serif"),
                    )
                    st.plotly_chart(fig_c, use_container_width=True)
                else:
                    st.caption("Need ≥2 tickers with overlapping history.")



if nav == "📈 Chart":
    st.subheader("📈 Chart workspace")
    mode = str(chart_mode) if "chart_mode" in dir() or "chart_mode" in locals() else "Analyze (pan/zoom stays)"
    if mode.startswith("Analyze"):
        st.caption("**Analyze mode** — auto-refresh paused. Pan/zoom/drawings stay put. Click refresh for new candles.")
        b1, b2 = st.columns([1, 4])
        with b1:
            if st.button("Refresh candles", key="chart_manual_refresh"):
                try:
                    get_shared_market_data.clear()
                except Exception:
                    pass
                st.rerun()
        with b2:
            st.caption("Switch to **Live** in the sidebar for ~45s candle updates (may reset pan/zoom).")
    else:
        st.caption("**Live mode** — candles refresh ~45s. Pan/zoom may reset on each update (Streamlit limitation). Use Analyze mode to study a level.")

    if not tkrs:
        st.info("Enter at least one ticker in the sidebar.")
    else:
        data = get_shared_market_data(tkrs, interval, lookback_bars, period, is_6h)
        if not data:
            st.error("No valid data loaded. Check tickers or resolution.")
        else:
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

            asset_options = list(data.keys())
            if asset_options:
                if "target_asset_select" not in st.session_state or st.session_state.target_asset_select not in asset_options:
                    st.session_state.target_asset_select = asset_options[0]
            m_tgt = st.selectbox(
                "Select Target Asset:",
                asset_options,
                key="target_asset_select",
            )
            if m_tgt and m_tgt in data:
                h_data = data[m_tgt].copy()
                if show_long_baseline:
                    long_avg = get_long_term_avg(m_tgt)
                    if long_avg is not None:
                        h_data["long_baseline"] = long_avg

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
                    revision_key=f"asset-view-{m_tgt}",
                )
                if show_long_baseline and "long_baseline" in h_data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=list(range(len(h_data))),
                            y=h_data["long_baseline"],
                            name="1Y Baseline",
                            line=dict(color="#FF8C00", width=1.8, dash="dashdot"),
                        ),
                        row=1, col=1,
                    )
                last_px = float(h_data["close"].iloc[-1]) if "close" in h_data.columns else None
                if last_px is not None:
                    st.metric(f"{m_tgt} last", format_price(last_px))

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "editable": True,
                        "edits": {"shapePosition": True, "annotationPosition": True},
                        "modeBarButtonsToAdd": [
                            "drawline",
                            "drawopenpath",
                            "drawcircle",
                            "drawrect",
                            "eraseshape",
                        ],
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    },
                    key=f"asset_chart_draw_{m_tgt}",
                )

                if show_vol_profile:
                    vp_fig = build_volume_by_price_chart(h_data)
                    if vp_fig is not None:
                        st.plotly_chart(vp_fig, use_container_width=True)

                if show_patterns:
                    if detected_patterns:
                        st.markdown("#### 🔎 Detected Chart Patterns")
                        pattern_df = pd.DataFrame([
                            {
                                "Pattern": p["type"],
                                "Direction": p["direction"],
                                "Confidence": p["confidence"],
                                "Price": format_price(p["price"]),
                                "Description": p["description"],
                            }
                            for p in detected_patterns
                        ])
                        st.dataframe(pattern_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No patterns detected on this window.")

                # Quick indicator snapshot
                cols = st.columns(5)
                if "rsi" in h_data.columns:
                    cols[0].metric("RSI", f"{h_data['rsi'].iloc[-1]:.1f}" if pd.notna(h_data['rsi'].iloc[-1]) else "—")
                if "macd" in h_data.columns:
                    cols[1].metric("MACD", f"{h_data['macd'].iloc[-1]:.4f}" if pd.notna(h_data['macd'].iloc[-1]) else "—")


if nav == "⚡ Crypto":
    st.subheader("⚡ Crypto")
    st.caption("Positioning, pivots, liquidity, on-chain & DeFi. Candles under **Chart** · News under **LiveMacro/Crypto/StockNews**.")
    if not tkrs:
        st.info("Enter at least one ticker in the sidebar.")
    else:
        data = get_shared_market_data(tkrs, interval, lookback_bars, period, is_6h)
        if not data:
            st.error("No valid data loaded. Check tickers or try a different resolution. Some data sources may be temporarily unavailable.")
        else:


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
                if "-USD" in t or t.replace("-", "").isalpha():
                    onchain_profiles[t] = get_real_onchain(t)
                if t in daily_pivots:
                    p = daily_pivots[t]
                    matrix_rows.append({
                        "Ticker": t, "Price": format_price(c_px), "Change": f"{delta:+.2f}%",
                        "Long %": f"{long_r:.1f}%", "Short %": f"{short_r:.1f}%",
                        "Pivot": f"${p['Pivot']:,.2f}", "S1": f"${p['S1']:,.2f}", "S2": f"${p['S2']:,.2f}",
                        "R1": f"${p['R1']:,.2f}", "R2": f"${p['R2']:,.2f}",
                    })
                else:
                    matrix_rows.append({
                        "Ticker": t, "Price": format_price(c_px), "Change": f"{delta:+.2f}%",
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
                                  font=dict(family="IBM Plex Sans, sans-serif"))
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("Candlestick workspace moved to the **📈 Chart** tab (Analyze mode keeps pan/zoom; Live mode auto-updates candles).")
            
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
                                    font=dict(family="IBM Plex Sans, sans-serif"))
                st.plotly_chart(fig_c, use_container_width=True)
            if onchain_profiles:
                st.subheader("🔗 Real On-chain Metrics")
                for ticker, metrics in onchain_profiles.items():
                    st.markdown(f"**{ticker}**")
                    st.dataframe(pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]),
                                 use_container_width=True, hide_index=True)
            if matrix_rows:
                st.subheader("🧮 Side-by-Side Pivot Matrix (Daily Pivots)")
                st.caption("Classic Floor Pivots calculated from the previous daily candle close parameters")
                st.dataframe(pd.DataFrame(matrix_rows).set_index("Ticker"), use_container_width=True)

    # DeFi (crypto liquidity layer)
    defi = load_defi_overview()
    # DeFi / Stablecoin layer
    st.markdown("---")
    st.markdown("#### 🦄 DeFi & Stablecoin Landscape (DefiLlama)")
    if defi.get("total_tvl"):
        d1, d2 = st.columns(2)
        d1.metric("Total Crypto TVL", f"${defi['total_tvl']:,.0f}")
        if defi.get("stablecoin_mcap"):
            d2.metric("Stablecoin Supply (approx)", f"${defi['stablecoin_mcap']:,.0f}")
        if defi.get("top_chains"):
            chain_df = pd.DataFrame(defi["top_chains"])
            chain_df["tvl"] = chain_df["tvl"].apply(lambda x: f"${x:,.0f}" if x else "N/A")
            st.dataframe(chain_df.rename(columns={"name": "Chain", "tvl": "TVL", "symbol": "Symbol"}),
                         use_container_width=True, hide_index=True)
    else:
        st.caption("DefiLlama data temporarily unavailable.")


if nav == "🌐 Macro/FRED":
    st.subheader("🌐 Macro / FRED")
    st.caption("Broad market telemetry · yields · commodities · official FRED series (CPI, Fed Funds, M2, unemployment).")

    st.subheader("Broad Market Telemetry")
    st.caption("Equities • Yields • Commodities • Volatility — free public feeds.")
    
    macro_data = load_macro_data()
    
    if not macro_data:
        st.info("Macro data temporarily unavailable. Try refreshing shortly.")
    else:
        # Metrics grid
        m_cols = st.columns(3)
        m_index = 0
        for name, m_df in macro_data.items():
            if not m_df.empty and "close" in m_df.columns:
                current_p = float(m_df["close"].iloc[-1])
                prior_p = float(m_df["close"].iloc[-2]) if len(m_df) > 1 else current_p
                pct_delta = ((current_p - prior_p) / prior_p) * 100
                col_target = m_cols[m_index % 3]
                if "Yield" in name or name == "VIX":
                    col_target.metric(name, f"{current_p:.2f}", f"{pct_delta:+.2f}%")
                else:
                    col_target.metric(name, f"${current_p:,.2f}", f"{pct_delta:+.2f}%")
                m_index += 1

        # 2s10s spread (classic recession signal)
        st.markdown("#### 📐 Yield Curve: 2Y vs 10Y")
        try:
            y2 = macro_data.get("U.S. 2Y Yield")
            y10 = macro_data.get("U.S. 10Y Treasury Yield")
            if y2 is not None and y10 is not None and not y2.empty and not y10.empty:
                common = y2["close"].index.intersection(y10["close"].index)
                if len(common) > 5:
                    spread = y10["close"].loc[common] - y2["close"].loc[common]
                    last_spread = float(spread.iloc[-1])
                    st.metric("2s10s Spread (10Y − 2Y)", f"{last_spread:.2f} pts",
                              "Inverted (recession watch)" if last_spread < 0 else "Positive")
                    fig_sp = go.Figure()
                    fig_sp.add_trace(go.Scatter(x=spread.index, y=spread, name="2s10s", line=dict(color="#FFD700", width=2)))
                    fig_sp.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig_sp.update_layout(
                        height=280, template="plotly_dark",
                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                        margin=dict(l=10, r=10, t=20, b=10),
                        font=dict(family="IBM Plex Sans, sans-serif"),
                        yaxis_title="Spread (pts)"
                    )
                    st.plotly_chart(fig_sp, use_container_width=True)
        except Exception:
            st.caption("Yield spread chart unavailable for this session.")

        st.markdown("#### 📉 Normalized Performance (6M)")
        macro_combined = pd.DataFrame()
        for name, m_df in macro_data.items():
            if not m_df.empty and "close" in m_df.columns and name != "Bitcoin":
                base_p = float(m_df["close"].iloc[0])
                if base_p:
                    macro_combined[name] = ((m_df["close"] / base_p) - 1) * 100
        if not macro_combined.empty:
            fig_macro = go.Figure()
            for col in macro_combined.columns:
                fig_macro.add_trace(go.Scatter(x=macro_combined.index, y=macro_combined[col], name=col, line=dict(width=1.5)))
            fig_macro.update_layout(
                height=420, template="plotly_dark",
                paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(orientation="h", y=1.12),
                font=dict(family="IBM Plex Sans, sans-serif")
            )
            st.plotly_chart(fig_macro, use_container_width=True)


    # ----- FRED Official Macro Series -----
    st.markdown("---")
    st.markdown("#### 🏛️ FRED Official Macro Series")
    st.caption("Federal Reserve Economic Data — free with your API key. Series cached 24h.")
    
    if not fred_api_key or not str(fred_api_key).strip():
        st.info(
            "Add a **free FRED API key** in the sidebar to unlock CPI, Fed Funds, M2, Unemployment, "
            "and more. Get one in ~1 minute at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html)."
        )
    else:
        fred_data = load_fred_dashboard(fred_api_key)
        if not fred_data:
            st.warning("FRED key present but no series loaded. Check the key or try again later.")
        else:
            fcols = st.columns(3)
            fi = 0
            for label, fdf in fred_data.items():
                if fdf is None or fdf.empty:
                    continue
                last = float(fdf["value"].iloc[-1])
                prev = float(fdf["value"].iloc[-2]) if len(fdf) > 1 else last
                delta = last - prev
                # Format by series type
                if "Rate" in label or "Unemployment" in label or "Yield" in label or "Funds" in label:
                    disp = f"{last:.2f}%"
                    d_disp = f"{delta:+.2f}"
                elif "M2" in label or "Production" in label:
                    disp = f"{last:,.1f}"
                    d_disp = f"{delta:+,.1f}"
                else:
                    disp = f"{last:,.2f}"
                    d_disp = f"{delta:+.2f}"
                fcols[fi % 3].metric(label, disp, d_disp)
                fi += 1
            
            # Small charts for key series
            chart_labels = [l for l in ["CPI (All Urban Consumers)", "Fed Funds Effective Rate", 
                                         "Unemployment Rate", "M2 Money Supply"] if l in fred_data]
            if chart_labels:
                n = len(chart_labels)
                ccols = st.columns(min(2, n))
                for i, label in enumerate(chart_labels):
                    fdf = fred_data[label]
                    with ccols[i % 2]:
                        fig_f = go.Figure()
                        fig_f.add_trace(go.Scatter(
                            x=fdf.index, y=fdf["value"],
                            name=label, line=dict(width=2, color="#00BFFF")
                        ))
                        fig_f.update_layout(
                            title=label,
                            height=260,
                            template="plotly_dark",
                            paper_bgcolor="#0e1117",
                            plot_bgcolor="#0e1117",
                            margin=dict(l=10, r=10, t=40, b=10),
                            font=dict(family="IBM Plex Sans, sans-serif"),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_f, use_container_width=True)


if nav == "📰 LiveMacro/Crypto/StockNews":
    st.subheader("📰 Live Macro / Crypto / Stock News")
    st.caption("Fed · SEC · Stocks · Crypto headlines with sentiment. Briefing headlines are unchanged.")

    fng_v, fng_c = get_fear_greed()
    m1, m2, m3 = st.columns(3)
    m1.metric("Fear & Greed", f"{fng_v}/100", str(fng_c)[:20] if fng_c else "")
    m2.metric("Sources", "Fed · SEC · Stocks · Crypto")
    m3.caption("Cached ~15 min · free RSS (no API key)")

    try:
        all_news = get_market_news(
            cache_buster="news-v1",
            max_items=48,
        )
    except Exception:
        all_news = []

    try:
        crypto_only = get_crypto_news(
            cache_buster="news-v1",
            max_items=20,
        )
    except Exception:
        crypto_only = []

    sia = get_sia() or SIA()

    def _render_article_list(items, empty_msg="No articles right now."):
        if not items:
            st.info(empty_msg)
            return
        for art in items:
            title = art.get("title") or "Untitled"
            with st.expander(title):
                score = sia.polarity_scores(title)["compound"]
                if score >= 0.05:
                    status_badge = '<span style="color:#22c55e; font-weight:bold;">Bullish</span>'
                elif score <= -0.05:
                    status_badge = '<span style="color:#ef4444; font-weight:bold;">Bearish</span>'
                else:
                    status_badge = '<span style="color:#94a3b8; font-weight:bold;">Neutral</span>'
                src = art.get("source") or ""
                pub = art.get("published") or ""
                cat = art.get("category") or ""
                summary = art.get("summary") or ""
                link = art.get("link") or "#"
                st.markdown(
                    f"""
                    <div class="news-box">
                        <p class="news-meta">
                            <strong>{src}</strong>
                            {" · " + cat if cat else ""}
                            {" · " + pub if pub else ""}
                            · Sentiment: {status_badge} ({score:+.2f})
                        </p>
                        <p class="news-summary">{summary}</p>
                        <a href="{link}" target="_blank" class="news-action-link">Open Article ↗</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    tab_macro, tab_stocks, tab_crypto, tab_all = st.tabs(
        ["Macro / Policy", "Stocks", "Crypto", "All mixed"]
    )

    with tab_macro:
        st.markdown("##### Fed · SEC")
        macro_items = [a for a in (all_news or []) if a.get("category") in ("Fed", "SEC")]
        _render_article_list(macro_items, "No macro/policy headlines right now.")

    with tab_stocks:
        st.markdown("##### Live stock market news")
        st.caption("Yahoo · CNBC · MarketWatch")
        stock_items = [a for a in (all_news or []) if a.get("category") == "Stocks"]
        _render_article_list(stock_items, "No stock market headlines right now.")

    with tab_crypto:
        st.markdown("##### Crypto desk feed")
        _render_article_list(
            crypto_only or [a for a in (all_news or []) if a.get("category") == "Crypto"],
            "No crypto headlines right now.",
        )

    with tab_all:
        st.markdown("##### Combined feed")
        _render_article_list(all_news, "No headlines available.")

# ============================================================
# PREDICTIVE ANALYTICS TAB
# ============================================================
if nav == "🔮 Predictive":
    st.subheader("🔮 Multi-Factor Predictive Signal Engine")
    st.caption("Synthesizes Technicals • Futures Positioning • Sentiment • Macro into a single Bias Score. This is a decision-support tool, not financial advice.")
    
    if not tkrs:
        st.info("Enter at least one ticker in the sidebar to generate predictions.")
    else:
        pred_data = load_ticker_data(tuple(tkrs), interval, lookback_bars, period, is_6h)
        fng_v, fng_c = get_fear_greed()
        news = get_crypto_news(cache_buster="news-v1")
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
                    st.metric("Illustrative upside", f"${prediction['projected_upside']:,.2f}")
                    st.metric("Illustrative downside", f"${prediction['projected_downside']:,.2f}")
                
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
                    font=dict(family="IBM Plex Sans, sans-serif"),
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
if nav == "🛠️ Tools":
    st.subheader("🛠️ Backtesting • Multi-Timeframe • Export")
    
    tool_sub = st.radio(
        "Select tool:",
        ["📈 Backtesting", "⏱️ Multi-Timeframe View", "📥 Export Data"],
        horizontal=True
    )
    
    # ---------- BACKTESTING ----------
    if tool_sub == "📈 Backtesting":
        st.markdown("### Simple Strategy Backtester")
        st.caption("Educational tool only. Past performance ≠ future results. No fees/slippage modeled.")
        
        if not tkrs:
            st.info("Load at least one ticker in the sidebar first.")
        else:
            bt_ticker = st.selectbox("Ticker to backtest", tkrs, key="bt_ticker")
            strategy = st.selectbox("Strategy", ["RSI Mean Reversion (30/70)", "SMA 50/200 Crossover"], key="bt_strat")
            
            bt_data = load_ticker_data((bt_ticker,), "1d", 500, "2y", False)
            df = bt_data.get(bt_ticker)
            
            if df is None or df.empty:
                st.warning("Could not load enough history for backtesting.")
            else:
                if strategy.startswith("RSI"):
                    result = run_rsi_backtest(df)
                else:
                    result = run_sma_cross_backtest(df)
                
                if result is None:
                    st.warning("Not enough data points for this strategy (need longer history).")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Strategy Return", f"{result['total_return']:+.1f}%")
                    c2.metric("Buy & Hold", f"{result['buy_hold']:+.1f}%")
                    c3.metric("Max Drawdown", f"{result['max_drawdown']:.1f}%")
                    c4.metric("Approx Trades", result.get("approx_trades", "N/A"))
                    
                    if "win_rate" in result:
                        st.caption(f"Win rate (approx): {result['win_rate']:.0f}%")
                    
                    fig_eq = go.Figure()
                    fig_eq.add_trace(go.Scatter(
                        x=result["equity"].index,
                        y=result["equity"],
                        name="Strategy Equity",
                        line=dict(color="#00FF7F", width=2)
                    ))
                    fig_eq.update_layout(
                        title=f"Equity Curve — {bt_ticker} — {strategy}",
                        height=380,
                        template="plotly_dark",
                        paper_bgcolor="#0e1117",
                        plot_bgcolor="#0e1117",
                        font=dict(family="IBM Plex Sans, sans-serif"),
                        margin=dict(l=10, r=10, t=40, b=10),
                        yaxis_title="Growth of $1"
                    )
                    st.plotly_chart(fig_eq, use_container_width=True)
    
    # ---------- MULTI-TIMEFRAME ----------
    elif tool_sub == "⏱️ Multi-Timeframe View":
        st.markdown("### Multi-Timeframe Snapshot")
        st.caption("Quick view of the same asset across 15m • 1h • Daily • Weekly")
        
        if not tkrs:
            st.info("Load at least one ticker first.")
        else:
            mtf_ticker = st.selectbox("Select ticker", tkrs, key="mtf_ticker")
            with st.spinner(f"Loading multi-timeframe data for {mtf_ticker}..."):
                mtf = get_multi_tf_summary(mtf_ticker)
            
            if not mtf:
                st.warning("Could not load multi-timeframe data.")
            else:
                cols = st.columns(len(mtf))
                for i, (tf, vals) in enumerate(mtf.items()):
                    with cols[i]:
                        st.markdown(f"**{tf.upper()}**")
                        st.metric("Price", f"${vals['price']:,.2f}", f"{vals['change']:+.2f}%")
                        if vals.get("rsi") is not None:
                            st.write(f"RSI: {vals['rsi']:.1f}")
                        if vals.get("sma50") is not None:
                            st.write(f"SMA50: ${vals['sma50']:,.2f}")
    elif tool_sub == "📥 Export Data":
        st.markdown("### Export Data")
        st.caption("Download current data as CSV files.")
        
        if not tkrs:
            st.info("Load tickers first to enable exports.")
        else:
            export_data = load_ticker_data(tuple(tkrs), interval, lookback_bars, period, is_6h)
            
            # Performance export
            perf_df = compute_multi_period_performance(export_data)
            if not perf_df.empty:
                st.download_button(
                    "Download Performance CSV",
                    data=dataframe_to_csv_bytes(perf_df.set_index("Ticker")),
                    file_name="performance.csv",
                    mime="text/csv"
                )
            
            # Volatility export
            vol_df = compute_volatility_stats(export_data)
            if not vol_df.empty:
                st.download_button(
                    "Download Volatility CSV",
                    data=dataframe_to_csv_bytes(vol_df.set_index("Ticker")),
                    file_name="volatility.csv",
                    mime="text/csv"
                )
            st.success("Use the buttons above to download CSV files.")
            
            # Combined research pack (single CSV of key metrics)
            st.markdown("#### Research Pack")
            try:
                pack_rows = []
                for t, df in export_data.items():
                    if df is None or df.empty or "close" not in df.columns:
                        continue
                    last = df.iloc[-1]
                    pack_rows.append({
                        "Ticker": t,
                        "Price": float(last["close"]),
                        "RSI": float(last["rsi"]) if "rsi" in df.columns and pd.notna(last.get("rsi")) else None,
                        "ATR": float(last["atr"]) if "atr" in df.columns and pd.notna(last.get("atr")) else None,
                    })
                if pack_rows:
                    pack_df = pd.DataFrame(pack_rows)
                    st.download_button(
                        "Download Research Pack CSV",
                        data=dataframe_to_csv_bytes(pack_df.set_index("Ticker")),
                        file_name="research_pack.csv",
                        mime="text/csv",
                        key="dl_research_pack"
                    )
            except Exception:
                pass



st.sidebar.success(f"Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S')}")
