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
from typing import Dict, List, Optional, Tuple, Any

# ============================================================
# 1. INITIAL SETUP
# ============================================================
nltk.download("vader_lexicon", quiet=True)

st.set_page_config(
    page_title="OneStop-InfoSpot | Asset Intelligence Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme + Font Styles (Safely excluding all system icons & ligatures)
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
    <style>
        /* Target exact textual blocks to prevent breaking Material Icons/Ligatures */
        html, body, .stApp, p, label, .stMarkdown p, .stCaption, 
        h1, h2, h3, h4, h5, h6,
        .stTextInput input, .stSelectbox div, .stSlider div, .stCheckbox label,
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"],
        .stDataFrame, table, th, td {
            font-family: 'Playfair Display', serif !important;
        }

        /* HARD OVERRIDE: Restore Streamlit's icon ligatures and graphic symbols */
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

        /* Reduce top padding */
        .block-container {
            padding-top: 1.2rem !important;
        }

        /* Main Title - Playfair Display Font */
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

        /* Subtitle - Playfair Display Font */
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

        /* Clean CSS framework for news items to remove overlapping buttons */
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
            transition: all 0.2s ease-index;
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
    "Accept": "application/json",
}

# ============================================================
# 2. CONFIGURATION & HELPERS
# ============================================================
RESOLUTION_MAP = {
    "15min": {"interval": "15m", "lookback": 96, "period": "7d", "is_6h": False},
    "1hr": {"interval": "1h", "lookback": 72, "period": "30d", "is_6h": False},
    "6hr": {"interval": "1h", "lookback": 84, "period": "60d", "is_6h": True},
    "Daily (1d)": {"interval": "1d", "lookback": 90, "period": "2y", "is_6h": False},
    "Weekly (1w)": {"interval": "1wk", "lookback": 52, "period": "2y", "is_6h": False},
}


def safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


# ============================================================
# 3. TECHNICAL INDICATORS
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


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "close" not in df.columns:
        return df
    df = df.copy()
    df["sma50"] = df["close"].rolling(50).mean()
    df["sma200"] = df["close"].rolling(200).mean()
    df["rsi"] = compute_rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = compute_macd(df["close"])
    df["bb_mid"], df["bb_upper"], df["bb_lower"] = compute_bollinger(df["close"])
    return df


# ============================================================
# 4. DATA LOADING
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
            hist = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
            if hist.empty:
                continue
            hist = normalize_columns(hist)

            if is_6h:
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

            hist = add_indicators(hist)
            display = hist.iloc[-lookback:] if len(hist) >= lookback else hist
            results[t] = display
        except Exception as e:
            st.warning(f"Could not load {t}: {e}")
    return results


@st.cache_data(ttl=300)
def get_long_term_avg(ticker: str) -> Optional[float]:
    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        hist = normalize_columns(hist)
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
            daily = normalize_columns(daily)
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
        data = res["data"][0]
        return data["value"], data["value_classification"]
    except Exception:
        return "N/A", "Unavailable"


@st.cache_data(ttl=300)
def get_crypto_news(max_items: int = 12) -> List[Dict]:
    feeds = {
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "Decrypt": "https://decrypt.co/feed",
    }
    articles = []
    for source, url in feeds.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:
                published = entry.get("published", entry.get("updated", "Unknown"))
                summary = entry.get("summary", "")
                if summary:
                    summary = summary.replace("<p>", "").replace("</p>", "")[:260] + "..."
                articles.append({
                    "title": entry.get("title", "No title"),
                    "link": entry.get("link", "#"),
                    "source": source,
                    "published": published,
                    "summary": summary,
                })
        except Exception:
            continue
    return sorted(articles, key=lambda x: x["published"], reverse=True)[:max_items]


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
            if t.lower() in name:
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

    try:
        slug_map = {
            "ONDO": "ondo-finance", "AAVE": "aave", "UNI": "uniswap", "LINK": "chainlink",
            "LDO": "lido", "MKR": "makerdao", "CRV": "curve-dex", "PENDLE": "pendle",
        }
        slug = slug_map.get(t, t.lower())
        r = requests.get(f"https://api.llama.fi/tvl/{slug}", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            val = r.json()
            if isinstance(val, (int, float)) and val > 0:
                return {"Protocol": t, "TVL": f"${val:,.0f}", "Source": "DefiLlama Protocol (free)"}
        r2 = requests.get(f"https://api.llama.fi/protocol/{slug}", headers=HEADERS, timeout=10)
        if r2.status_code == 200:
            data = r2.json()
            return {
                "Protocol": data.get("name", t),
                "Category": data.get("category", "N/A"),
                "Symbol": data.get("symbol", t),
                "Market Cap": f"${data.get('mcap', 0):,.0f}" if data.get("mcap") else "N/A",
                "Source": "DefiLlama Protocol (free)",
            }
    except Exception:
        pass

    try:
        slug_map = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple",
            "ADA": "cardano", "AVAX": "avalanche-2", "DOT": "polkadot", "DOGE": "dogecoin",
            "LINK": "chainlink", "ONDO": "ondo-finance", "NEAR": "near", "ATOM": "cosmos",
            "SUI": "sui", "APT": "aptos",
        }
        slug = slug_map.get(t, t.lower())
        url = (
            f"https://api.coingecko.com/api/v3/simple/price"
            f"?ids={slug}&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true"
        )
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if slug in data:
                d = data[slug]
                return {
                    "Asset": t,
                    "Price (USD)": f"${d.get('usd', 0):,.4f}",
                    "Market Cap": f"${d.get('usd_market_cap', 0):,.0f}" if d.get('usd_market_cap') else "N/A",
                    "24h Volume": f"${d.get('usd_24h_vol', 0):,.0f}" if d.get('usd_24h_vol') else "N/A",
                    "Source": "CoinGecko (keyless free)",
                }
    except Exception as e:
        return {"Error": f"All free sources failed for {t}", "Details": str(e)[:120]}

    return {"Note": f"No free on-chain data found for {t}", "Source": "Free endpoints only"}


@st.cache_data(ttl=60)
def get_futures_metrics(ticker: str) -> Dict[str, Any]:
    t = ticker.upper().replace("-USD", "").replace("USDT", "").replace("USD", "").strip()
    symbol_binance = f"{t}USDT"
    symbol_okx = f"{t}-USDT-SWAP"

    result = {
        "source": "Simulated (no exchange data)",
        "long_pct": 50.0,
        "short_pct": 50.0,
        "oi": None,
        "funding_rate": None,
    }

    try:
        r = requests.get(
            f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol_binance}&period=1h&limit=1",
            headers=HEADERS, timeout=8,
        )
        if r.status_code == 200 and r.json():
            data = r.json()
            long_acc = safe_float(data.get("longAccount"), 0.5)
            short_acc = safe_float(data.get("shortAccount"), 0.5)
            total = long_acc + short_acc
            if total > 0:
                result["long_pct"] = round(long_acc / total * 100, 1)
                result["short_pct"] = round(short_acc / total * 100, 1)
                result["source"] = "Binance"

        r2 = requests.get(
            f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol_binance}",
            headers=HEADERS, timeout=8,
        )
        if r2.status_code == 200:
            oi = safe_float(r2.json().get("openInterest"))
            if oi > 0:
                result["oi"] = oi

        r3 = requests.get(
            f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol_binance}",
            headers=HEADERS, timeout=8,
        )
        if r3.status_code == 200:
            fr = safe_float(r3.json().get("lastFundingRate"))
            result["funding_rate"] = fr * 100

        if result["source"] == "Binance":
            return result
    except Exception:
        pass

    try:
        r = requests.get(
            f"https://api.bybit.com/v5/market/account-ratio?category=linear&symbol={symbol_binance}&period=1h&limit=1",
            headers=HEADERS, timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                item = data["result"]["list"]
                buy_ratio = safe_float(item.get("buyRatio"), 0.5)
                sell_ratio = safe_float(item.get("sellRatio"), 0.5)
                total = buy_ratio + sell_ratio
                if total > 0:
                    result["long_pct"] = round(buy_ratio / total * 100, 1)
                    result["short_pct"] = round(sell_ratio / total * 100, 1)
                    result["source"] = "Bybit"

        r2 = requests.get(
            f"https://api.bybit.com/v5/market/open-interest?category=linear&symbol={symbol_binance}&intervalTime=1h&limit=1",
            headers=HEADERS, timeout=8,
        )
        if r2.status_code == 200:
            data = r2.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                oi = safe_float(data["result"]["list"].get("openInterest"))
                if oi > 0:
                    result["oi"] = oi

        r3 = requests.get(
            f"https://api.bybit.com/v5/market/funding/history?category=linear&symbol={symbol_binance}&limit=1",
            headers=HEADERS, timeout=8,
        )
        if r3.status_code == 200:
            data = r3.json()
            if data.get("retCode") == 0 and data.get("result", {}).get("list"):
                fr = safe_float(data["result"]["list"].get("fundingRate"))
                result["funding_rate"] = fr * 100

        if result["source"] == "Bybit":
            return result
    except Exception:
        pass

    try:
        r = requests.get(
            f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy={t}&period=1H",
            headers=HEADERS, timeout=8,
        )
        if r.status_code == 200:
            data = r.json()
            if data.get("code") == "0" and data.get("data"):
                item = data["data"]
                ratio = safe_float(item if len(item) > 1 else 1.0)
                long_pct = round(ratio / (1 + ratio) * 100, 1)
                result["long_pct"] = long_pct
                result["short_pct"] = round(100 - long_pct, 1)
                result["source"] = "OKX"

        r2 = requests.get(
            f"https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId={symbol_okx}",
            headers=HEADERS, timeout=8,
        )
        if r2.status_code == 200:
            data = r2.json()
            if data.get("code") == "0" and data.get("data"):
                oi = safe_float(data["data"].get("oi"))
                if oi > 0:
                    result["oi"] = oi

        r3 = requests.get(
            f"https://www.okx.com/api/v5/public/funding-rate?instId={symbol_okx}",
            headers=HEADERS, timeout=8,
        )
        if r3.status_code == 200:
            data = r3.json()
            if data.get("code") == "0" and data.get("data"):
                fr = safe_float(data["data"].get("fundingRate"))
                result["funding_rate"] = fr * 100

        if result["source"] == "OKX":
            return result
    except Exception:
        pass

    return result


# ============================================================
# NEW COMPONENT: DATA PROCESSING LAYER FOR GLOBAL MACRO
# ============================================================
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
                macro_results[label] = normalize_columns(m_hist)
        except Exception:
            continue
    return macro_results


# ============================================================
# 5. CHART BUILDING
# ============================================================
def build_asset_chart(
    h_data: pd.DataFrame,
    show_candles: bool,
    show_sma50: bool,
    show_sma200: bool,
    show_bb: bool,
    show_rsi: bool,
    show_macd: bool,
    show_vol_profile: bool,
    show_baseline: bool,
    show_long_baseline: bool,
    show_vwap: bool,
    show_vs_btc: bool,
    btc_data: Optional[pd.DataFrame] = None,
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
    if show_vol_profile: rows += 1
    if show_vs_btc and "vs_btc" in h_data.columns: rows += 1

    row_heights = [0.50] + [0.125] * (rows - 1) if rows > 1 else [1.0]
    subplot_titles = ["Price"]
    if show_rsi: subplot_titles.append("RSI (14)")
    if show_macd: subplot_titles.append("MACD")
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

    current_row += 1

    if show_rsi and "rsi" in h_data.columns:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["rsi"], name="RSI",
                                 line=dict(color="#FFD700", width=1.5)), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="lime", row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1

    if show_macd and "macd_hist" in h_data.columns:
        colors = ["#00FF7F" if v >= 0 else "#FF4500" for v in h_data["macd_hist"]]
        fig.add_trace(go.Bar(x=h_data.index, y=h_data["macd_hist"], name="Histogram",
                             marker_color=colors, showlegend=False), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["macd"], name="MACD",
                                 line=dict(color="#00BFFF", width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["macd_signal"], name="Signal",
                                 line=dict(color="#FF69B4", width=1.5)), row=current_row, col=1)
        current_row += 1

    if show_vol_profile and "volume" in h_data.columns:
        try:
            price_bins = pd.cut(h_data["close"], bins=24)
            vol_profile = h_data.groupby(price_bins, observed=True)["volume"].sum()
            bin_centers = [interval.mid for interval in vol_profile.index]
            fig.add_trace(go.Bar(x=vol_profile.values, y=bin_centers, orientation="h",
                                 name="Vol Profile", marker_color="rgba(0, 191, 255, 0.45)", showlegend=False),
                          row=current_row, col=1)
        except Exception:
            pass
        current_row += 1

    if show_vs_btc and "vs_btc" in h_data.columns:
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
t_raw = st.sidebar.text_input("Enter Tickers (commas):", value="BTC-USD, ETH-USD, SOL-USD, XRP-USD")
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
show_vol_profile = st.sidebar.checkbox("Volume Profile", value=True)

st.sidebar.subheader("Overlays & On-chain")
show_baseline = st.sidebar.checkbox("Historical Baseline (Period Avg)", value=True)
show_long_baseline = st.sidebar.checkbox("Long-term Baseline (1Y Avg)", value=False)
show_vwap = st.sidebar.checkbox("VWAP", value=False)
show_vs_btc = st.sidebar.checkbox("Relative Strength vs BTC", value=False)
show_onchain = st.sidebar.checkbox("Show On-chain Metrics", value=True)

res_cfg = RESOLUTION_MAP[res_disp]
interval = res_cfg["interval"]
lookback_bars = res_cfg["lookback"]
period = res_cfg["period"]
is_6h = res_cfg["is_6h"]

st_autorefresh(interval=rf_rate * 1000, key="datarefresh")

# ============================================================
# 7. MAIN APP ENTERPRISE WORKSPACE
# ============================================================

st.markdown('<div class="silver-title">OneStop-InfoSpot</div>', unsafe_allow_html=True)
st.markdown('<div class="silver-subtitle">📈 Asset Intelligence Terminal</div>', unsafe_allow_html=True)

st.caption("Dark mode • Candlesticks • Long/Short + OI + Funding • RSI • MACD • Bollinger • On-chain • Global Macro")

# Initializing Native UI Workspace Decoupling Tabs
tab_crypto, tab_macro = st.tabs(["⚡ Crypto Intelligence Terminal", "🌐 Global Macro Insights"])

# ------------------------------------------------------------
# TAB 1: CRYPTO INTELLIGENCE WORKSPACE (YOUR ORIGINAL LOGIC)
# ------------------------------------------------------------
with tab_crypto:
    if not tkrs:
        st.info("Enter at least one ticker in the sidebar.")
    else:
        data = load_ticker_data(tuple(tkrs), interval, lookback_bars, period, is_6h)

        if not data:
            st.error("No valid data loaded. Check tickers or try a different resolution.")
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

                fut = get_futures_metrics(t)
                long_r = fut["long_pct"]
                short_r = fut["short_pct"]
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

            st.subheader("🔥 Asset Analysis Frameworks")
            m_tgt = st.selectbox("Select Target Asset:", list(liquidity_profiles.keys()))

            if m_tgt and m_tgt in data:
                h_data = data[m_tgt].copy()

                if show_long_baseline:
                    long_avg = get_long_term_avg(m_tgt)
                    if long_avg is not None:
                        h_data["long_baseline"] = long_avg

                fig = build_asset_chart(
                    h_data=h_data,
                    show_candles=show_candles,
                    show_sma50=show_sma50,
                    show_sma200=show_sma200,
                    show_bb=show_bb,
                    show_rsi=show_rsi,
                    show_macd=show_macd,
                    show_vol_profile=show_vol_profile,
                    show_baseline=show_baseline,
                    show_long_baseline=show_long_baseline,
                    show_vwap=show_vwap,
                    show_vs_btc=show_vs_btc,
                    btc_data=btc_data,
                )

                if show_long_baseline and "long_baseline" in h_data.columns:
                    fig.add_trace(
                        go.Scatter(x=h_data.index, y=h_data["long_baseline"], name="1Y Baseline",
                                   line=dict(color="#FF8C00", width=1.8, dash="dashdot")),
                        row=1, col=1,
                    )

                st.plotly_chart(fig, use_container_width=True)

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
                st.caption(f"Source of Long/Short: **{p['source']}** | Liquidation levels are estimated")

                levels = [p["25x_l"], p["50x_l"], p["100x_l"], p["price"], p["100x_s"], p["50x_s"], p["25x_s"]]
                labels = ["25x Long Liq", "50x Long Liq", "100x Long Liq", "Current Price",
                          "100x Short Liq", "50x Short Liq", "25x Short Liq"]
                fig_map = go.Figure(go.Bar(x=labels, y=levels, marker=dict(color=levels, colorscale="Portland")))
                fig_map.update_layout(height=320, template="plotly_dark",
                                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                      font=dict(family="Playfair Display, serif"))
                st.plotly_chart(fig_map, use_container_width=True)

            st.subheader("📊 Real-Time Futures Metrics")
            st.caption("Long/Short Account Ratio • Open Interest • Funding Rate")
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
                st.caption("Classic Floor Pivots calculated from the previous daily candle")
                st.dataframe(pd.DataFrame(matrix_rows).set_index("Ticker"), use_container_width=True)

            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📰 Live Crypto News")
                news = get_crypto_news()
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

            with col2:
                st.subheader("📊 Sentiment")
                fng_v, fng_c = get_fear_greed()
                st.metric("Fear & Greed", f"{fng_v}/100", fng_c)

# ------------------------------------------------------------
# TAB 2: GLOBAL MACRO INSIGHTS WORKSPACE (NEW COMPONENT)
# ------------------------------------------------------------
with tab_macro:
    st.subheader("🌐 Broad Market Telemetry Framework")
    st.caption("Asynchronous tracking layer pulling multi-asset global correlations over a rolling 90-day window.")
    
    macro_data = load_macro_data()
    
    if not macro_data:
        st.info("Macro matrix data endpoints temporarily unavailable. Retrying query package updates...")
    else:
        # 1. Macro Summary Cards Grid
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

        # 2. Normalized Comparative Performance Visualizer
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

st.sidebar.success(f"Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S')}")
