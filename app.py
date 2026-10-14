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

nltk.download('vader_lexicon', quiet=True)
st.cache_data.clear()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json"
}

st.set_page_config(page_title="Terminal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    section[data-testid="stSidebar"] { background-color: #1a1d24; }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown { color: #fafafa !important; }
    [data-testid="stMetricValue"] { color: #00FF7F !important; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Asset Intelligence Terminal")
st.caption("Dark mode • Real Long/Short + Open Interest + Funding Rate from Binance • RSI • MACD • Bollinger • On-chain")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Controls")
t_raw = st.sidebar.text_input("Enter Tickers (commas):", value="BTC-USD, ETH-USD, SOL-USD, XRP-USD")
tkrs = [t.strip().upper() for t in t_raw.split(",") if t.strip()]

res_disp = st.sidebar.selectbox("Resolution", ["15min", "1hr", "6hr", "Daily (1d)", "Weekly (1w)"])
rf_rate = st.sidebar.slider("Refresh Rate (Seconds)", 15, 120, 45)

st.sidebar.subheader("Indicators")
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

if res_disp == "15min":
    interval, lookback_bars, period = "15m", 96, "7d"
elif res_disp == "1hr":
    interval, lookback_bars, period = "1h", 72, "30d"
elif res_disp == "6hr":
    interval, lookback_bars, period = "1h", 84, "60d"
elif "Weekly" in res_disp:
    interval, lookback_bars, period = "1wk", 52, "2y"
else:
    interval, lookback_bars, period = "1d", 90, "2y"

st_autorefresh(interval=rf_rate * 1000, key="datarefresh")

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(series, window=20, num_std=2):
    sma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return sma, upper, lower

@st.cache_data(ttl=300)
def get_fear_greed():
    try:
        res = requests.get("https://api.alternative.me/fng/", headers=HEADERS, timeout=8).json()
        data = res["data"][0]
        return data["value"], data["value_classification"]
    except Exception:
        return "N/A", "Unavailable"

@st.cache_data(ttl=60, show_spinner="Fetching market data...")
def load_ticker_data(tickers: tuple, interval: str, lookback: int, period: str, is_6h: bool):
    results = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period=period, interval=interval, auto_adjust=True)
            if hist.empty:
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            hist.columns = [str(c).lower() for c in hist.columns]
            hist.index = hist.index.tz_localize(None)
            if is_6h:
                hist = hist.resample("6h").agg({
                    "open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"
                }).dropna()
            hist["sma50"] = hist["close"].rolling(50).mean()
            hist["sma200"] = hist["close"].rolling(200).mean()
            hist["rsi"] = compute_rsi(hist["close"])
            hist["macd"], hist["macd_signal"], hist["macd_hist"] = compute_macd(hist["close"])
            hist["bb_mid"], hist["bb_upper"], hist["bb_lower"] = compute_bollinger(hist["close"])
            display = hist.iloc[-lookback:] if len(hist) >= lookback else hist
            results[t] = display
        except Exception as e:
            st.warning(f"Could not load {t}: {e}")
    return results

@st.cache_data(ttl=300)
def get_long_term_avg(ticker: str):
    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        if hist.empty:
            return None
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        hist.columns = [str(c).lower() for c in hist.columns]
        return float(hist["close"].mean())
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_daily_pivots(tickers: tuple):
    pivots = {}
    for t in tickers:
        try:
            daily = yf.Ticker(t).history(period="10d", interval="1d", auto_adjust=True)
            if len(daily) < 2:
                continue
            if isinstance(daily.columns, pd.MultiIndex):
                daily.columns = daily.columns.get_level_values(0)
            daily.columns = [str(c).lower() for c in daily.columns]
            prev = daily.iloc[-2]
            high = float(prev["high"])
            low = float(prev["low"])
            close = float(prev["close"])
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            pivots[t] = {"Pivot": pivot, "R1": r1, "R2": r2, "S1": s1, "S2": s2}
        except Exception:
            continue
    return pivots

@st.cache_data(ttl=300)
def get_crypto_news(max_items=12):
    feeds = {
        "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "CoinTelegraph": "https://cointelegraph.com/rss",
        "Decrypt": "https://decrypt.co/feed"
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
                    "summary": summary
                })
        except Exception:
            continue
    return sorted(articles, key=lambda x: x["published"], reverse=True)[:max_items]

@st.cache_data(ttl=90)
def get_real_onchain(ticker: str):
    t = ticker.upper().replace("-USD", "").replace("USDT", "").replace("USD", "").strip()
    
    if t in ["BTC", "BITCOIN"]:
        try:
            stats = requests.get("https://api.blockchain.info/stats", headers=HEADERS, timeout=10).json()
            addr_resp = requests.get(
                "https://api.blockchain.info/charts/n-unique-addresses?timespan=5days&format=json",
                headers=HEADERS, timeout=10
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
                "Source": "Blockchain.com (free)"
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
            if symbol == t or (t == "XRP" and "xrpl" in name) or (t == "ETH" and name == "ethereum") or (t == "SOL" and name == "solana"):
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
                "Source": "DefiLlama (free)"
            }
    except Exception:
        pass

    try:
        slug_map = {"ONDO": "ondo-finance", "AAVE": "aave", "UNI": "uniswap", "LINK": "chainlink",
                    "LDO": "lido", "MKR": "makerdao", "CRV": "curve-dex", "PENDLE": "pendle"}
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
                "Source": "DefiLlama Protocol (free)"
            }
    except Exception:
        pass

    try:
        cg_ids = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "XRP": "ripple",
            "ADA": "cardano", "AVAX": "avalanche-2", "DOT": "polkadot", "DOGE": "dogecoin",
            "LINK": "chainlink", "ONDO": "ondo-finance", "NEAR": "near", "ATOM": "cosmos",
            "SUI": "sui", "APT": "aptos"
        }
        cg_id = cg_ids.get(t, t.lower())
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd&include_market_cap=true&include_24hr_vol=true"
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if cg_id in data:
                d = data[cg_id]
                return {
                    "Asset": t,
                    "Price (USD)": f"${d.get('usd', 0):,.4f}",
                    "Market Cap": f"${d.get('usd_market_cap', 0):,.0f}" if d.get("usd_market_cap") else "N/A",
                    "24h Volume": f"${d.get('usd_24h_vol', 0):,.0f}" if d.get("usd_24h_vol") else "N/A",
                    "Source": "CoinGecko (keyless free)"
                }
    except Exception as e:
        return {"Error": f"All free sources failed for {t}", "Details": str(e)[:120]}

    return {"Note": f"No free on-chain data found for {t}", "Source": "Free endpoints only"}

# ---------- Improved Binance Futures Metrics (Long/Short + OI + Funding) ----------
@st.cache_data(ttl=60)
def get_binance_futures_metrics(ticker: str):
    t = ticker.upper().replace("-USD", "").replace("USDT", "").replace("USD", "").strip()
    possible_symbols = [f"{t}USDT", f"{t}USD", t]
    
    result = {
        "source": "Simulated (no Binance data)",
        "long_pct": 50.0,
        "short_pct": 50.0,
        "oi": None,
        "funding_rate": None
    }
    
    for symbol in possible_symbols:
        try:
            # Long/Short Ratio
            for period in ["1h", "5m", "15m"]:
                url_ls = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period={period}&limit=1"
                r = requests.get(url_ls, headers=HEADERS, timeout=8)
                if r.status_code == 200:
                    data = r.json()
                    if isinstance(data, list) and len(data) > 0:
                        item = data[0]
                        long_account = float(item.get("longAccount", 0.5))
                        short_account = float(item.get("shortAccount", 0.5))
                        total = long_account + short_account
                        if total > 0:
                            result["long_pct"] = round((long_account / total) * 100, 1)
                            result["short_pct"] = round((short_account / total) * 100, 1)
                            result["source"] = f"Binance Global L/S ({period})"
                            break
            
            # Open Interest
            url_oi = f"https://fapi.binance.com/fapi/v1/openInterest?symbol={symbol}"
            r2 = requests.get(url_oi, headers=HEADERS, timeout=8)
            if r2.status_code == 200:
                oi_data = r2.json()
                oi_val = float(oi_data.get("openInterest", 0))
                if oi_val > 0:
                    result["oi"] = oi_val
            
            # Funding Rate
            url_fund = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={symbol}"
            r3 = requests.get(url_fund, headers=HEADERS, timeout=8)
            if r3.status_code == 200:
                fund_data = r3.json()
                fr = float(fund_data.get("lastFundingRate", 0))
                result["funding_rate"] = fr * 100  # convert to %
            
            if "Binance" in result["source"]:
                break
                
        except Exception:
            continue
    
    return result

# -----------------------------
# Load Data
# -----------------------------
if not tkrs:
    st.info("Enter at least one ticker.")
    st.stop()

is_6h = res_disp == "6hr"
data = load_ticker_data(tuple(tkrs), interval, lookback_bars, period, is_6h)

if not data:
    st.error("No valid data loaded.")
    st.stop()

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
chart_historical_store = data
futures_table_rows = []

for t, hist in data.items():
    s_px = float(hist["close"].iloc[0])
    combined_df[t] = ((hist["close"] / s_px) - 1) * 100

    c_px = float(hist["close"].iloc[-1])
    p_px = float(hist["close"].iloc[-2]) if len(hist) > 1 else c_px
    delta = ((c_px - p_px) / p_px) * 100

    # Real Futures Data
    fut = get_binance_futures_metrics(t)
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
        "oi": oi_val,
        "funding": funding,
        "source": source_label
    }

    # For the futures table
    futures_table_rows.append({
        "Ticker": t,
        "Long %": f"{long_r:.1f}%",
        "Short %": f"{short_r:.1f}%",
        "Open Interest": f"{oi_val:,.0f}" if oi_val else "N/A",
        "Funding Rate": f"{funding:+.4f}%" if funding is not None else "N/A",
        "Source": source_label
    })

    if show_onchain and ("-USD" in t or t.replace("-", "").isalpha()):
        onchain_profiles[t] = get_real_onchain(t)

    if t in daily_pivots:
        p = daily_pivots[t]
        matrix_rows.append({
            "Ticker": t, "Price": f"${c_px:,.2f}", "Change": f"{delta:+.2f}%",
            "Long %": f"{long_r:.1f}%", "Short %": f"{short_r:.1f}%",
            "Pivot": f"${p['Pivot']:,.2f}", "S1": f"${p['S1']:,.2f}", "S2": f"${p['S2']:,.2f}",
            "R1": f"${p['R1']:,.2f}", "R2": f"${p['R2']:,.2f}"
        })
    else:
        matrix_rows.append({
            "Ticker": t, "Price": f"${c_px:,.2f}", "Change": f"{delta:+.2f}%",
            "Long %": f"{long_r:.1f}%", "Short %": f"{short_r:.1f}%",
            "Pivot": "N/A", "S1": "N/A", "S2": "N/A", "R1": "N/A", "R2": "N/A"
        })

# -----------------------------
# Charts
# -----------------------------
st.subheader("📊 Comparative Relative Performance")
fig = go.Figure()
for col in combined_df.columns:
    fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df[col], name=col))
fig.update_layout(height=320, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔥 Asset Analysis Frameworks")
m_tgt = st.selectbox("Select Target Asset:", list(liquidity_profiles.keys()))

if m_tgt:
    h_data = chart_historical_store[m_tgt].copy()

    if show_baseline:
        h_data["baseline"] = h_data["close"].mean()
    if show_long_baseline:
        long_avg = get_long_term_avg(m_tgt)
        if long_avg is not None:
            h_data["long_baseline"] = long_avg
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

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=row_heights, subplot_titles=subplot_titles)
    current_row = 1

    fig.add_trace(go.Scatter(x=h_data.index, y=h_data["close"], name="Price", line=dict(color="#00FFFF", width=2)), row=current_row, col=1)

    if show_bb:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["bb_upper"], name="BB Upper", line=dict(color="rgba(255,255,255,0.3)", width=1), showlegend=False), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["bb_lower"], name="BB Lower", line=dict(color="rgba(255,255,255,0.3)", width=1), fill="tonexty", fillcolor="rgba(0,255,127,0.08)"), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["bb_mid"], name="BB Mid", line=dict(color="rgba(255,255,255,0.5)", width=1, dash="dot")), row=current_row, col=1)

    if show_sma50:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["sma50"], name="SMA 50", line=dict(color="#00FF7F", width=1.5, dash="dash")), row=current_row, col=1)
    if show_sma200:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["sma200"], name="SMA 200", line=dict(color="#FF4500", width=1.5)), row=current_row, col=1)
    if show_baseline and "baseline" in h_data.columns:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["baseline"], name="Hist. Baseline (Avg)", line=dict(color="#FFD700", width=1.8, dash="dot")), row=current_row, col=1)
    if show_long_baseline and "long_baseline" in h_data.columns:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["long_baseline"], name="1Y Baseline", line=dict(color="#FF8C00", width=1.8, dash="dashdot")), row=current_row, col=1)
    if show_vwap and "vwap" in h_data.columns:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["vwap"], name="VWAP", line=dict(color="#FF00FF", width=1.6)), row=current_row, col=1)

    current_row += 1

    if show_rsi:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["rsi"], name="RSI", line=dict(color="#FFD700", width=1.5)), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="lime", row=current_row, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
        current_row += 1

    if show_macd:
        colors = ["#00FF7F" if v >= 0 else "#FF4500" for v in h_data["macd_hist"]]
        fig.add_trace(go.Bar(x=h_data.index, y=h_data["macd_hist"], name="Histogram", marker_color=colors, showlegend=False), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["macd"], name="MACD", line=dict(color="#00BFFF", width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["macd_signal"], name="Signal", line=dict(color="#FF69B4", width=1.5)), row=current_row, col=1)
        current_row += 1

    if show_vol_profile and "volume" in h_data.columns:
        price_bins = pd.cut(h_data["close"], bins=24)
        vol_profile = h_data.groupby(price_bins, observed=True)["volume"].sum()
        bin_centers = [interval.mid for interval in vol_profile.index]
        fig.add_trace(go.Bar(x=vol_profile.values, y=bin_centers, orientation="h", name="Vol Profile", marker_color="rgba(0, 191, 255, 0.45)", showlegend=False), row=current_row, col=1)
        current_row += 1

    if show_vs_btc and "vs_btc" in h_data.columns:
        fig.add_trace(go.Scatter(x=h_data.index, y=h_data["vs_btc"], name="vs BTC %", line=dict(color="#00FFAA", width=1.8)), row=current_row, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=current_row, col=1)

    fig.update_layout(height=220 + (rows * 150), template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(5)
    if show_rsi and not h_data["rsi"].isna().all():
        rsi_val = h_data["rsi"].iloc[-1]
        cols[0].metric("RSI", f"{rsi_val:.1f}", "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral")
    if show_macd:
        cols[1].metric("MACD", f"{h_data['macd'].iloc[-1]:.2f}")
    if show_bb:
        bb_range = h_data["bb_upper"].iloc[-1] - h_data["bb_lower"].iloc[-1]
        if bb_range > 0:
            bb_pos = (h_data["close"].iloc[-1] - h_data["bb_lower"].iloc[-1]) / bb_range
            cols[2].metric("%B (BB)", f"{bb_pos*100:.1f}%")
    if show_baseline and "baseline" in h_data.columns:
        dist = ((h_data["close"].iloc[-1] / h_data["baseline"].iloc[-1]) - 1) * 100
        cols[3].metric("vs Baseline", f"{dist:+.1f}%")
    if show_vs_btc and "vs_btc" in h_data.columns:
        cols[4].metric("vs BTC", f"{h_data['vs_btc'].iloc[-1]:+.1f}%")

    # Liquidity Map
    st.markdown("#### 🔥 Liquidity Map (Estimated Liquidation Levels)")
    p = liquidity_profiles[m_tgt]
    st.caption(f"Source of Long/Short: **{p['source']}** | Liquidation levels are estimated")
    
    levels = [p["25x_l"], p["50x_l"], p["100x_l"], p["price"], p["100x_s"], p["50x_s"], p["25x_s"]]
    labels = ["25x Long Liq", "50x Long Liq", "100x Long Liq", "Current Price", "100x Short Liq", "50x Short Liq", "25x Short Liq"]
    fig_map = go.Figure(go.Bar(x=labels, y=levels, marker=dict(color=levels, colorscale="Portland")))
    fig_map.update_layout(height=320, template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig_map, use_container_width=True)

# Real-time Futures Table
st.subheader("📊 Real-Time Futures Metrics (Binance)")
st.caption("Long/Short Account Ratio • Open Interest • Funding Rate (updated ~every minute)")
st.dataframe(pd.DataFrame(futures_table_rows).set_index("Ticker"), use_container_width=True)

# Leverage Allocation Chart
st.subheader("📊 Long vs Short Allocation")
c_as = list(liquidity_profiles.keys())
fig_c = go.Figure()
fig_c.add_trace(go.Bar(
    x=c_as, 
    y=[liquidity_profiles[k]["longs"] for k in c_as], 
    name="Long %", 
    marker_color="#00FF7F",
    text=[f"{liquidity_profiles[k]['longs']:.1f}%" for k in c_as],
    textposition="auto"
))
fig_c.add_trace(go.Bar(
    x=c_as, 
    y=[liquidity_profiles[k]["shorts"] for k in c_as], 
    name="Short %", 
    marker_color="#FF4500",
    text=[f"{liquidity_profiles[k]['shorts']:.1f}%" for k in c_as],
    textposition="auto"
))
fig_c.update_layout(height=320, barmode="group", template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", yaxis_title="%")
st.plotly_chart(fig_c, use_container_width=True)

# On-chain
if show_onchain and onchain_profiles:
    st.subheader("🔗 Real On-chain Metrics")
    for ticker, metrics in onchain_profiles.items():
        st.markdown(f"**{ticker}**")
        st.dataframe(pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]), use_container_width=True, hide_index=True)

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
                st.markdown(f"**{art['source']}** • {art['published']}")
                if art["summary"]:
                    st.write(art["summary"])
                score = sia.polarity_scores(art["title"])["compound"]
                if score >= 0.05:
                    st.success(f"Bullish ({score:+.2f})")
                elif score <= -0.05:
                    st.error(f"Bearish ({score:+.2f})")
                else:
                    st.info(f"Neutral ({score:+.2f})")
                st.markdown(f"[Read →]({art['link']})")

with col2:
    st.subheader("📊 Sentiment")
    fng_v, fng_c = get_fear_greed()
    st.metric("Fear & Greed", f"{fng_v}/100", fng_c)

st.sidebar.success(f"Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S')}")
