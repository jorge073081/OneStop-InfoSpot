import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import feedparser

nltk.download('vader_lexicon', quiet=True)

# -----------------------------
# Dark Mode (forced)
# -----------------------------
st.set_page_config(
    page_title="Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force dark theme via CSS (works even if user has light browser preference)
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1d24;
    }
    /* Headers & text */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        color: #fafafa !important;
    }
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00FF7F !important;
    }
    /* Dataframes */
    .stDataFrame {
        background-color: #1a1d24;
    }
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1a1d24;
        color: #fafafa;
    }
    /* Inputs */
    .stTextInput > div > div > input,
    .stSelectbox > div > div,
    .stSlider {
        background-color: #262730;
        color: #fafafa;
    }
    /* Success / info boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        background-color: #1a1d24;
    }
    /* Plotly containers already dark */
</style>
""", unsafe_allow_html=True)

st.title("📈 Asset Intelligence Terminal")
st.caption("Dark mode forced • Real BTC on-chain data from Blockchain.com")

# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Controls")
t_raw = st.sidebar.text_input("Enter Tickers (commas):", value="BTC-USD, AAPL")
tkrs = [t.strip().upper() for t in t_raw.split(",") if t.strip()]

res_disp = st.sidebar.selectbox("Resolution", ["Weekly (1w)", "Daily (1d)"])
rf_rate = st.sidebar.slider("Refresh Rate (Seconds)", 15, 120, 45)

show_sma50 = st.sidebar.checkbox("Show 50-Period SMA", value=True)
show_sma200 = st.sidebar.checkbox("Show 200-Period SMA", value=True)

if "Weekly" in res_disp:
    interval, lookback_bars = "1wk", 52
else:
    interval, lookback_bars = "1d", 90

# Auto-refresh
st_autorefresh(interval=rf_rate * 1000, key="datarefresh")

# -----------------------------
# Helper Functions
# -----------------------------
@st.cache_data(ttl=300)
def get_fear_greed():
    try:
        res = requests.get("https://api.alternative.me/fng/", timeout=5).json()
        data = res["data"][0]
        return data["value"], data["value_classification"]
    except Exception:
        return "N/A", "Unavailable"


@st.cache_data(ttl=60, show_spinner="Fetching market data...")
def load_ticker_data(tickers: tuple, interval: str, lookback: int):
    results = {}
    for t in tickers:
        try:
            hist = yf.Ticker(t).history(period="2y", interval=interval, auto_adjust=True)
            if hist.empty:
                continue

            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            hist.columns = [str(c).lower() for c in hist.columns]
            hist.index = hist.index.tz_localize(None)

            hist["sma50"] = hist["close"].rolling(window=50).mean()
            hist["sma200"] = hist["close"].rolling(window=200).mean()

            display = hist.iloc[-lookback:] if len(hist) >= lookback else hist
            results[t] = display
        except Exception as e:
            st.warning(f"Could not load {t}: {e}")
    return results


@st.cache_data(ttl=300)
def get_crypto_news(max_items: int = 12):
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

    articles = sorted(articles, key=lambda x: x["published"], reverse=True)
    return articles[:max_items]


@st.cache_data(ttl=180)
def get_real_onchain(ticker: str):
    """
    Real on-chain data (keyless).
    Currently full support for BTC / BTC-USD via Blockchain.com.
    Other assets return a clear message.
    """
    t = ticker.upper().replace("-USD", "").replace("USDT", "")
    
    if t in ["BTC", "BITCOIN"]:
        try:
            # Main stats
            stats = requests.get("https://api.blockchain.info/stats", timeout=8).json()
            
            # Latest unique addresses (last few days)
            addr_resp = requests.get(
                "https://api.blockchain.info/charts/n-unique-addresses?timespan=5days&format=json",
                timeout=8
            ).json()
            latest_addr = int(addr_resp["values"][-1]["y"]) if addr_resp.get("values") else "N/A"
            
            # Hash rate is in GH/s in the API → convert to EH/s for readability
            hash_rate_ehs = stats.get("hash_rate", 0) / 1e9   # approx to EH/s
            
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
                "Source": "Blockchain.com (live)"
            }
        except Exception as e:
            return {"Error": f"Failed to fetch BTC on-chain: {e}"}
    
    # For other cryptos we currently don't have reliable free keyless sources
    return {
        "Note": f"Real on-chain metrics for {ticker} require a paid or keyed API (Glassnode, CryptoQuant, Etherscan, etc.).",
        "Status": "Only Bitcoin has full free keyless support in this terminal right now."
    }


# -----------------------------
# Load Data
# -----------------------------
if not tkrs:
    st.info("Enter at least one ticker in the sidebar.")
    st.stop()

data = load_ticker_data(tuple(tkrs), interval, lookback_bars)

if not data:
    st.error("No valid data could be loaded for the provided tickers.")
    st.stop()

# -----------------------------
# Build derived metrics
# -----------------------------
combined_df = pd.DataFrame()
matrix_rows = []
liquidity_profiles = {}
onchain_profiles = {}
chart_historical_store = data

for t, hist in data.items():
    display_hist = hist
    s_px = float(display_hist["close"].iloc[0])
    combined_df[t] = ((display_hist["close"] / s_px) - 1) * 100

    c_px = float(display_hist["close"].iloc[-1])
    p_px = float(display_hist["close"].iloc[-2]) if len(display_hist) > 1 else c_px
    delta = ((c_px - p_px) / p_px) * 100

    v_now = float(display_hist["volume"].iloc[-1]) if "volume" in display_hist.columns else 1.0
    v_avg = float(display_hist["volume"].mean()) if "volume" in display_hist.columns else 1.0
    intensity = v_now / v_avg if v_avg > 0 else 1.0

    long_r = min(75.0, 50.0 + (intensity * 3)) if delta > 0 else 100.0 - min(75.0, 50.0 + (intensity * 3))
    short_r = 100.0 - long_r

    liquidity_profiles[t] = {
        "price": c_px,
        "100x_l": c_px * 0.99, "50x_l": c_px * 0.98, "25x_l": c_px * 0.96,
        "100x_s": c_px * 1.01, "50x_s": c_px * 1.02, "25x_s": c_px * 1.04,
        "longs": long_r, "shorts": short_r
    }

    # Real on-chain (only for crypto tickers)
    is_crypto = "-USD" in t or t in ["BTC", "ETH", "SOL", "ADA", "XRP", "BITCOIN", "ETHEREUM"]
    if is_crypto:
        onchain_profiles[t] = get_real_onchain(t)

    l_sess = display_hist.iloc[-2] if len(display_hist) > 1 else display_hist.iloc[-1]
    piv = (float(l_sess["high"]) + float(l_sess["low"]) + float(l_sess["close"])) / 3
    r1 = (2 * piv) - float(l_sess["low"])
    s1 = (2 * piv) - float(l_sess["high"])

    matrix_rows.append({
        "Ticker": t,
        "Price": f"${c_px:,.2f}",
        "Change": f"{delta:+.2f}%",
        "Long %": f"{long_r:.1f}%",
        "Short %": f"{short_r:.1f}%",
        "Pivot": f"${piv:,.2f}",
        "S1": f"${s1:,.2f}",
        "R1": f"${r1:,.2f}"
    })

# -----------------------------
# Charts & Tables
# -----------------------------
st.subheader("📊 Comparative Relative Performance (% Returns)")
fig = go.Figure()
for col in combined_df.columns:
    fig.add_trace(go.Scatter(x=combined_df.index, y=combined_df[col], name=col))
fig.update_layout(height=350, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10),
                  paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔥 Asset Analysis Frameworks")
m_tgt = st.selectbox("Select Target Asset View:", list(liquidity_profiles.keys()))

if m_tgt:
    st.markdown(f"#### 📈 {m_tgt} Price Chart + SMAs")
    h_data = chart_historical_store[m_tgt]

    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(
        x=h_data.index, y=h_data["close"],
        name="Spot Price", line=dict(color="#00FFFF", width=2)
    ))

    if show_sma50:
        fig_sma.add_trace(go.Scatter(
            x=h_data.index, y=h_data["sma50"],
            name="SMA 50", line=dict(color="#00FF7F", width=1.5, dash="dash")
        ))
    if show_sma200:
        fig_sma.add_trace(go.Scatter(
            x=h_data.index, y=h_data["sma200"],
            name="SMA 200", line=dict(color="#FF4500", width=1.5)
        ))

    fig_sma.update_layout(
        height=380, template="plotly_dark",
        yaxis_title="Price ($)", margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117"
    )
    st.plotly_chart(fig_sma, use_container_width=True)

    # Simulated liquidity map
    st.markdown("#### 🔥 Simulated Liquidity Price Point Map")
    st.caption("⚠️ These levels are purely illustrative (arbitrary % offsets). They are **not** real liquidation data.")
    p = liquidity_profiles[m_tgt]
    levels = [p["25x_l"], p["50x_l"], p["100x_l"], p["price"], p["100x_s"], p["50x_s"], p["25x_s"]]
    labels = ["25x Long", "50x Long", "100x Long", "Spot", "100x Short", "50x Short", "25x Short"]

    fig_map = go.Figure()
    fig_map.add_trace(go.Bar(x=labels, y=levels, marker=dict(color=levels, colorscale="Portland")))
    fig_map.update_layout(height=320, template="plotly_dark", title=f"{m_tgt} Simulated Levels",
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
    st.plotly_chart(fig_map, use_container_width=True)

st.subheader("📊 Simulated Leverage Allocation")
st.caption("⚠️ Long/Short percentages are derived only from recent price change + volume intensity. Not real positioning data.")
c_as = list(liquidity_profiles.keys())
fig_c = go.Figure()
fig_c.add_trace(go.Bar(x=c_as, y=[liquidity_profiles[k]["longs"] for k in c_as], name="Long", marker_color="green"))
fig_c.add_trace(go.Bar(x=c_as, y=[liquidity_profiles[k]["shorts"] for k in c_as], name="Short", marker_color="red"))
fig_c.update_layout(height=300, barmode="group", template="plotly_dark",
                    paper_bgcolor="#0e1117", plot_bgcolor="#0e1117")
st.plotly_chart(fig_c, use_container_width=True)

# Real On-Chain section
if onchain_profiles:
    st.subheader("🔗 Real On-Chain Metrics")
    st.caption("✅ Bitcoin data is live from Blockchain.com (keyless). Other assets currently require paid APIs.")
    
    for ticker, metrics in onchain_profiles.items():
        st.markdown(f"**{ticker}**")
        # Convert to nice dataframe
        df_on = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        st.dataframe(df_on, use_container_width=True, hide_index=True)

if matrix_rows:
    st.subheader("🧮 Side-by-Side Pivot Matrix (Support & Resistance)")
    st.dataframe(pd.DataFrame(matrix_rows).set_index("Ticker"), use_container_width=True)

# -----------------------------
# News & Sentiment
# -----------------------------
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📰 Live Crypto News Feed")
    news = get_crypto_news(12)

    if not news:
        st.warning("Could not load news feeds at the moment.")
    else:
        sia = SIA()
        for art in news:
            with st.expander(f"{art['title']}"):
                st.markdown(f"**Source:** {art['source']}  \n**Published:** {art['published']}")
                if art["summary"]:
                    st.write(art["summary"])
                
                score = sia.polarity_scores(art["title"])["compound"]
                if score >= 0.05:
                    st.success(f"🟢 Bullish leaning ({score:+.2f})")
                elif score <= -0.05:
                    st.error(f"🔴 Bearish leaning ({score:+.2f})")
                else:
                    st.info(f"⚪ Neutral ({score:+.2f})")
                
                st.markdown(f"[Read full article →]({art['link']})")

with col2:
    st.subheader("📊 Sentiment Engine")
    fng_v, fng_c = get_fear_greed()
    st.metric(label="Fear & Greed Index", value=f"{fng_v}/100", delta=fng_c)

    st.markdown("#### Market Context")
    st.info("Fear & Greed is pulled live from alternative.me. News sentiment is calculated with VADER on real headlines.")

st.sidebar.success(f"Last refresh: {pd.Timestamp.now().strftime('%H:%M:%S')}")
