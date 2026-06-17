import streamlit as st
import pandas as pd
import numpy as np
import feedparser
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib import font_manager
from transformers import pipeline

st.set_page_config(page_title="Market Sentiment Monitor", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.5rem; max-width: 1100px;}
html, body, [class*="css"] {font-family: 'Inter', -apple-system, sans-serif;}
.app-header {border-bottom: 1px solid #e6e8eb; padding-bottom: 1rem; margin-bottom: 1.5rem;}
.app-header h1 {font-size: 1.9rem; font-weight: 700; color: #0f172a; margin: 0;}
.app-header p {color: #64748b; font-size: 0.95rem; margin: 0.4rem 0 0 0;}
.verdict {background: #f8fafc; border-left: 4px solid #1e3a8a; border-radius: 6px;
          padding: 1.1rem 1.3rem; margin: 0.5rem 0 1.5rem 0;}
.verdict h3 {color: #0f172a; font-size: 1.25rem; font-weight: 600; margin: 0 0 0.4rem 0;}
.verdict p {color: #475569; font-size: 0.97rem; margin: 0;}
div[data-testid="stMetric"] {background: #ffffff; border: 1px solid #e6e8eb;
          border-radius: 8px; padding: 1rem 1.2rem;}
div[data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 700;}
.stButton button {background: #1e3a8a; color: white; border: none; border-radius: 6px;
          padding: 0.5rem 1.6rem; font-weight: 600;}
.stButton button:hover {background: #1e40af; color: white;}
.read-hint {color: #64748b; font-size: 0.9rem; margin: 1.2rem 0 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

tickers = {
    "AAPL": "Apple", "TSLA": "Tesla", "NVDA": "Nvidia", "MSFT": "Microsoft",
    "AMZN": "Amazon", "GOOGL": "Google", "META": "Meta", "NFLX": "Netflix",
    "AMD": "AMD", "JPM": "JPMorgan"
}

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="ProsusAI/finbert")

@st.cache_data(ttl=1800)
def get_news(name):
    url = f"https://news.google.com/rss/search?q={name}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    return pd.DataFrame([{"headline": e.title, "published": e.published} for e in feed.entries])

@st.cache_data(ttl=1800)
def get_prices(symbol):
    data = yf.download(symbol, period="2mo", progress=False)["Close"]
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()
    return data

def analyze(symbol):
    name = tickers[symbol]
    model = load_model()
    news = get_news(name)
    if news.empty:
        return None, None, None
    news = news.head(120)
    scored = model(news["headline"].tolist(), truncation=True)
    news["sentiment"] = [r["label"] for r in scored]
    news["published"] = pd.to_datetime(news["published"], errors="coerce", utc=True)
    news = news.dropna(subset=["published"]).set_index("published").sort_index()

    daily_net = news["sentiment"].resample("D").apply(
        lambda x: (x == "positive").sum() - (x == "negative").sum()
    )
    daily_count = news["sentiment"].resample("D").count()
    net_ratio = daily_net / daily_count.where(daily_count >= 2, np.nan)
    net_ratio = net_ratio.rolling(7, min_periods=2).mean().ffill()

    prices = get_prices(symbol)
    return news, net_ratio, prices

def build_verdict(name, news, net_ratio, prices):
    counts = news["sentiment"].value_counts()
    pos = int(counts.get("positive", 0)); neg = int(counts.get("negative", 0)); neu = int(counts.get("neutral", 0))
    total = max(pos + neg + neu, 1)
    pos_pct = pos / total * 100

    if pos > neg * 1.3:
        mood = "leaning positive"
    elif neg > pos * 1.3:
        mood = "leaning negative"
    else:
        mood = "mixed"

    headline = f"News on {name} is {mood} ({pos_pct:.0f}% of recent headlines are positive)"

    direction = ""
    if len(net_ratio.dropna()) >= 5:
        recent = net_ratio.dropna().iloc[-5:]
        if recent.iloc[-1] < recent.iloc[0] - 0.05:
            direction = "Coverage has cooled over recent days"
        elif recent.iloc[-1] > recent.iloc[0] + 0.05:
            direction = "Coverage has warmed over recent days"
        else:
            direction = "Coverage has held broadly steady"

    price_note = ""
    if len(prices.dropna()) >= 5:
        p = prices.dropna()
        change = float((p.iloc[-1] - p.iloc[-5]) / p.iloc[-5] * 100)
        arrow = "up" if change > 0 else "down"
        price_note = f", while the share price moved {arrow} {abs(change):.1f}% over the same window"

    note = f"{direction}{price_note}." if direction else ""
    return headline, note

def styled_chart(name, symbol, prices, net_ratio):
    plt.rcParams.update({
        "font.size": 11, "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.8,
        "text.color": "#334155", "axes.labelcolor": "#334155",
        "xtick.color": "#64748b", "ytick.color": "#64748b"
    })
    fig, ax1 = plt.subplots(figsize=(11, 4.8))
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    ax1.plot(prices.index, prices.values, color="#1e3a8a", linewidth=2.2, label="Share price")
    ax1.set_ylabel("Share Price ($)", color="#1e3a8a", fontweight="600")
    ax1.tick_params(axis="y", labelcolor="#1e3a8a")
    ax1.grid(True, axis="y", color="#f1f5f9", linewidth=1)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(net_ratio.index, net_ratio.values, color="#dc2626", linewidth=2.2, label="News mood")
    ax2.set_ylabel("News Mood", color="#dc2626", fontweight="600")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.axhline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax2.spines["top"].set_visible(False)

    ax1.set_title(f"{name} ({symbol})  ·  News Mood vs Share Price",
                  fontsize=13, fontweight="700", color="#0f172a", loc="left", pad=14)
    fig.tight_layout()
    return fig

st.markdown(f"""
<div class="app-header">
  <h1>Market Sentiment Monitor</h1>
  <p>Reads the latest news on major companies, scores the mood with FinBERT — a model trained on financial language — and tracks it against the share price.</p>
</div>
""", unsafe_allow_html=True)

col_a, col_b = st.columns([3, 1])
with col_a:
    symbol = st.selectbox("Company", list(tickers.keys()),
                          format_func=lambda s: f"{tickers[s]}  ({s})", label_visibility="collapsed")
with col_b:
    run = st.button("Analyze", use_container_width=True)

if run:
    with st.spinner("Reading the latest news and scoring the mood..."):
        news, net_ratio, prices = analyze(symbol)

    if news is None:
        st.warning("No recent news found for this company. Try another.")
    else:
        name = tickers[symbol]
        headline, note = build_verdict(name, news, net_ratio, prices)

        st.markdown(f"""
        <div class="verdict">
          <h3>{headline}</h3>
          <p>{note}</p>
        </div>
        """, unsafe_allow_html=True)

        counts = news["sentiment"].value_counts()
        pos = int(counts.get("positive", 0)); neg = int(counts.get("negative", 0)); neu = int(counts.get("neutral", 0))
        total = max(pos + neg + neu, 1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", f"{pos/total*100:.0f}%")
        c2.metric("Neutral", f"{neu/total*100:.0f}%")
        c3.metric("Negative", f"{neg/total*100:.0f}%")

        st.markdown('<p class="read-hint">Blue line: share price. Red line: news mood — rising means recent coverage is more positive, falling means more negative.</p>', unsafe_allow_html=True)
        st.pyplot(styled_chart(name, symbol, prices, net_ratio))

        st.markdown("##### Latest headlines")
        table = news[["headline", "sentiment"]].tail(15).iloc[::-1].reset_index(drop=True)
        table.columns = ["Headline", "Sentiment"]
        st.dataframe(table, use_container_width=True, hide_index=True)
