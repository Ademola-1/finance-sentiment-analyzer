import streamlit as st
import pandas as pd
import numpy as np
import feedparser
import yfinance as yf
import matplotlib.pyplot as plt
from transformers import pipeline

st.set_page_config(page_title="Financial News Sentiment", layout="wide")

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
    return yf.download(symbol, period="2mo", progress=False)["Close"]

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
    pos = int(counts.get("positive", 0))
    neg = int(counts.get("negative", 0))
    neu = int(counts.get("neutral", 0))
    total = max(pos + neg + neu, 1)
    pos_pct = pos / total * 100

    if pos > neg * 1.3:
        mood = "mostly positive"
    elif neg > pos * 1.3:
        mood = "mostly negative"
    else:
        mood = "mixed"

    headline = f"News about {name} is {mood} right now ({pos_pct:.0f}% of recent headlines are positive)."

    direction = ""
    if len(net_ratio.dropna()) >= 5:
        recent = net_ratio.dropna().iloc[-5:]
        if recent.iloc[-1] < recent.iloc[0] - 0.05:
            direction = "Sentiment has been cooling over the past few days"
        elif recent.iloc[-1] > recent.iloc[0] + 0.05:
            direction = "Sentiment has been improving over the past few days"
        else:
            direction = "Sentiment has been broadly steady recently"

    price_note = ""
    if len(prices.dropna()) >= 5:
        p = prices.dropna()
        change = (p.iloc[-1] - p.iloc[-5]) / p.iloc[-5] * 100
        arrow = "up" if change > 0 else "down"
        price_note = f", while the stock is {arrow} {abs(change):.1f}% over the same stretch"

    note = f"{direction}{price_note}." if direction else ""
    return headline, note

st.title("Financial News Sentiment vs Stock Price")
st.caption("Pick a company. This tool reads its latest news, scores the mood with an AI model trained on financial language (FinBERT), and shows it against the stock price.")

symbol = st.selectbox("Select a company", list(tickers.keys()),
                      format_func=lambda s: f"{tickers[s]} ({s})")

if st.button("Analyze"):
    with st.spinner("Reading the latest news and scoring the mood..."):
        news, net_ratio, prices = analyze(symbol)

    if news is None:
        st.warning("No recent news found for this company. Try another.")
    else:
        name = tickers[symbol]
        headline, note = build_verdict(name, news, net_ratio, prices)

        st.subheader(headline)
        if note:
            st.write(note)

        counts = news["sentiment"].value_counts()
        pos = int(counts.get("positive", 0)); neg = int(counts.get("negative", 0)); neu = int(counts.get("neutral", 0))
        total = max(pos + neg + neu, 1)
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive news", f"{pos/total*100:.0f}%")
        c2.metric("Neutral news", f"{neu/total*100:.0f}%")
        c3.metric("Negative news", f"{neg/total*100:.0f}%")

        st.markdown("**How to read the chart:** blue line is the stock price. Red line is the news mood — when it rises, recent news is more positive; when it falls, news is more negative.")

        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(prices.index, prices.values, color="#1f77b4", linewidth=2)
        ax1.set_ylabel("Stock Price ($)", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

        ax2 = ax1.twinx()
        ax2.plot(net_ratio.index, net_ratio.values, color="#d62728", linewidth=2)
        ax2.set_ylabel("News Mood", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)

        plt.title(f"{name} ({symbol}) — News Mood vs Stock Price")
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Latest headlines and how they scored")
        st.dataframe(news[["headline", "sentiment"]].tail(15).iloc[::-1],
                     use_container_width=True)
