import streamlit as st
import pandas as pd
import numpy as np
import feedparser
import yfinance as yf
import matplotlib.pyplot as plt
from transformers import pipeline

st.set_page_config(page_title="Financial News Sentiment", layout="wide")

tickers = {
    "AAPL": "Apple",
    "TSLA": "Tesla",
    "NVDA": "Nvidia",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Google",
    "META": "Meta",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "JPM": "JPMorgan"
}

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="ProsusAI/finbert")

@st.cache_data(ttl=1800)
def get_news(name):
    url = f"https://news.google.com/rss/search?q={name}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries:
        rows.append({"headline": entry.title, "published": entry.published})
    return pd.DataFrame(rows)

@st.cache_data(ttl=1800)
def get_prices(symbol):
    data = yf.download(symbol, period="2mo", progress=False)["Close"]
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

st.title("Financial News Sentiment vs Stock Price")
st.write("Live news sentiment scored with FinBERT, tracked against recent stock price movement.")

symbol = st.selectbox("Select a company", list(tickers.keys()),
                      format_func=lambda s: f"{tickers[s]} ({s})")

if st.button("Analyze"):
    with st.spinner("Pulling news and scoring sentiment..."):
        news, net_ratio, prices = analyze(symbol)

    if news is None:
        st.warning("No recent news found for this company. Try another.")
    else:
        counts = news["sentiment"].value_counts()
        pos = int(counts.get("positive", 0))
        neg = int(counts.get("negative", 0))
        neu = int(counts.get("neutral", 0))
        total = pos + neg + neu

        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", f"{pos} ({pos/total*100:.0f}%)")
        c2.metric("Neutral", f"{neu} ({neu/total*100:.0f}%)")
        c3.metric("Negative", f"{neg} ({neg/total*100:.0f}%)")

        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(prices.index, prices.values, color="#1f77b4", linewidth=2)
        ax1.set_ylabel("Stock Price ($)", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

        ax2 = ax1.twinx()
        ax2.plot(net_ratio.index, net_ratio.values, color="#d62728", linewidth=2)
        ax2.set_ylabel("News Sentiment (net positivity)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)

        plt.title(f"{tickers[symbol]} ({symbol}) — News Sentiment vs Stock Price")
        fig.tight_layout()
        st.pyplot(fig)

        st.subheader("Recent headlines")
        st.dataframe(news[["headline", "sentiment"]].tail(15).iloc[::-1],
                     use_container_width=True)
