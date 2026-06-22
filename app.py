import streamlit as st
import pandas as pd
import numpy as np
import feedparser
import yfinance as yf
import matplotlib.pyplot as plt
from transformers import pipeline

st.set_page_config(page_title="Market Sentiment Monitor", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1150px;}
html, body, [class*="css"] {font-family: 'Inter', -apple-system, sans-serif;}
h1, h2, h3, h4 {color: #0f172a; letter-spacing: -0.01em;}
.app-title {font-size: 2rem; font-weight: 800; margin: 0;}
.app-sub {color: #64748b; font-size: 0.97rem; margin: 0.4rem 0 0 0;}
.divider {border-top: 1px solid #e6e8eb; margin: 1.2rem 0 1.6rem 0;}
.verdict {background: #f8fafc; border-left: 4px solid #1e3a8a; border-radius: 8px;
          padding: 1.1rem 1.3rem; margin: 0.4rem 0 1.4rem 0;}
.verdict h3 {font-size: 1.3rem; font-weight: 700; margin: 0 0 0.4rem 0;}
.verdict p {color: #475569; font-size: 0.97rem; margin: 0;}
.card {background: #fff; border: 1px solid #e6e8eb; border-radius: 10px; padding: 1rem 1.2rem;}
.driver-pos {border-left: 4px solid #16a34a;}
.driver-neg {border-left: 4px solid #dc2626;}
.driver-label {font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
          letter-spacing: 0.04em; margin-bottom: 0.3rem;}
.read-hint {color: #64748b; font-size: 0.88rem; margin: 1rem 0 0.4rem 0;}
.ov-row {display: flex; align-items: center; gap: 1rem; padding: 0.7rem 0.2rem;
          border-bottom: 1px solid #f1f5f9;}
.ov-rank {width: 24px; color: #94a3b8; font-weight: 700; font-size: 0.9rem;}
.ov-name {width: 150px; font-weight: 600; color: #0f172a;}
.ov-bar-wrap {flex: 1; height: 8px; background: #f1f5f9; border-radius: 999px; overflow: hidden;}
.ov-bar {height: 100%; border-radius: 999px;}
.ov-score {width: 90px; text-align: right; font-weight: 700;}
.stButton button {background: #1e3a8a; color: #fff; border: none; border-radius: 8px;
          padding: 0.55rem 1.6rem; font-weight: 600;}
.stButton button:hover {background: #1e40af; color: #fff;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e6e8eb;
          border-radius: 10px; padding: 0.9rem 1.1rem;}
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

def score_meta(score):
    if score >= 60:
        return "Bullish", "#16a34a"
    if score <= 40:
        return "Bearish", "#dc2626"
    return "Neutral", "#64748b"

def sentiment_score(pos, neg, total):
    return max(0.0, min(100.0, 50 + 50 * (pos - neg) / max(total, 1)))

@st.cache_data(ttl=1800, show_spinner=False)
def market_overview(limit=25):
    model = load_model()
    rows = []
    for sym, nm in tickers.items():
        news = get_news(nm)
        if news.empty:
            continue
        heads = news["headline"].head(limit).tolist()
        labels = [r["label"] for r in model(heads, truncation=True)]
        pos, neg, neu = labels.count("positive"), labels.count("negative"), labels.count("neutral")
        total = pos + neg + neu
        rows.append({"symbol": sym, "name": nm,
                     "score": sentiment_score(pos, neg, total)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

def analyze(symbol):
    name = tickers[symbol]
    model = load_model()
    news = get_news(name)
    if news.empty:
        return None, None, None
    news = news.head(120)
    scored = model(news["headline"].tolist(), truncation=True)
    news["sentiment"] = [r["label"] for r in scored]
    news["confidence"] = [float(r["score"]) for r in scored]
    news["published"] = pd.to_datetime(news["published"], errors="coerce", utc=True)
    news = news.dropna(subset=["published"]).set_index("published").sort_index()

    daily_net = news["sentiment"].resample("D").apply(
        lambda x: (x == "positive").sum() - (x == "negative").sum())
    daily_count = news["sentiment"].resample("D").count()
    net_ratio = daily_net / daily_count.where(daily_count >= 2, np.nan)
    net_ratio = net_ratio.rolling(7, min_periods=2).mean().ffill()

    prices = get_prices(symbol)
    return news, net_ratio, prices

def relationship(net_ratio, prices):
    s, p = net_ratio.dropna(), prices.dropna()
    if len(s) < 6 or len(p) < 6:
        return None
    sd = pd.Series(s.values, index=[d.date() for d in s.index])
    pp = pd.Series([float(x) for x in p.values], index=[d.date() for d in p.index])
    common = sorted(set(sd.index) & set(pp.index))
    if len(common) < 6:
        return None
    sa, pa = sd.loc[common].astype(float), pp.loc[common].astype(float)
    if sa.std() == 0 or pa.std() == 0:
        return None
    return float(np.corrcoef(sa.values, pa.values)[0, 1])

def verdict_text(name, news, net_ratio, prices):
    c = news["sentiment"].value_counts()
    pos, neg, neu = int(c.get("positive", 0)), int(c.get("negative", 0)), int(c.get("neutral", 0))
    total = max(pos + neg + neu, 1)
    mood = "leaning positive" if pos > neg * 1.3 else "leaning negative" if neg > pos * 1.3 else "mixed"
    headline = f"News on {name} is {mood} ({pos/total*100:.0f}% of recent headlines positive)"

    direction = ""
    r = net_ratio.dropna()
    if len(r) >= 5:
        recent = r.iloc[-5:]
        if recent.iloc[-1] < recent.iloc[0] - 0.05:
            direction = "Coverage has cooled over recent days"
        elif recent.iloc[-1] > recent.iloc[0] + 0.05:
            direction = "Coverage has warmed over recent days"
        else:
            direction = "Coverage has held broadly steady"

    price_note = ""
    p = prices.dropna()
    if len(p) >= 5:
        chg = float((p.iloc[-1] - p.iloc[-5]) / p.iloc[-5] * 100)
        price_note = f", while the share price moved {'up' if chg > 0 else 'down'} {abs(chg):.1f}% over the same window"

    note = f"{direction}{price_note}." if direction else ""
    return headline, note, pos, neg, neu, total

def meter(score):
    label, color = score_meta(score)
    return f'''
    <div style="margin:0.3rem 0 0.6rem 0;">
      <div style="display:flex; justify-content:space-between; align-items:baseline;">
        <span style="font-size:2.6rem; font-weight:800; color:{color};">{score:.0f}<span style="font-size:1rem; color:#94a3b8; font-weight:600;"> / 100</span></span>
        <span style="background:{color}1a; color:{color}; padding:0.3rem 0.8rem; border-radius:999px; font-weight:700; font-size:0.85rem;">{label}</span>
      </div>
      <div style="position:relative; height:10px; border-radius:999px; margin-top:0.6rem; background:linear-gradient(90deg,#dc2626,#e5e7eb,#16a34a);">
        <div style="position:absolute; top:-3px; left:calc({score:.0f}% - 2px); width:4px; height:16px; background:#0f172a; border-radius:2px;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#94a3b8; margin-top:0.3rem;">
        <span>Bearish</span><span>Neutral</span><span>Bullish</span>
      </div>
    </div>'''

def styled_chart(name, symbol, prices, net_ratio):
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.8,
                         "xtick.color": "#64748b", "ytick.color": "#64748b"})
    fig, ax1 = plt.subplots(figsize=(11, 4.6))
    fig.patch.set_facecolor("white"); ax1.set_facecolor("white")
    ax1.plot(prices.index, prices.values, color="#1e3a8a", linewidth=2.2)
    ax1.set_ylabel("Share Price ($)", color="#1e3a8a", fontweight="600")
    ax1.tick_params(axis="y", labelcolor="#1e3a8a")
    ax1.grid(True, axis="y", color="#f1f5f9", linewidth=1)
    ax1.spines["top"].set_visible(False)
    ax2 = ax1.twinx()
    ax2.plot(net_ratio.index, net_ratio.values, color="#dc2626", linewidth=2.2)
    ax2.set_ylabel("News Mood", color="#dc2626", fontweight="600")
    ax2.tick_params(axis="y", labelcolor="#dc2626")
    ax2.axhline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax2.spines["top"].set_visible(False)
    ax1.set_title(f"{name} ({symbol})   News Mood vs Share Price",
                  fontsize=13, fontweight="700", color="#0f172a", loc="left", pad=14)
    fig.tight_layout()
    return fig

st.markdown('<p class="app-title">Market Sentiment Monitor</p>'
            '<p class="app-sub">Reads the latest news on major companies, scores the mood with FinBERT — an AI model trained on financial language — and tracks it against share price.</p>',
            unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Market Overview", "Company Deep Dive"])

with tab1:
    st.markdown("#### Where the market mood sits right now")
    st.caption("Live sentiment across all tracked companies, ranked most positive to most negative. Runs once, then cached for 30 minutes.")
    if st.button("Scan the market", key="scan"):
        with st.spinner("Scoring sentiment across all companies..."):
            ov = market_overview()
        if ov.empty:
            st.warning("Could not load market data right now. Try again shortly.")
        else:
            html = ""
            for i, row in ov.iterrows():
                label, color = score_meta(row["score"])
                html += (f'<div class="ov-row"><div class="ov-rank">{i+1}</div>'
                         f'<div class="ov-name">{row["name"]} ({row["symbol"]})</div>'
                         f'<div class="ov-bar-wrap"><div class="ov-bar" style="width:{row["score"]:.0f}%; background:{color};"></div></div>'
                         f'<div class="ov-score" style="color:{color};">{row["score"]:.0f} · {label}</div></div>')
            st.markdown(html, unsafe_allow_html=True)
            top, bottom = ov.iloc[0], ov.iloc[-1]
            st.markdown(f'<p class="read-hint">Most positive coverage: <b>{top["name"]}</b>. Most negative: <b>{bottom["name"]}</b>. '
                        f'Open the Deep Dive tab for the full breakdown on any company.</p>', unsafe_allow_html=True)

with tab2:
    ca, cb = st.columns([3, 1])
    with ca:
        symbol = st.selectbox("Company", list(tickers.keys()),
                              format_func=lambda s: f"{tickers[s]}  ({s})", label_visibility="collapsed")
    with cb:
        run = st.button("Analyze", use_container_width=True, key="dive")

    if run:
        with st.spinner("Reading the latest news and scoring the mood..."):
            news, net_ratio, prices = analyze(symbol)
        if news is None:
            st.warning("No recent news found for this company. Try another.")
        else:
            name = tickers[symbol]
            headline, note, pos, neg, neu, total = verdict_text(name, news, net_ratio, prices)
            score = sentiment_score(pos, neg, total)

            st.markdown(f'<div class="verdict"><h3>{headline}</h3><p>{note}</p></div>', unsafe_allow_html=True)

            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Overall sentiment score**")
                st.markdown(meter(score), unsafe_allow_html=True)
            with right:
                m1, m2, m3 = st.columns(3)
                m1.metric("Positive", f"{pos/total*100:.0f}%")
                m2.metric("Neutral", f"{neu/total*100:.0f}%")
                m3.metric("Negative", f"{neg/total*100:.0f}%")
                rel = relationship(net_ratio, prices)
                if rel is not None:
                    strength = "strong" if abs(rel) > 0.6 else "moderate" if abs(rel) > 0.3 else "weak"
                    sign = "positive" if rel > 0 else "negative"
                    st.caption(f"Over this window, mood and price showed a {strength} {sign} relationship (correlation ≈ {rel:.2f}). Short windows are noisy — this is descriptive, not predictive.")

            st.markdown('<p class="read-hint">Blue line: share price. Red line: news mood — rising means coverage is turning more positive, falling means more negative.</p>', unsafe_allow_html=True)
            st.pyplot(styled_chart(name, symbol, prices, net_ratio))

            st.markdown("#### What's moving the mood")
            d1, d2 = st.columns(2)
            pos_rows = news[news["sentiment"] == "positive"].sort_values("confidence", ascending=False)
            neg_rows = news[news["sentiment"] == "negative"].sort_values("confidence", ascending=False)
            with d1:
                if not pos_rows.empty:
                    st.markdown(f'<div class="card driver-pos"><div class="driver-label" style="color:#16a34a;">Strongest positive</div>{pos_rows.iloc[0]["headline"]}</div>', unsafe_allow_html=True)
            with d2:
                if not neg_rows.empty:
                    st.markdown(f'<div class="card driver-neg"><div class="driver-label" style="color:#dc2626;">Strongest negative</div>{neg_rows.iloc[0]["headline"]}</div>', unsafe_allow_html=True)

            st.markdown("#### Latest headlines")
            table = news[["headline", "sentiment"]].tail(15).iloc[::-1].reset_index(drop=True)
            table.columns = ["Headline", "Sentiment"]
            st.dataframe(table, use_container_width=True, hide_index=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption("Built with FinBERT, yfinance and Streamlit. For research and educational use — not financial advice.")
