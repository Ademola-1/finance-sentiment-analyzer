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
          padding: 1.1rem 1.3rem; margin: 0.4rem 0 1.2rem 0;}
.verdict h3 {font-size: 1.3rem; font-weight: 700; margin: 0 0 0.4rem 0;}
.verdict p {color: #475569; font-size: 0.97rem; margin: 0;}
.alert {border-radius: 8px; padding: 1rem 1.2rem; margin: 0.2rem 0 1.4rem 0;
        background: #fffbeb; border-left: 4px solid #d97706;}
.alert .tag {font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
        letter-spacing: 0.05em; color: #b45309; margin-bottom: 0.3rem;}
.alert p {margin: 0; color: #78350f; font-size: 0.95rem;}
.card {background: #fff; border: 1px solid #e6e8eb; border-radius: 10px; padding: 1rem 1.2rem;}
.driver-pos {border-left: 4px solid #16a34a;}
.driver-neg {border-left: 4px solid #dc2626;}
.driver-label {font-size: 0.75rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.04em; margin-bottom: 0.3rem;}
.read-hint {color: #64748b; font-size: 0.88rem; margin: 1rem 0 0.4rem 0;}
.ov-row {display: flex; align-items: center; gap: 1rem; padding: 0.7rem 0.2rem;
        border-bottom: 1px solid #f1f5f9;}
.ov-rank {width: 24px; color: #94a3b8; font-weight: 700; font-size: 0.9rem;}
.ov-name {width: 170px; font-weight: 600; color: #0f172a;}
.ov-bar-wrap {flex: 1; height: 8px; background: #f1f5f9; border-radius: 999px; overflow: hidden;}
.ov-bar {height: 100%; border-radius: 999px;}
.ov-score {width: 100px; text-align: right; font-weight: 700;}
.stButton button {background: #1e3a8a; color: #fff; border: none; border-radius: 8px;
        padding: 0.55rem 1.6rem; font-weight: 600;}
.stButton button:hover {background: #1e40af; color: #fff;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e6e8eb;
        border-radius: 10px; padding: 0.9rem 1.1rem;}
</style>
""", unsafe_allow_html=True)

tickers = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
    "GOOGL": "Google", "META": "Meta", "TSLA": "Tesla", "NFLX": "Netflix",
    "AMD": "AMD", "JPM": "JPMorgan", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "V": "Visa", "MA": "Mastercard", "DIS": "Disney", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "WMT": "Walmart", "NKE": "Nike", "BA": "Boeing",
    "INTC": "Intel", "ORCL": "Oracle", "CRM": "Salesforce", "UBER": "Uber",
    "PYPL": "PayPal", "XOM": "ExxonMobil", "PFE": "Pfizer", "T": "AT&T"
}

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="ProsusAI/finbert")

@st.cache_data(ttl=86400)
def resolve_name(symbol):
    try:
        info = yf.Ticker(symbol).get_info()
        return info.get("shortName") or info.get("longName") or symbol
    except Exception:
        return symbol

@st.cache_data(ttl=1800)
def get_news(query):
    url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    df = pd.DataFrame([{"headline": e.title, "published": e.published} for e in feed.entries])
    if not df.empty:
        df = df.drop_duplicates(subset="headline").reset_index(drop=True)
    return df

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

def weighted_score(labels, scores):
    pos = sum(s for l, s in zip(labels, scores) if l == "positive")
    neg = sum(s for l, s in zip(labels, scores) if l == "negative")
    total = sum(scores)
    if total == 0:
        return 50.0
    return max(0.0, min(100.0, 50 + 50 * (pos - neg) / total))

@st.cache_data(ttl=1800, show_spinner=False)
def market_overview(limit=25):
    model = load_model()
    rows = []
    for sym, nm in tickers.items():
        news = get_news(nm)
        if news.empty:
            continue
        heads = news["headline"].head(limit).tolist()
        out = model(heads, truncation=True)
        labels = [r["label"] for r in out]
        scores = [float(r["score"]) for r in out]
        rows.append({"symbol": sym, "name": nm, "score": weighted_score(labels, scores)})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

def analyze(symbol, name):
    model = load_model()
    news = get_news(name)
    if news.empty:
        return None, None, None
    news = news.head(120)
    out = model(news["headline"].tolist(), truncation=True)
    news["sentiment"] = [r["label"] for r in out]
    news["confidence"] = [float(r["score"]) for r in out]
    news["published"] = pd.to_datetime(news["published"], errors="coerce", utc=True)
    news = news.dropna(subset=["published"]).set_index("published").sort_index()

    daily_net = news.resample("D").apply(
        lambda d: sum(c if s == "positive" else -c if s == "negative" else 0
                      for s, c in zip(d["sentiment"], d["confidence"])))
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

def divergence(net_ratio, prices):
    s, p = net_ratio.dropna(), prices.dropna()
    if len(s) < 6 or len(p) < 6:
        return None
    ms = s.iloc[-1] - s.iloc[-min(7, len(s))]
    pc = (p.iloc[-1] - p.iloc[-min(7, len(p))]) / p.iloc[-min(7, len(p))]
    if ms > 0.05 and pc < -0.01:
        return "News mood has been improving while the share price has been falling. When the two pull apart like this, it can flag a market that has not yet caught up to better news, a gap worth watching."
    if ms < -0.05 and pc > 0.01:
        return "News mood has been souring while the share price has been rising. A rising price on worsening coverage can signal momentum running ahead of the story, a gap worth watching."
    return None

def verdict_text(name, news, net_ratio, prices, score):
    c = news["sentiment"].value_counts()
    pos, neg, neu = int(c.get("positive", 0)), int(c.get("negative", 0)), int(c.get("neutral", 0))
    total = max(pos + neg + neu, 1)
    mood = "leaning positive" if score >= 55 else "leaning negative" if score <= 45 else "mixed"
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
    fig, ax1 = plt.subplots(figsize=(11, 5))
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
    ax1.set_title(f"{name} ({symbol}): News Mood vs Share Price",
                  fontsize=13, fontweight="700", color="#0f172a", loc="left", pad=14)
    fig.tight_layout()
    return fig

st.markdown('<p class="app-title">Market Sentiment Monitor</p>'
            '<p class="app-sub">Reads the latest news on any public company, scores the mood with FinBERT, a model trained on financial language, then tracks it against the share price and flags where the two disagree.</p>',
            unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Market Overview", "Company Deep Dive"])

with tab1:
    st.markdown("#### Where the market mood sits right now")
    st.caption("Live, confidence-weighted sentiment across tracked companies, ranked most positive to most negative. Runs once, then cached for 30 minutes.")
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
                         f'<div class="ov-score" style="color:{color};">{row["score"]:.0f}, {label}</div></div>')
            st.markdown(html, unsafe_allow_html=True)
            top, bottom = ov.iloc[0], ov.iloc[-1]
            st.markdown(f'<p class="read-hint">Most positive coverage: <b>{top["name"]}</b>. Most negative: <b>{bottom["name"]}</b>. '
                        f'Open the Deep Dive tab for the full breakdown on any company.</p>', unsafe_allow_html=True)

with tab2:
    st.markdown("Pick a company below, or type any ticker symbol (for example DIS, KO, BA) to analyze it live.")
    ca, cb, cc = st.columns([2, 2, 1])
    with ca:
        picked = st.selectbox("From the list", list(tickers.keys()),
                              format_func=lambda s: f"{tickers[s]}  ({s})", label_visibility="collapsed")
    with cb:
        typed = st.text_input("Or type a ticker", placeholder="Any ticker, e.g. DIS", label_visibility="collapsed")
    with cc:
        run = st.button("Analyze", use_container_width=True, key="dive")

    if run:
        symbol = typed.strip().upper() if typed.strip() else picked
        name = tickers.get(symbol) or resolve_name(symbol)
        with st.spinner("Reading the latest news and scoring the mood..."):
            news, net_ratio, prices = analyze(symbol, name)
        if news is None or prices is None or len(prices.dropna()) == 0:
            st.warning(f"Could not find enough data for '{symbol}'. Check the ticker symbol and try again.")
        else:
            out_labels = news["sentiment"].tolist()
            out_scores = news["confidence"].tolist()
            score = weighted_score(out_labels, out_scores)
            headline, note, pos, neg, neu, total = verdict_text(name, news, net_ratio, prices, score)

            st.markdown(f'<div class="verdict"><h3>{headline}</h3><p>{note}</p></div>', unsafe_allow_html=True)

            div_msg = divergence(net_ratio, prices)
            if div_msg:
                st.markdown(f'<div class="alert"><div class="tag">Divergence detected</div><p>{div_msg} This is a descriptive observation, not a prediction or financial advice.</p></div>', unsafe_allow_html=True)

            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Overall sentiment score** (confidence-weighted)")
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
                    st.caption(f"Over this window, mood and price showed a {strength} {sign} relationship (correlation about {rel:.2f}). Short windows are noisy, so treat this as descriptive, not predictive.")

            st.markdown('<p class="read-hint">Blue line: share price. Red line: news mood. When it rises, coverage is turning more positive; when it falls, more negative.</p>', unsafe_allow_html=True)
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
st.caption("Built with FinBERT, yfinance and Streamlit. Sentiment is confidence-weighted and de-duplicated. For research and educational use, not financial advice.")
