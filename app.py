import streamlit as st
import pandas as pd
import numpy as np
import feedparser
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from urllib.parse import quote
from transformers import pipeline

st.set_page_config(page_title="Market Sentiment Monitor", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 1140px;}
html, body, [class*="css"] {font-family: 'Inter', -apple-system, sans-serif;}
h1,h2,h3,h4 {color:#0f172a; letter-spacing:-0.01em;}
.hero {background: linear-gradient(135deg, #1e293b 0%, #1e3a8a 100%);
       border-radius: 16px; padding: 1.8rem 2rem; margin-bottom: 1.6rem; color:#fff;}
.hero h1 {color:#fff; font-size:1.95rem; font-weight:800; margin:0;}
.hero p {color:#cbd5e1; font-size:0.98rem; margin:0.5rem 0 0 0; max-width:760px; line-height:1.5;}
.hero-hook {color:#fff; font-size:1.3rem; font-weight:700; margin:0.7rem 0 0.2rem 0; line-height:1.4;}
.divider {border-top:1px solid #e6e8eb; margin:1.1rem 0 1.5rem 0;}
.section-label {font-size:0.78rem; font-weight:700; text-transform:uppercase;
       letter-spacing:0.05em; color:#94a3b8; margin-bottom:0.4rem;}
.insight {background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 100%); color:#fff;
       border-radius:12px; padding:1.3rem 1.5rem; margin:0.3rem 0 1.2rem 0;}
.insight .lead {font-size:0.74rem; font-weight:700; text-transform:uppercase;
       letter-spacing:0.06em; color:#93c5fd; margin-bottom:0.45rem;}
.insight h3 {color:#fff; font-size:1.32rem; font-weight:700; margin:0 0 0.3rem 0; line-height:1.35;}
.insight p {color:#cbd5e1; font-size:0.93rem; margin:0;}
.signal {display:inline-flex; align-items:center; gap:0.5rem; font-weight:700;
       font-size:0.85rem; padding:0.3rem 0.85rem; border-radius:999px; margin-top:0.7rem;}
.verdict {background:#f8fafc; border-left:4px solid #1e3a8a; border-radius:10px;
          padding:1rem 1.2rem; margin:0.2rem 0 1.2rem 0;}
.verdict p {color:#475569; font-size:0.95rem; margin:0;}
.alert {border-radius:10px; padding:1rem 1.2rem; margin:0.2rem 0 1.4rem 0;
        background:#fffbeb; border-left:4px solid #d97706;}
.alert .tag {font-size:0.72rem; font-weight:800; text-transform:uppercase;
        letter-spacing:0.05em; color:#b45309; margin-bottom:0.3rem;}
.alert p {margin:0; color:#78350f; font-size:0.95rem;}
.card {background:#fff; border:1px solid #e6e8eb; border-radius:12px; padding:1rem 1.2rem;}
.driver-pos {border-left:4px solid #16a34a;}
.driver-neg {border-left:4px solid #dc2626;}
.driver-label {font-size:0.75rem; font-weight:700; text-transform:uppercase;
        letter-spacing:0.04em; margin-bottom:0.35rem;}
.impact {background:#f8fafc; border:1px solid #e6e8eb; border-radius:12px;
        padding:1.1rem 1.3rem; margin:0.4rem 0 1rem 0;}
.impact .t {font-size:0.78rem; font-weight:700; text-transform:uppercase;
        letter-spacing:0.05em; color:#94a3b8; margin-bottom:0.4rem;}
.impact .big {font-size:1.15rem; font-weight:700; color:#0f172a;}
.impact .sub {color:#64748b; font-size:0.86rem; margin-top:0.35rem;}
.read-hint {color:#64748b; font-size:0.88rem; margin:0.6rem 0 0.3rem 0;}
.ov-row {display:flex; align-items:center; gap:1rem; padding:0.7rem 0.2rem;
        border-bottom:1px solid #f1f5f9;}
.ov-rank {width:24px; color:#94a3b8; font-weight:700; font-size:0.9rem;}
.ov-name {width:200px; font-weight:600; color:#0f172a;}
.ov-bar-wrap {flex:1; height:9px; background:#f1f5f9; border-radius:999px; overflow:hidden;}
.ov-bar {height:100%; border-radius:999px;}
.ov-score {width:110px; text-align:right; font-weight:700;}
.stButton button {background:#1e3a8a; color:#fff; border:none; border-radius:9px;
        padding:0.55rem 1.6rem; font-weight:600;}
.stButton button:hover {background:#1e40af; color:#fff;}
div[data-testid="stMetric"] {background:#fff; border:1px solid #e6e8eb;
        border-radius:12px; padding:0.9rem 1.1rem;}
</style>
""", unsafe_allow_html=True)

tickers = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia", "AMZN": "Amazon",
    "GOOGL": "Alphabet (Google)", "META": "Meta", "TSLA": "Tesla", "NFLX": "Netflix",
    "AMD": "AMD", "INTC": "Intel", "ORCL": "Oracle", "CRM": "Salesforce",
    "IBM": "IBM", "ADBE": "Adobe", "UBER": "Uber", "PYPL": "PayPal",
    "AVGO": "Broadcom", "QCOM": "Qualcomm", "TXN": "Texas Instruments", "CSCO": "Cisco",
    "DELL": "Dell", "PLTR": "Palantir", "SHOP": "Shopify", "SPOT": "Spotify",
    "ABNB": "Airbnb", "SNAP": "Snap", "COIN": "Coinbase",
    "RBLX": "Roblox", "DASH": "DoorDash",
    "BABA": "Alibaba", "TSM": "TSMC", "SONY": "Sony", "TM": "Toyota",
    "JPM": "JPMorgan Chase", "BAC": "Bank of America", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "WFC": "Wells Fargo", "C": "Citigroup",
    "AXP": "American Express", "BLK": "BlackRock", "BRK-B": "Berkshire Hathaway",
    "V": "Visa", "MA": "Mastercard", "DIS": "Disney", "KO": "Coca-Cola",
    "PEP": "PepsiCo", "MCD": "McDonald's", "SBUX": "Starbucks", "CMG": "Chipotle",
    "NKE": "Nike", "LULU": "Lululemon", "WMT": "Walmart", "COST": "Costco",
    "HD": "Home Depot", "LOW": "Lowe's", "TGT": "Target", "BA": "Boeing",
    "DAL": "Delta Air Lines", "UAL": "United Airlines", "F": "Ford",
    "GM": "General Motors", "GE": "GE Aerospace",
    "XOM": "ExxonMobil", "CVX": "Chevron", "PFE": "Pfizer", "JNJ": "Johnson & Johnson",
    "LLY": "Eli Lilly", "UNH": "UnitedHealth", "CVS": "CVS Health",
    "T": "AT&T", "VZ": "Verizon"
}

OVERVIEW_SYMBOLS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","NFLX","AMD","AVGO",
    "INTC","QCOM","CRM","ADBE","ORCL","PLTR","UBER","SHOP","COIN","DELL",
    "CSCO","IBM","TXN","SPOT","ABNB","BABA","TSM","SONY",
    "JPM","BAC","GS","MS","WFC","C","V","MA","BLK","AXP",
    "DIS","KO","PEP","MCD","SBUX","NKE","WMT","COST","HD","BA",
    "XOM","CVX","LLY","UNH","JNJ"
]

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="ProsusAI/finbert")

@st.cache_data(ttl=1800)
def get_news(query):
    url = f"https://news.google.com/rss/search?q={quote(query)}+stock&hl=en-US&gl=US&ceid=US:en"
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
def market_overview(limit=8):
    model = load_model()
    rows = []
    for sym in OVERVIEW_SYMBOLS:
        nm = tickers[sym]
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

def aligned(net_ratio, prices):
    s, p = net_ratio.dropna(), prices.dropna()
    if len(s) < 6 or len(p) < 6:
        return None, None
    sd = pd.Series(s.values, index=[d.date() for d in s.index])
    pp = pd.Series([float(x) for x in p.values], index=[d.date() for d in p.index])
    common = sorted(set(sd.index) & set(pp.index))
    if len(common) < 6:
        return None, None
    return sd.loc[common].astype(float), pp.loc[common].astype(float)

def relationship(net_ratio, prices):
    m, p = aligned(net_ratio, prices)
    if m is None or m.std() == 0 or p.std() == 0:
        return None
    return float(np.corrcoef(m.values, p.values)[0, 1])

def sentiment_impact(net_ratio, prices):
    m, p = aligned(net_ratio, prices)
    if m is None or len(m) < 8 or m.std() == 0:
        return None
    ret = p.pct_change()
    df = pd.DataFrame({"mood": m, "ret": ret}).dropna()
    if len(df) < 8 or df["mood"].std() == 0:
        return None
    x, y = df["mood"].values, df["ret"].values
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"per_tenth": b * 100 * 0.1, "r2": r2, "n": len(df)}

def divergence(net_ratio, prices):
    s, p = net_ratio.dropna(), prices.dropna()
    if len(s) < 6 or len(p) < 6:
        return None
    ms = s.iloc[-1] - s.iloc[-min(7, len(s))]
    pc = (p.iloc[-1] - p.iloc[-min(7, len(p))]) / p.iloc[-min(7, len(p))]
    if ms > 0.05 and pc < -0.01:
        return "News mood has been improving while the share price has been falling. When the two pull apart like this, it can flag a market that has not yet caught up to better news."
    if ms < -0.05 and pc > 0.01:
        return "News mood has been souring while the share price has been rising. A rising price on worsening coverage can signal momentum running ahead of the story."
    return None

def lead_insight(name, rel):
    if rel is None:
        return f"How {name}'s news mood lines up with its share price", "Not enough recent data to measure the relationship yet."
    strength = "strongly" if abs(rel) > 0.6 else "moderately" if abs(rel) > 0.3 else "weakly"
    if rel > 0.3:
        return (f"{name}'s news mood and share price have been moving together",
                f"Over the recent window they tracked each other {strength} (correlation about {rel:.2f}). When coverage improved, the price tended to rise with it.")
    if rel < -0.3:
        return (f"{name}'s news mood and share price have been moving against each other",
                f"Over the recent window they moved {strength} in opposite directions (correlation about {rel:.2f}). Worth a closer look at why.")
    return (f"{name}'s news mood and share price show little clear link right now",
            f"Over the recent window the relationship was {strength} (correlation about {rel:.2f}). Short windows are noisy, so read this as a snapshot.")

def verdict_text(name, news, net_ratio, prices, score):
    c = news["sentiment"].value_counts()
    pos, neg, neu = int(c.get("positive", 0)), int(c.get("negative", 0)), int(c.get("neutral", 0))
    total = max(pos + neg + neu, 1)
    mood = "leaning positive" if score >= 55 else "leaning negative" if score <= 45 else "mixed"
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
        price_note = f", and the share price moved {'up' if chg > 0 else 'down'} {abs(chg):.1f}%"
    summary = f"News on {name} is currently {mood} ({pos/total*100:.0f}% of recent headlines positive). {direction}{price_note}." if direction else f"News on {name} is currently {mood} ({pos/total*100:.0f}% of recent headlines positive)."
    return summary, pos, neg, neu, total

def meter(score):
    label, color = score_meta(score)
    return f'''
    <div style="margin:0.3rem 0 0.6rem 0;">
      <div style="display:flex; justify-content:space-between; align-items:baseline;">
        <span style="font-size:2.8rem; font-weight:800; color:{color};">{score:.0f}<span style="font-size:1rem; color:#94a3b8; font-weight:600;"> / 100</span></span>
        <span style="background:{color}1a; color:{color}; padding:0.3rem 0.85rem; border-radius:999px; font-weight:700; font-size:0.85rem;">{label}</span>
      </div>
      <div style="position:relative; height:11px; border-radius:999px; margin-top:0.6rem; background:linear-gradient(90deg,#dc2626,#e5e7eb,#16a34a);">
        <div style="position:absolute; top:-3px; left:calc({score:.0f}% - 2px); width:4px; height:17px; background:#0f172a; border-radius:2px;"></div>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:0.72rem; color:#94a3b8; margin-top:0.3rem;">
        <span>Bearish</span><span>Neutral</span><span>Bullish</span>
      </div>
    </div>'''

def style_sentiment_table(news):
    rows = news[["headline", "sentiment"]].tail(15).iloc[::-1].reset_index(drop=True)
    colors = {"positive": ("#16a34a", "#dcfce7"),
              "negative": ("#dc2626", "#fee2e2"),
              "neutral": ("#64748b", "#f1f5f9")}
    html = '<div style="border:1px solid #e6e8eb; border-radius:12px; overflow:hidden;">'
    for i, r in rows.iterrows():
        bg = "#ffffff" if i % 2 == 0 else "#fafbfc"
        fg, chip = colors.get(r["sentiment"], colors["neutral"])
        html += (f'<div style="display:flex; align-items:center; gap:1rem; padding:0.7rem 1rem; background:{bg}; border-bottom:1px solid #f1f5f9;">'
                 f'<div style="flex:1; color:#334155; font-size:0.92rem;">{r["headline"]}</div>'
                 f'<span style="background:{chip}; color:{fg}; padding:0.22rem 0.7rem; border-radius:999px; font-size:0.75rem; font-weight:700; text-transform:capitalize; white-space:nowrap;">{r["sentiment"]}</span>'
                 f'</div>')
    html += '</div>'
    return html

def plotly_chart(name, symbol, prices, net_ratio):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=net_ratio.index, y=net_ratio.values, name="News mood",
        line=dict(color="#dc2626", width=2.4, shape="spline"),
        fill="tozeroy", fillcolor="rgba(220,38,38,0.08)",
        hovertemplate="%{x|%b %d}<br>Mood: %{y:.2f}<extra></extra>"),
        secondary_y=True)
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values, name="Share price",
        line=dict(color="#1e3a8a", width=3, shape="spline"),
        hovertemplate="%{x|%b %d}<br>Price: $%{y:.2f}<extra></extra>"),
        secondary_y=False)
    fig.add_hline(y=0, line_dash="dot", line_color="#cbd5e1", secondary_y=True)
    fig.update_layout(
        title=dict(text=f"<b>{name} ({symbol})</b>  News Mood vs Share Price",
                   font=dict(size=17, color="#0f172a"), y=0.97),
        height=520, plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", color="#334155", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=10, r=10, t=80, b=20),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor="#e2e8f0", font_size=12))
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0",
                     rangeslider=dict(visible=True, thickness=0.08, bgcolor="#f8fafc"))
    fig.update_yaxes(title_text="Share Price ($)", secondary_y=False,
                     showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0",
                     title_font_color="#1e3a8a", tickfont_color="#1e3a8a", tickprefix="$")
    fig.update_yaxes(title_text="News Mood", secondary_y=True,
                     showgrid=False, title_font_color="#dc2626", tickfont_color="#dc2626", zeroline=False)
    return fig

st.markdown("""
<div class="hero">
  <h1>Market Sentiment Monitor</h1>
  <div class="hero-hook">Does the news move the stock, or the stock move the news?</div>
  <p>Every day, hundreds of headlines shape how investors feel about a company, often before the share price reflects it. No one can read them all. This tool does: it scores the tone of the latest financial news with FinBERT, tracks that mood against the share price, and flags the moments when the two pull apart, the gaps that often matter most.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Market Overview", "Company Deep Dive"])

with tab1:
    st.markdown("#### Today's news, ranked from bullish to bearish")
    st.markdown('<p style="color:#475569; font-size:0.97rem; margin:0.2rem 0 1rem 0;">News moves a stock before the numbers do. In one click, this reads the latest coverage on the world\'s biggest companies, scores the tone of every headline with FinBERT, and ranks the entire market from the most upbeat to the most uneasy. A live pulse on where sentiment sits right now.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex; gap:0.8rem; margin:0.5rem 0 1.3rem 0; flex-wrap:wrap;">
      <div style="flex:1; min-width:200px; background:#f8fafc; border:1px solid #e6e8eb; border-radius:12px; padding:1rem 1.1rem;">
        <div style="font-size:1.4rem; font-weight:800; color:#1e3a8a;">1</div>
        <div style="font-weight:600; color:#0f172a; margin:0.2rem 0;">Pulls live news</div>
        <div style="color:#64748b; font-size:0.86rem;">The latest headlines on dozens of major companies, gathered in real time.</div>
      </div>
      <div style="flex:1; min-width:200px; background:#f8fafc; border:1px solid #e6e8eb; border-radius:12px; padding:1rem 1.1rem;">
        <div style="font-size:1.4rem; font-weight:800; color:#1e3a8a;">2</div>
        <div style="font-weight:600; color:#0f172a; margin:0.2rem 0;">Scores the mood</div>
        <div style="color:#64748b; font-size:0.86rem;">FinBERT reads each headline as positive, negative or neutral.</div>
      </div>
      <div style="flex:1; min-width:200px; background:#f8fafc; border:1px solid #e6e8eb; border-radius:12px; padding:1rem 1.1rem;">
        <div style="font-size:1.4rem; font-weight:800; color:#1e3a8a;">3</div>
        <div style="font-weight:600; color:#0f172a; margin:0.2rem 0;">Ranks the market</div>
        <div style="color:#64748b; font-size:0.86rem;">Every company lined up from most bullish to most bearish coverage.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Scan the market", key="scan"):
        with st.spinner("Scoring sentiment across the market..."):
            ov = market_overview()
        if ov.empty:
            st.warning("Could not load market data right now. Try again shortly.")
        else:
            bullish = int((ov["score"] >= 60).sum())
            bearish = int((ov["score"] <= 40).sum())
            neutral = len(ov) - bullish - bearish
            avg = ov["score"].mean()
            if avg >= 55:
                tone = "leaning optimistic"
            elif avg <= 45:
                tone = "leaning cautious"
            else:
                tone = "broadly mixed"
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#0f172a,#1e3a8a); color:#fff; '
                f'border-radius:12px; padding:1.2rem 1.4rem; margin:0.3rem 0 1.3rem 0;">'
                f'<div style="font-size:0.74rem; font-weight:700; text-transform:uppercase; letter-spacing:0.06em; color:#93c5fd; margin-bottom:0.4rem;">Market mood right now</div>'
                f'<div style="font-size:1.3rem; font-weight:700;">The market is {tone} today.</div>'
                f'<div style="color:#cbd5e1; font-size:0.92rem; margin-top:0.35rem;">'
                f'{bullish} bullish, {neutral} neutral, {bearish} bearish across {len(ov)} major companies. Average mood score {avg:.0f} out of 100.</div></div>',
                unsafe_allow_html=True)
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
    st.markdown("#### Go deep on one company")
    st.markdown('<p style="color:#475569; font-size:0.97rem; margin:0.2rem 0 1rem 0;">Pick any major company to see the full story: whether its news mood and share price are moving together or pulling apart, how strongly sentiment has tracked the stock, the headlines driving the mood right now, and a live feed of the latest coverage scored one by one.</p>', unsafe_allow_html=True)
    ca, cb = st.columns([3, 1])
    with ca:
        options = sorted(tickers.keys(), key=lambda s: tickers[s].lower())
        default_idx = options.index("AAPL") if "AAPL" in options else 0
        symbol = st.selectbox("Company", options, index=default_idx,
                              format_func=lambda s: tickers[s],
                              label_visibility="collapsed")
    with cb:
        run = st.button("Analyze", use_container_width=True, key="dive")
    st.caption("Start typing a company name to filter the list.")

    if run:
        name = tickers[symbol]
        with st.spinner("Reading the latest news and scoring the mood..."):
            news, net_ratio, prices = analyze(symbol, name)
        if news is None or prices is None or len(prices.dropna()) == 0:
            st.warning(f"Could not find enough recent data for {name}. Try another company.")
        else:
            score = weighted_score(news["sentiment"].tolist(), news["confidence"].tolist())
            rel = relationship(net_ratio, prices)

            lead_h, lead_p = lead_insight(name, rel)
            div_msg = divergence(net_ratio, prices)
            if div_msg:
                sig = '<span class="signal" style="background:#fef3c7; color:#b45309;">Mood and price are diverging</span>'
            elif rel is not None and abs(rel) > 0.3:
                sig = '<span class="signal" style="background:#dcfce7; color:#15803d;">Mood and price are moving together</span>'
            else:
                sig = '<span class="signal" style="background:#f1f5f9; color:#64748b;">No clear link right now</span>'
            st.markdown(f'<div class="insight"><div class="lead">Key insight</div><h3>{lead_h}</h3><p>{lead_p}</p>{sig}</div>', unsafe_allow_html=True)

            if div_msg:
                st.markdown(f'<div class="alert"><div class="tag">Divergence detected</div><p>{div_msg} This is a descriptive observation, not a prediction or financial advice.</p></div>', unsafe_allow_html=True)

            st.markdown('<p class="read-hint">Watch how the two lines move, together or apart. Blue is the share price, red is the news mood. Hover any point for the exact date, price and mood.</p>', unsafe_allow_html=True)
            st.plotly_chart(plotly_chart(name, symbol, prices, net_ratio), use_container_width=True)

            imp = sentiment_impact(net_ratio, prices)
            if imp is not None:
                conf = "a clear" if imp["r2"] > 0.3 else "a modest" if imp["r2"] > 0.1 else "a faint"
                st.markdown(
                    f'<div class="impact"><div class="t">Sentiment impact (linear regression)</div>'
                    f'<div class="big">A 0.1 rise in news mood was associated with a {imp["per_tenth"]:+.2f}% move in the share price.</div>'
                    f'<div class="sub">The regression explains {conf} share of the price movement (R squared about {imp["r2"]:.2f}, based on {imp["n"]} days). Association over a short window, not causation.</div></div>',
                    unsafe_allow_html=True)

            st.markdown("#### The breakdown")
            st.markdown(f'<div class="verdict"><p>{verdict_text(name, news, net_ratio, prices, score)[0]}</p></div>', unsafe_allow_html=True)
            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Overall sentiment score** (confidence-weighted)")
                st.markdown(meter(score), unsafe_allow_html=True)
            with right:
                _, pos, neg, neu, total = verdict_text(name, news, net_ratio, prices, score)
                m1, m2, m3 = st.columns(3)
                m1.metric("Positive", f"{pos/total*100:.0f}%")
                m2.metric("Neutral", f"{neu/total*100:.0f}%")
                m3.metric("Negative", f"{neg/total*100:.0f}%")

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
            st.markdown(style_sentiment_table(news), unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption("Built with FinBERT, yfinance, Plotly and Streamlit. Sentiment is confidence-weighted and de-duplicated. For research and educational use, not financial advice.")
