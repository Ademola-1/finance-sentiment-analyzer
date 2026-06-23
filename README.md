# Market Sentiment Monitor

A live web app that reads the latest news on major companies, scores the tone with FinBERT (a language model trained on financial text), and tracks that news mood against the company's share price. It also flags moments where mood and price pull apart, which can be early signals worth watching.

**Live app:** [https://finance-sentiment-analyzer.streamlit.app/]

## What it does

The tool answers a simple question in a clear way: *is the news around this company positive or negative right now, and is the stock moving with it?*

**Market Overview**
Scans a set of well-known companies live and ranks them from most positive to most negative coverage using a confidence-weighted sentiment score. One glance shows where the market mood sits.

**Company Deep Dive**
Pick a company from the list (start typing to filter it) and it shows:
- A plain-English verdict on the current news mood
- A 0 to 100 sentiment score with a Bearish, Neutral, or Bullish reading
- An interactive chart of news mood against share price, with hover details for any date
- A divergence alert when mood and price move in opposite directions
- The strongest positive and negative headlines driving the score
- The latest headlines with their individual sentiment labels

## How it works

1. **News collection.** Recent headlines are pulled live from Google News for the chosen company, then de-duplicated so one widely-syndicated story does not skew the result.
2. **Sentiment scoring.** Each headline is scored by FinBERT (`ProsusAI/finbert`) as positive, negative, or neutral, with a confidence value.
3. **Confidence-weighted score.** Headlines the model is more certain about count more toward the overall score, rather than treating every headline equally.
4. **Mood trend.** Daily sentiment is smoothed into a rolling net-positivity line, with thinly-covered days excluded so single headlines do not create false swings.
5. **Price overlay.** Daily closing prices are pulled with yfinance for the same window and charted alongside the mood line using Plotly.
6. **Relationship and divergence.** The app reports the correlation between mood and price over the window, and flags when the two diverge.

## Tech stack

- **Python**
- **FinBERT** via Hugging Face Transformers for financial sentiment
- **yfinance** for price data
- **feedparser** for live news
- **Plotly** for the interactive chart
- **Streamlit** for the app and deployment
- **pandas, numpy** for processing

## Running it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Limitations and future work

This is built as a research and learning tool, and it is honest about its edges:

- **Headline-level sentiment.** The model reads headlines, not full articles. Headlines can be vague or sensational, so the score is a useful proxy rather than a complete read of each story. Reading article bodies is a natural next step.
- **Short window.** The free news source returns mainly recent items, so the analysis covers a few recent weeks. Correlations over short windows are noisy and should be read as descriptive, not predictive.
- **Divergence is a heuristic.** The divergence flag uses chosen thresholds, not a statistically validated trading signal. It is meant to draw attention, not to recommend action.
- **Curated company list.** The app covers a set of major, well-covered companies where news and price data are reliable. Extending coverage cleanly to more names is a future step.

**Planned v2:**
- Article-level sentiment rather than headlines only
- A lead-lag analysis to test whether mood tends to move ahead of price
- A Nigeria edition using NGX price data and local news sources

## Disclaimer

For research and educational use only. This is not financial advice.

## Author

Built by Mubarak Lawal.
Portfolio: [https://mubaraklawal.com/]
