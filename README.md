# Market Sentiment Monitor

**Does the news move the stock?** This app reads the latest news on major companies, scores the tone with FinBERT, and tracks that news mood against the share price, flagging the moments when the two pull apart.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FinBERT](https://img.shields.io/badge/Model-FinBERT-red) ![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b) ![License](https://img.shields.io/badge/Use-Educational-green)

**Live app:** [finance-sentiment-analyzer.streamlit.app](https://finance-sentiment-analyzer.streamlit.app/)

---

## The idea

Markets do not move on numbers alone. They move on stories. Before a share price shifts, there is usually a wave of news pushing people to act. This tool measures that wave: it scores the mood of a company's recent news and lays it next to the share price, so you can see whether the two move together, and catch the moments when they do not.

## What it does

**Market Overview** ranks major companies live, from the most positive news coverage to the most negative, using a confidence-weighted sentiment score. A quick read on where the market's mood sits today.

**Company Deep Dive** gives a full breakdown for any company in the list:
- A clear headline insight on how news mood and share price relate right now
- An always-on signal: moving together, diverging, or no clear link
- An interactive chart of news mood against share price
- A linear-regression "sentiment impact" estimate, how much price movement is associated with a shift in mood
- The strongest positive and negative headlines driving the score
- The latest headlines, each labelled by sentiment

## How it works

1. **News collection.** Recent headlines are pulled live from Google News for the chosen company, then de-duplicated so one widely-syndicated story does not skew the result.
2. **Sentiment scoring.** Each headline is scored by FinBERT (`ProsusAI/finbert`) as positive, negative, or neutral, with a confidence value.
3. **Confidence-weighted score.** Headlines the model is more certain about count more toward the overall score.
4. **Mood trend.** Daily sentiment is smoothed into a rolling line, with thinly-covered days excluded so single headlines do not create false swings.
5. **Price overlay.** Daily closing prices are pulled with yfinance for the same window and charted alongside the mood line using Plotly.
6. **Relationship, impact, and divergence.** The app reports the correlation between mood and price, estimates the impact with a simple regression, and flags when the two diverge.

## Skills demonstrated

- **Natural language processing**, financial sentiment classification with a transformer model (FinBERT)
- **Regression**, estimating the association between news mood and price movement
- **Data pipelines**, live data collection, cleaning, de-duplication, and time-series alignment
- **Deployment**, a working, interactive web app, not just a notebook

## Tech stack

Python, FinBERT (Hugging Face Transformers), yfinance, feedparser, Plotly, Streamlit, pandas, numpy.

## Running it locally

```bash
git clone https://github.com/Ademola-1/finance-sentiment-analyzer.git
cd finance-sentiment-analyzer
pip install -r requirements.txt
streamlit run app.py
```

## Limitations and future work

This is built as a research and learning tool, and it is honest about its edges:

- **Headline-level sentiment.** The model reads headlines, not full articles. Headlines can be vague or sensational, so the score is a useful proxy rather than a complete read of each story.
- **Short window.** The free news source returns mainly recent items, so the analysis covers a few recent weeks. Correlations over short windows are noisy and should be read as descriptive, not predictive.
- **Divergence is a heuristic.** The divergence flag uses chosen thresholds, not a statistically validated trading signal.
- **Curated company list.** The app covers a set of major, well-covered companies where news and price data are reliable.

**Planned v2:** article-level sentiment, a lead-lag analysis to test whether mood moves ahead of price, and a Nigeria edition using NGX price data and local news sources.

## Disclaimer

For research and educational use only. This is not financial advice.

## Author

**Mubarak Lawal**
Website: [mubaraklawal.com](https://mubaraklawal.com/)
