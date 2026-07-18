"""
stock_analyzer.py  — v3 (ML + Gemini AI edition)
-------------------------------------------------
Three AI/ML layers on top of the original rule-based scorer:

  1. Price Prediction   — RandomForestRegressor trained on 2 years of
                          technical features to forecast 5-day return.
  2. Sentiment Analysis — Pulls recent Yahoo Finance headlines, scores
                          each with VADER, and aggregates a sentiment signal.
  3. Gemini AI Summary  — Sends all metrics + scores to Google Gemini
                          (free tier) and gets a plain-English analysis back.

Requirements:
    pip install yfinance pandas numpy colorama scikit-learn \
                vaderSentiment google-generativeai requests beautifulsoup4

Free Gemini API key → https://aistudio.google.com/app/apikey
Environment variable:
    export GEMINI_API_KEY="AIza..."
"""

import os
import sys
import textwrap
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from colorama import init, Fore, Style
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai

init(autoreset=True)

# ─────────────────────────────────────────────
#  WEIGHTS
# ─────────────────────────────────────────────
WEIGHTS = {
    "pe_ratio":       0.12,
    "debt_to_equity": 0.10,
    "current_ratio":  0.08,
    "profit_margin":  0.08,
    "revenue_growth": 0.07,
    "rsi":            0.15,
    "ma_trend":       0.15,
    "volatility":     0.13,
    "momentum":       0.12,
}

# ─────────────────────────────────────────────
#  METRIC SCORERS  (0–10, 10 = safest)
# ─────────────────────────────────────────────

def score_pe(pe):
    if pe is None or np.isnan(pe): return 5, "N/A"
    if pe <= 0:  return 1, f"{pe:.1f} (negative earnings)"
    if pe < 15:  return 9, f"{pe:.1f} (undervalued)"
    if pe < 25:  return 7, f"{pe:.1f} (fair)"
    if pe < 40:  return 4, f"{pe:.1f} (elevated)"
    return 2, f"{pe:.1f} (very high)"

def score_debt_to_equity(de):
    if de is None or np.isnan(de): return 5, "N/A"
    if de < 0.5: return 9, f"{de:.2f} (low debt)"
    if de < 1.0: return 7, f"{de:.2f} (moderate)"
    if de < 2.0: return 4, f"{de:.2f} (high)"
    return 2, f"{de:.2f} (very high)"

def score_current_ratio(cr):
    if cr is None or np.isnan(cr): return 5, "N/A"
    if cr >= 2.0: return 9, f"{cr:.2f} (strong liquidity)"
    if cr >= 1.5: return 7, f"{cr:.2f} (healthy)"
    if cr >= 1.0: return 5, f"{cr:.2f} (acceptable)"
    return 2, f"{cr:.2f} (liquidity risk)"

def score_profit_margin(pm):
    if pm is None or np.isnan(pm): return 5, "N/A"
    pct = pm * 100
    if pct >= 20: return 9, f"{pct:.1f}%"
    if pct >= 10: return 7, f"{pct:.1f}%"
    if pct >= 0:  return 5, f"{pct:.1f}%"
    return 2, f"{pct:.1f}% (loss)"

def score_revenue_growth(rg):
    if rg is None or np.isnan(rg): return 5, "N/A"
    pct = rg * 100
    if pct >= 20: return 9, f"{pct:.1f}%"
    if pct >= 10: return 7, f"{pct:.1f}%"
    if pct >= 0:  return 5, f"{pct:.1f}%"
    return 2, f"{pct:.1f}% (declining)"

def score_rsi(rsi):
    if rsi is None or np.isnan(rsi): return 5, "N/A"
    if 40 <= rsi <= 60:  return 8, f"{rsi:.1f} (neutral)"
    if 30 <= rsi < 40:   return 6, f"{rsi:.1f} (slightly oversold)"
    if 60 < rsi <= 70:   return 6, f"{rsi:.1f} (slightly overbought)"
    if rsi < 30:         return 3, f"{rsi:.1f} (oversold)"
    return 3, f"{rsi:.1f} (overbought)"

def score_ma_trend(price, ma50, ma200):
    if any(v is None or np.isnan(v) for v in [price, ma50, ma200]): return 5, "N/A"
    if price > ma50 > ma200: return 9, "Bullish (price > MA50 > MA200)"
    if price > ma200:        return 7, "Price above MA200"
    if price > ma50:         return 5, "Price above MA50 only"
    return 3, "Price below both MAs (bearish)"

def score_volatility(ann_vol):
    if ann_vol is None or np.isnan(ann_vol): return 5, "N/A"
    pct = ann_vol * 100
    if pct < 15: return 9, f"{pct:.1f}% (low)"
    if pct < 30: return 7, f"{pct:.1f}% (moderate)"
    if pct < 50: return 4, f"{pct:.1f}% (high)"
    return 2, f"{pct:.1f}% (very high)"

def score_momentum(ret_30d):
    if ret_30d is None or np.isnan(ret_30d): return 5, "N/A"
    pct = ret_30d * 100
    if pct >= 10:  return 8, f"+{pct:.1f}% (strong)"
    if pct >= 0:   return 6, f"+{pct:.1f}% (positive)"
    if pct >= -10: return 4, f"{pct:.1f}% (slight pullback)"
    return 2, f"{pct:.1f}% (sharp decline)"


# ─────────────────────────────────────────────
#  DATA FETCHER
# ─────────────────────────────────────────────

def fetch_data(ticker: str):
    t    = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period="2y")
    if hist.empty:
        print(Fore.RED + f"  ✗ No price history found for '{ticker}'.")
        sys.exit(1)

    closes    = hist["Close"]
    price     = float(closes.iloc[-1])
    ma50      = float(closes.rolling(50).mean().iloc[-1])  if len(closes) >= 50  else float("nan")
    ma200     = float(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else float("nan")
    daily_ret = closes.pct_change().dropna()
    ann_vol   = float(daily_ret.std() * np.sqrt(252))
    ret_30d   = float((closes.iloc[-1] - closes.iloc[-22]) / closes.iloc[-22]) if len(closes) >= 22 else float("nan")

    delta = closes.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    rsi   = float(100 - 100 / (1 + rs.iloc[-1]))

    def safe(key):
        v = info.get(key)
        return float(v) if v is not None else float("nan")

    de_raw = safe("debtToEquity")

    return {
        "name":           info.get("longName", ticker.upper()),
        "sector":         info.get("sector", "N/A"),
        "price":          price,
        "ma50":           ma50,
        "ma200":          ma200,
        "ann_vol":        ann_vol,
        "ret_30d":        ret_30d,
        "rsi":            rsi,
        "closes":         closes,
        "pe_ratio":       safe("trailingPE"),
        "debt_to_equity": de_raw / 100 if not np.isnan(de_raw) else float("nan"),
        "current_ratio":  safe("currentRatio"),
        "profit_margin":  safe("profitMargins"),
        "revenue_growth": safe("revenueGrowth"),
    }


# ─────────────────────────────────────────────
#  ML MODULE 1 — PRICE PREDICTION
#  RandomForest trained on lagged technical
#  features to predict 5-day forward return.
# ─────────────────────────────────────────────

def build_features(closes: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"close": closes})
    df["ret1"]    = df["close"].pct_change(1)
    df["ret5"]    = df["close"].pct_change(5)
    df["ret10"]   = df["close"].pct_change(10)
    df["ret20"]   = df["close"].pct_change(20)
    df["ma10"]    = df["close"].rolling(10).mean()
    df["ma20"]    = df["close"].rolling(20).mean()
    df["ma50"]    = df["close"].rolling(50).mean()
    df["vol10"]   = df["ret1"].rolling(10).std()
    df["vol20"]   = df["ret1"].rolling(20).std()
    # RSI-14
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    # Price relative to MAs
    df["p_ma10"] = df["close"] / df["ma10"] - 1
    df["p_ma20"] = df["close"] / df["ma20"] - 1
    df["p_ma50"] = df["close"] / df["ma50"] - 1
    # Target: 5-day forward return
    df["target"] = df["close"].pct_change(5).shift(-5)
    return df.dropna()


def ml_price_prediction(closes: pd.Series):
    """
    Train a RandomForest on historical features → predict next-5-day return.
    Returns (predicted_return_pct, direction_label, confidence_str).
    """
    df = build_features(closes)
    if len(df) < 60:
        return None, "Insufficient data", "N/A"

    feature_cols = [c for c in df.columns if c not in ["close", "target"]]
    X = df[feature_cols].values
    y = df["target"].values

    # Train on all but last 20 days; test on last 20 for a quick accuracy check
    split = len(X) - 20
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Directional accuracy on holdout
    preds_test   = model.predict(X_test)
    dir_accuracy = float(np.mean(np.sign(preds_test) == np.sign(y_test)) * 100)

    # Predict on the latest row
    X_latest   = scaler.transform(df[feature_cols].iloc[[-1]].values)
    pred_return = float(model.predict(X_latest)[0]) * 100  # as %

    if pred_return >= 1.5:
        direction = "Bullish ↑"
    elif pred_return <= -1.5:
        direction = "Bearish ↓"
    else:
        direction = "Neutral →"

    confidence = f"Model dir. accuracy (20-day holdout): {dir_accuracy:.0f}%"
    return pred_return, direction, confidence


# ─────────────────────────────────────────────
#  ML MODULE 2 — SENTIMENT ANALYSIS
#  Scrape Yahoo Finance headlines for ticker,
#  score with VADER, aggregate signal.
# ─────────────────────────────────────────────

def fetch_headlines(ticker: str) -> list[str]:
    url     = f"https://finance.yahoo.com/quote/{ticker}/news/"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        soup = BeautifulSoup(resp.text, "html.parser")
        headlines = []
        for tag in soup.find_all(["h3", "h2"], limit=30):
            text = tag.get_text(strip=True)
            if len(text) > 20:
                headlines.append(text)
        return headlines[:15]
    except Exception:
        return []


def sentiment_analysis(ticker: str):
    """
    Returns (compound_score, label, headlines_used).
    compound_score: -1 (very negative) to +1 (very positive)
    """
    analyzer  = SentimentIntensityAnalyzer()
    headlines = fetch_headlines(ticker)

    if not headlines:
        return 0.0, "No headlines found", []

    scores = [analyzer.polarity_scores(h)["compound"] for h in headlines]
    avg    = float(np.mean(scores))

    if avg >= 0.15:
        label = "Positive 📈"
    elif avg <= -0.15:
        label = "Negative 📉"
    else:
        label = "Neutral 😐"

    return avg, label, headlines


# ─────────────────────────────────────────────
#  AI MODULE 3 — CLAUDE SUMMARY
# ─────────────────────────────────────────────

def claude_summary(ticker, name, sector, price, scores, final_score,
                   verdict, pred_return, pred_direction,
                   sentiment_score, sentiment_label, headlines):
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "(Set ANTHROPIC_API_KEY environment variable to enable AI summary.)"

    metric_lines = "\n".join(
        f"  - {k}: score {v[0]}/10, value: {v[1]}"
        for k, v in scores.items()
    )
    headline_block = "\n".join(f"  • {h}" for h in headlines[:8]) or "  (none available)"

    prompt = f"""
You are a financial analyst assistant. A user asked for an assessment of the stock {ticker} ({name}).

Here is the data:

COMPANY: {name} | Ticker: {ticker} | Sector: {sector}
Current Price: ${price:.2f}

RULE-BASED METRIC SCORES (0=risky, 10=safe):
{metric_lines}

Composite Score: {final_score:.2f}/10
Rule-Based Verdict: {verdict}

ML PRICE PREDICTION (5-day forward):
  Predicted return: {pred_return:.2f}%
  Direction: {pred_direction}

NEWS SENTIMENT:
  Aggregate VADER score: {sentiment_score:.3f} (-1 to +1)
  Sentiment label: {sentiment_label}
  Recent headlines:
{headline_block}

Write a concise 3-4 paragraph investment analysis in plain English. Cover:
1. What the fundamental metrics say about the company's financial health.
2. What the technical indicators and ML prediction suggest about near-term price action.
3. How news sentiment might be affecting the stock right now.
4. A final balanced summary of key risks and strengths.

Be direct. Do not disclaim that you are an AI. Do not give a buy/sell recommendation.
Keep the tone like a professional analyst report.
"""

    try:
        client   = anthropic.Anthropic(api_key=api_key)
        message  = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 800,
            messages   = [{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        return f"(Claude API error: {e})"


# ─────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────

BAR_FULL  = "█"
BAR_EMPTY = "░"

def score_bar(score, width=20):
    filled = int(round(score / 10 * width))
    bar    = BAR_FULL * filled + BAR_EMPTY * (width - filled)
    color  = Fore.GREEN if score >= 7 else (Fore.YELLOW if score >= 4 else Fore.RED)
    return color + bar + Style.RESET_ALL + f"  {score:.1f}/10"

def verdict_label(final_score):
    if final_score >= 6.5: return Fore.GREEN,  "SAFE",    "✔", "Metrics suggest relatively low risk."
    if final_score >= 4.5: return Fore.YELLOW, "NEUTRAL", "~", "Mixed signals. Proceed with caution."
    return Fore.RED, "RISKY", "✘", "Multiple red flags detected. High risk profile."

def section(title):
    print(Fore.CYAN + Style.BRIGHT + f"\n  ── {title} " + "─" * (52 - len(title)))


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def analyze(ticker: str):
    print(Fore.CYAN + Style.BRIGHT + f"\n  Fetching data for {ticker.upper()} …")

    d = fetch_data(ticker)

    scores = {
        "pe_ratio":       score_pe(d["pe_ratio"]),
        "debt_to_equity": score_debt_to_equity(d["debt_to_equity"]),
        "current_ratio":  score_current_ratio(d["current_ratio"]),
        "profit_margin":  score_profit_margin(d["profit_margin"]),
        "revenue_growth": score_revenue_growth(d["revenue_growth"]),
        "rsi":            score_rsi(d["rsi"]),
        "ma_trend":       score_ma_trend(d["price"], d["ma50"], d["ma200"]),
        "volatility":     score_volatility(d["ann_vol"]),
        "momentum":       score_momentum(d["ret_30d"]),
    }

    final_score = sum(scores[k][0] * WEIGHTS[k] for k in WEIGHTS)
    color, verdict, emoji, advice = verdict_label(final_score)

    # ── Header ──────────────────────────────────────────────────────────
    print(Style.BRIGHT + "\n" + "=" * 58)
    print(f"  {d['name']}  ({ticker.upper()})")
    print(f"  Sector: {d['sector']}   |   Price: ${d['price']:.2f}")
    print(Style.BRIGHT + "=" * 58)

    # ── Metric table ────────────────────────────────────────────────────
    section("FUNDAMENTAL + TECHNICAL METRICS")
    col_labels = {
        "pe_ratio":       "P/E Ratio",
        "debt_to_equity": "Debt / Equity",
        "current_ratio":  "Current Ratio",
        "profit_margin":  "Profit Margin",
        "revenue_growth": "Revenue Growth (YoY)",
        "rsi":            "RSI (14)",
        "ma_trend":       "MA Trend (50/200)",
        "volatility":     "Annualised Volatility",
        "momentum":       "30-Day Momentum",
    }
    print(f"\n  {'METRIC':<24} {'VALUE':<30} SCORE")
    print("  " + "-" * 56)
    for key, lbl in col_labels.items():
        sc, val_str = scores[key]
        print(f"  {lbl:<24} {val_str:<30} {score_bar(sc)}")

    # ── ML: Price Prediction ─────────────────────────────────────────────
    section("ML — 5-DAY PRICE PREDICTION  (RandomForest)")
    print(Fore.YELLOW + "  Training model on 2 years of technical features …")
    pred_return, pred_direction, confidence = ml_price_prediction(d["closes"])
    if pred_return is not None:
        ret_color = Fore.GREEN if pred_return >= 0 else Fore.RED
        print(f"  Predicted 5-day return : {ret_color}{pred_return:+.2f}%{Style.RESET_ALL}")
        print(f"  Direction signal       : {pred_direction}")
        print(f"  {Fore.CYAN}{confidence}{Style.RESET_ALL}")
    else:
        print(f"  {pred_direction}")

    # ── Sentiment Analysis ───────────────────────────────────────────────
    section("NLP — NEWS SENTIMENT  (VADER)")
    print(Fore.YELLOW + "  Scraping headlines and scoring sentiment …")
    sent_score, sent_label, headlines = sentiment_analysis(ticker)
    sent_color = Fore.GREEN if sent_score >= 0.15 else (Fore.RED if sent_score <= -0.15 else Fore.YELLOW)
    print(f"  Sentiment label  : {sent_color}{sent_label}{Style.RESET_ALL}")
    print(f"  Aggregate score  : {sent_color}{sent_score:+.3f}{Style.RESET_ALL}  (–1 = very negative, +1 = very positive)")
    if headlines:
        print(f"\n  Recent headlines ({len(headlines)} found):")
        for h in headlines[:6]:
            print(f"    • {h[:80]}")
    else:
        print("  No headlines retrieved.")

    # ── Composite Verdict ────────────────────────────────────────────────
    print("\n" + Style.BRIGHT + "=" * 58)
    print(f"\n  Rule-Based Composite Score : {final_score:.2f} / 10")
    print(f"\n  Verdict : {color}{Style.BRIGHT}[ {emoji} {verdict} ]{Style.RESET_ALL}")
    print(f"  {advice}")
    print(Style.BRIGHT + "\n" + "=" * 58)

    # ── Claude AI Summary ────────────────────────────────────────────────
    section("AI ANALYSIS  (Claude)")
    print(Fore.YELLOW + "  Generating analysis …\n")
    summary = claude_summary(
        ticker, d["name"], d["sector"], d["price"],
        scores, final_score, verdict,
        pred_return or 0.0, pred_direction,
        sent_score, sent_label, headlines
    )
    for para in summary.split("\n"):
        wrapped = textwrap.fill(para.strip(), width=72, initial_indent="  ", subsequent_indent="  ")
        if wrapped.strip():
            print(wrapped)
        else:
            print()

    print(Style.BRIGHT + "\n" + "=" * 58)
    print(Fore.RED + "\n  ⚠  Not financial advice. Always do your own research.\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        ticker = input(Fore.CYAN + "  Enter ticker symbol: ").strip().upper()
    else:
        ticker = sys.argv[1].upper()

    if not ticker:
        print(Fore.RED + "  No ticker provided. Exiting.")
        sys.exit(1)

    analyze(ticker)