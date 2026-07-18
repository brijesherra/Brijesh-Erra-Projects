"""
Portfolio Risk Analysis — XGBoost Model
Uses yfinance to pull data, engineers risk features, and predicts
forward 20-day realized volatility using XGBoost with early stopping.
Supports user-defined tickers, dollar amounts, and recurring contributions.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─────────────────────────────────────────────
# FIXED CONFIG
# ─────────────────────────────────────────────
BENCHMARK        = "SPY"
START_DATE       = "2019-01-01"
END_DATE         = "2024-12-31"
FORWARD_WINDOW   = 20
ROLLING_SHORT    = 20
ROLLING_LONG     = 60
RISK_FREE_RATE   = 0.05 / 252
CONFIDENCE_LEVEL = 0.05

# ─────────────────────────────────────────────
# USER INPUT
# ─────────────────────────────────────────────
def get_user_inputs():
    print("\n" + "═"*50)
    print("   📈  Portfolio Risk Analyzer")
    print("═"*50 + "\n")

    # ── Tickers ──
    print("Enter stock tickers separated by commas.")
    print("Example: AAPL, MSFT, GOOGL, TSLA\n")
    while True:
        raw = input("Your tickers: ").strip()
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        if len(tickers) >= 2:
            break
        print("  ⚠️  Please enter at least 2 tickers.\n")

    # ── Allocation weights ──
    print(f"\nHow do you want to split your investment across {len(tickers)} stocks?")
    print("  1 — Equal weight (split evenly)")
    print("  2 — Custom weights (you specify %)\n")
    while True:
        choice = input("Choice (1 or 2): ").strip()
        if choice in ("1", "2"):
            break
        print("  ⚠️  Enter 1 or 2.\n")

    if choice == "1":
        weights = [1 / len(tickers)] * len(tickers)
        print(f"  ✓ Each stock gets {100/len(tickers):.1f}%")
    else:
        weights = []
        print(f"\nEnter % allocation for each ticker (must sum to 100):")
        while True:
            weights = []
            for t in tickers:
                while True:
                    try:
                        w = float(input(f"  {t} (%): ").strip())
                        if w < 0:
                            print("    ⚠️  Must be positive.")
                            continue
                        weights.append(w / 100)
                        break
                    except ValueError:
                        print("    ⚠️  Enter a number.")
            if abs(sum(weights) - 1.0) < 0.01:
                break
            print(f"\n  ⚠️  Weights sum to {sum(weights)*100:.1f}%, not 100%. Try again.\n")

    # ── Initial investment ──
    print("\n" + "─"*40)
    while True:
        try:
            initial = float(input("\nInitial investment amount ($): $").strip().replace(",", ""))
            if initial <= 0:
                print("  ⚠️  Must be greater than $0.")
                continue
            break
        except ValueError:
            print("  ⚠️  Enter a number (e.g. 10000).")

    # ── Recurring contributions ──
    print("\nDo you plan to add money regularly? (yes / no)")
    recurring_yn = input("Answer: ").strip().lower()
    recurring = 0.0
    recurring_freq = None

    if recurring_yn in ("yes", "y"):
        print("\nHow often will you contribute?")
        print("  1 — Monthly")
        print("  2 — Weekly")
        print("  3 — Annually")
        while True:
            freq_choice = input("Choice (1/2/3): ").strip()
            if freq_choice in ("1", "2", "3"):
                break
            print("  ⚠️  Enter 1, 2, or 3.")
        freq_map = {"1": "monthly", "2": "weekly", "3": "annually"}
        recurring_freq = freq_map[freq_choice]

        while True:
            try:
                recurring = float(input(f"  Amount per {recurring_freq} contribution ($): $").strip().replace(",", ""))
                if recurring <= 0:
                    print("  ⚠️  Must be greater than $0.")
                    continue
                break
            except ValueError:
                print("  ⚠️  Enter a number.")

    # ── Projection horizon ──
    print("\nHow many years do you want to project forward?")
    while True:
        try:
            horizon_years = int(input("Years (e.g. 5, 10, 20): ").strip())
            if horizon_years <= 0:
                print("  ⚠️  Must be at least 1 year.")
                continue
            break
        except ValueError:
            print("  ⚠️  Enter a whole number.")

    # ── Output path ──
    print("\nWhere should the report be saved?")
    print("Press Enter to save to your Desktop, or type a full path.\n")
    path_input = input("Save path: ").strip()
    if not path_input:
        output_path = os.path.expanduser("~/Desktop/portfolio_risk_report.png")
    else:
        output_path = path_input
    if not output_path.endswith(".png"):
        output_path += ".png"

    print("\n" + "═"*50)
    print("  Summary of your inputs:")
    print(f"  Tickers:      {', '.join(tickers)}")
    print(f"  Weights:      {[f'{w*100:.1f}%' for w in weights]}")
    print(f"  Initial:      ${initial:,.2f}")
    if recurring > 0:
        print(f"  Recurring:    ${recurring:,.2f} / {recurring_freq}")
    else:
        print(f"  Recurring:    None")
    print(f"  Horizon:      {horizon_years} years")
    print(f"  Report path:  {output_path}")
    print("═"*50 + "\n")

    input("Looks good? Press Enter to run, or Ctrl+C to cancel: ")

    return {
        "tickers":        tickers,
        "weights":        weights,
        "initial":        initial,
        "recurring":      recurring,
        "recurring_freq": recurring_freq,
        "horizon_years":  horizon_years,
        "output_path":    output_path,
    }

# ─────────────────────────────────────────────
# PHASE 1: DATA COLLECTION
# ─────────────────────────────────────────────
def fetch_data(tickers, benchmark, start, end):
    print("\n📥 Fetching price data from yfinance...")
    try:
        all_tickers = tickers + [benchmark]
        raw = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
        prices = raw["Close"].dropna()
        if len(prices) > 100:
            returns = prices.pct_change().dropna()
            print(f"   Loaded {len(prices)} trading days for {len(all_tickers)} tickers.\n")
            return prices, returns
        raise ValueError("Insufficient data from yfinance — using synthetic fallback.")
    except Exception as e:
        print(f"   yfinance unavailable ({e}). Using realistic synthetic market data.\n")
        return _generate_synthetic_data(tickers, benchmark, start, end)

def _generate_synthetic_data(tickers, benchmark, start, end):
    np.random.seed(42)
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)
    all_tickers = tickers + [benchmark]

    n_stocks = len(all_tickers)
    mu_default, sigma_default = 0.12, 0.22

    # Build a valid positive-definite correlation matrix
    rng = np.random.default_rng(42)
    # Factor model: each stock loads on a common market factor + idiosyncratic noise
    market_loadings = rng.uniform(0.4, 0.8, n_stocks)
    corr_matrix = np.outer(market_loadings, market_loadings)
    np.fill_diagonal(corr_matrix, 1.0)

    L = np.linalg.cholesky(corr_matrix)
    dt = 1 / 252
    rets_matrix = []
    for _ in all_tickers:
        mu, sigma = mu_default, sigma_default
        rets_matrix.append(np.random.normal((mu - 0.5*sigma**2)*dt, sigma*np.sqrt(dt), n))
    rets_corr = (L @ np.array(rets_matrix)).T

    crash_start = int(n * 0.25)
    rets_corr[crash_start:crash_start+30] -= 0.025

    prices_df = pd.DataFrame(100 * np.exp(np.cumsum(rets_corr, axis=0)),
                             index=dates, columns=all_tickers)
    returns_df = prices_df.pct_change().dropna()
    print(f"   Generated {len(prices_df)} synthetic trading days.\n")
    return prices_df, returns_df

# ─────────────────────────────────────────────
# PHASE 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────
def compute_features(returns, ticker, benchmark=BENCHMARK):
    df = pd.DataFrame(index=returns.index)
    r  = returns[ticker]
    m  = returns[benchmark]

    df["vol_20"]    = r.rolling(ROLLING_SHORT).std()
    df["vol_60"]    = r.rolling(ROLLING_LONG).std()
    df["vol_ratio"] = df["vol_20"] / df["vol_60"]

    def rolling_beta(r, m, window):
        cov = r.rolling(window).cov(m)
        var = m.rolling(window).var()
        return cov / var
    df["beta_20"] = rolling_beta(r, m, ROLLING_SHORT)
    df["beta_60"] = rolling_beta(r, m, ROLLING_LONG)

    df["mom_5"]  = r.rolling(5).mean()
    df["mom_20"] = r.rolling(ROLLING_SHORT).mean()

    df["sharpe_20"] = (df["mom_20"] - RISK_FREE_RATE) / df["vol_20"]

    downside = r.copy()
    downside[downside > 0] = 0
    df["sortino_20"] = (df["mom_20"] - RISK_FREE_RATE) / downside.rolling(ROLLING_SHORT).std()

    def rolling_max_drawdown(series, window):
        def mdd(x):
            peak = np.maximum.accumulate(x)
            dd   = (x - peak) / peak
            return dd.min()
        return series.rolling(window).apply(mdd, raw=True)
    df["max_dd_60"] = rolling_max_drawdown((1 + r).cumprod(), ROLLING_LONG)

    df["skew_20"]    = r.rolling(ROLLING_SHORT).skew()
    df["kurt_20"]    = r.rolling(ROLLING_SHORT).kurt()
    df["return_abs"] = r.abs()
    df["atr_20"]     = df["return_abs"].rolling(ROLLING_SHORT).mean()

    df["target_vol"] = r.shift(-FORWARD_WINDOW).rolling(FORWARD_WINDOW).std().shift(-(FORWARD_WINDOW-1))

    return df.dropna()

# ─────────────────────────────────────────────
# PHASE 3: TRAIN XGBOOST MODEL
# ─────────────────────────────────────────────
def train_model(features_df):
    feature_cols = [c for c in features_df.columns if c != "target_vol"]
    X = features_df[feature_cols].values
    y = features_df["target_vol"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)

    xgb_params = dict(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        early_stopping_rounds=40,
        eval_metric="rmse",
        random_state=42,
        verbosity=0,
    )

    cv_rmse = []
    best_iterations = []
    for train_idx, val_idx in tscv.split(X_scaled):
        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        m = XGBRegressor(**xgb_params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = m.predict(X_val)
        cv_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))
        best_iterations.append(m.best_iteration)

    final_n = max(50, int(np.median(best_iterations)))
    final_params = {k: v for k, v in xgb_params.items()
                    if k not in ("early_stopping_rounds", "eval_metric")}
    final_params["n_estimators"] = final_n

    model = XGBRegressor(**final_params)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    metrics = {
        "cv_rmse_mean": np.mean(cv_rmse),
        "cv_rmse_std":  np.std(cv_rmse),
        "r2":           r2_score(y, y_pred),
        "mae":          mean_absolute_error(y, y_pred),
        "best_n_est":   final_n,
    }

    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)

    latest_pred = model.predict(scaler.transform(X[-1].reshape(1, -1)))[0]

    return model, scaler, metrics, importances, y, y_pred, latest_pred

# ─────────────────────────────────────────────
# PHASE 4: PORTFOLIO-LEVEL RISK METRICS
# ─────────────────────────────────────────────
def portfolio_risk_metrics(returns, tickers, weights):
    r = returns[tickers]
    w = np.array(weights)
    port_returns = r.dot(w)

    ann_vol    = port_returns.std() * np.sqrt(252)
    ann_return = port_returns.mean() * 252
    sharpe     = (ann_return - 0.05) / ann_vol

    var_95  = np.percentile(port_returns, 5)
    cvar_95 = port_returns[port_returns <= var_95].mean()

    cum      = (1 + port_returns).cumprod()
    peak     = cum.cummax()
    drawdown = (cum - peak) / peak
    max_dd   = drawdown.min()

    corr = r.corr()

    return {
        "ann_return":   ann_return,
        "ann_vol":      ann_vol,
        "sharpe":       sharpe,
        "var_95":       var_95,
        "cvar_95":      cvar_95,
        "max_dd":       max_dd,
        "corr":         corr,
        "port_returns": port_returns,
        "cum_returns":  cum,
    }

# ─────────────────────────────────────────────
# PORTFOLIO PROJECTION (with recurring contributions)
# ─────────────────────────────────────────────
def project_portfolio(ann_return, ann_vol, initial, recurring,
                      recurring_freq, horizon_years):
    """
    Monte Carlo simulation of portfolio value over time.
    Simulates 1000 paths using the historical mean return and volatility,
    adding recurring contributions at the specified frequency.
    Returns percentile bands (10th, 50th, 90th) and contribution totals.
    """
    np.random.seed(0)
    N_SIMS   = 1000
    n_days   = horizon_years * 252
    daily_mu = ann_return / 252
    daily_sig = ann_vol / np.sqrt(252)

    # Contribution schedule: how many dollars added on each trading day
    contrib_per_day = np.zeros(n_days)
    if recurring > 0 and recurring_freq:
        if recurring_freq == "monthly":
            interval = 21
        elif recurring_freq == "weekly":
            interval = 5
        else:  # annually
            interval = 252
        contrib_per_day[::interval] = recurring

    # Simulate paths
    paths = np.zeros((N_SIMS, n_days + 1))
    paths[:, 0] = initial
    daily_returns = np.random.normal(daily_mu, daily_sig, (N_SIMS, n_days))

    for t in range(1, n_days + 1):
        paths[:, t] = (paths[:, t-1] * (1 + daily_returns[:, t-1])
                       + contrib_per_day[t-1])

    years = np.linspace(0, horizon_years, n_days + 1)
    p10   = np.percentile(paths, 10, axis=0)
    p50   = np.percentile(paths, 50, axis=0)
    p90   = np.percentile(paths, 90, axis=0)

    total_contributions = initial + recurring * (n_days // (
        21 if recurring_freq == "monthly" else
        5  if recurring_freq == "weekly" else
        252 if recurring_freq == "annually" else 1
    )) if recurring > 0 else initial

    return years, p10, p50, p90, total_contributions

# ─────────────────────────────────────────────
# PHASE 5: VISUALIZATIONS
# ─────────────────────────────────────────────
def plot_report(prices, returns, tickers, weights, portfolio_metrics,
                all_results, user_inputs, output_path):
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(20, 34))
    fig.suptitle("Portfolio Risk Analysis — XGBoost Model", fontsize=18, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(6, 3, figure=fig, hspace=0.48, wspace=0.35)

    colors = plt.cm.tab10.colors

    # ── 1. Cumulative Returns ──
    ax1 = fig.add_subplot(gs[0, :2])
    for i, t in enumerate(tickers):
        cum = (1 + returns[t]).cumprod()
        ax1.plot(cum, label=t, color=colors[i % 10], linewidth=1.5)
    ax1.plot(portfolio_metrics["cum_returns"], label="Portfolio", color="black",
             linewidth=2.5, linestyle="--")
    ax1.set_title("Cumulative Returns (Historical)")
    ax1.set_ylabel("Growth of $1")
    ax1.legend(fontsize=8)

    # ── 2. Correlation Heatmap ──
    ax2 = fig.add_subplot(gs[0, 2])
    sns.heatmap(portfolio_metrics["corr"], annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax2, linewidths=0.5, annot_kws={"size": 8})
    ax2.set_title("Return Correlations")

    # ── 3. Rolling Volatility ──
    ax3 = fig.add_subplot(gs[1, :2])
    for i, t in enumerate(tickers):
        rv = returns[t].rolling(20).std() * np.sqrt(252)
        ax3.plot(rv, label=t, color=colors[i % 10], linewidth=1.2, alpha=0.8)
    pv = portfolio_metrics["port_returns"].rolling(20).std() * np.sqrt(252)
    ax3.plot(pv, label="Portfolio", color="black", linewidth=2.5, linestyle="--")
    ax3.set_title("Rolling 20-Day Annualized Volatility")
    ax3.set_ylabel("Annualized Vol")
    ax3.legend(fontsize=8)

    # ── 4. Risk Metrics Summary ──
    ax4 = fig.add_subplot(gs[1, 2])
    initial   = user_inputs["initial"]
    recurring = user_inputs["recurring"]
    freq      = user_inputs["recurring_freq"]
    metrics_display = {
        "Initial Investment": f"${initial:,.0f}",
        "Recurring":          f"${recurring:,.0f}/{freq}" if recurring > 0 else "None",
        "Ann. Return":        f"{portfolio_metrics['ann_return']:.1%}",
        "Ann. Volatility":    f"{portfolio_metrics['ann_vol']:.1%}",
        "Sharpe Ratio":       f"{portfolio_metrics['sharpe']:.2f}",
        "VaR (95%, daily)":   f"{portfolio_metrics['var_95']:.2%}",
        "CVaR (95%, daily)":  f"{portfolio_metrics['cvar_95']:.2%}",
        "Max Drawdown":       f"{portfolio_metrics['max_dd']:.1%}",
    }
    ax4.axis("off")
    ax4.set_title("Portfolio Summary", fontweight="bold")
    y_pos = 0.95
    for k, v in metrics_display.items():
        color = "#d32f2f" if k in ["VaR (95%, daily)", "CVaR (95%, daily)", "Max Drawdown"] \
                else "#1a237e"
        ax4.text(0.03, y_pos, k, transform=ax4.transAxes, fontsize=10, color="gray")
        ax4.text(0.62, y_pos, v, transform=ax4.transAxes, fontsize=10,
                 fontweight="bold", color=color)
        y_pos -= 0.115

    # ── 5. ML Predictions (first 3 tickers) ──
    for i, ticker in enumerate(tickers[:3]):
        ax = fig.add_subplot(gs[2, i])
        res = all_results[ticker]
        ax.plot(res["y_true"] * np.sqrt(252), label="Actual Vol", alpha=0.7, linewidth=1)
        ax.plot(res["y_pred"] * np.sqrt(252), label="Predicted", linestyle="--", linewidth=1)
        ax.set_title(f"{ticker} — Predicted vs Actual Vol")
        ax.set_ylabel("Ann. Vol")
        ax.legend(fontsize=7)

    # ── 6. Feature Importances ──
    ax6 = fig.add_subplot(gs[3, :2])
    avg_imp = pd.concat([all_results[t]["importances"] for t in tickers], axis=1).mean(axis=1).sort_values(ascending=True)
    avg_imp.tail(12).plot(kind="barh", ax=ax6, color="steelblue")
    ax6.set_title("Avg Feature Importances (All Tickers)")
    ax6.set_xlabel("Importance")

    # ── 7. Forward Vol Forecast ──
    ax7 = fig.add_subplot(gs[3, 2])
    fwd_vols = {t: all_results[t]["latest_pred"] * np.sqrt(252) for t in tickers}
    bars = ax7.bar(list(fwd_vols.keys()), list(fwd_vols.values()),
                   color=[colors[i % 10] for i in range(len(tickers))])
    ax7.set_title(f"Predicted Forward {FORWARD_WINDOW}-Day\nAnnualized Volatility")
    ax7.set_ylabel("Annualized Vol")
    for bar, val in zip(bars, fwd_vols.values()):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # ── 8. Portfolio Projection (Monte Carlo) ──
    ax8 = fig.add_subplot(gs[4, :])
    years, p10, p50, p90, total_contrib = project_portfolio(
        portfolio_metrics["ann_return"],
        portfolio_metrics["ann_vol"],
        user_inputs["initial"],
        user_inputs["recurring"],
        user_inputs["recurring_freq"],
        user_inputs["horizon_years"],
    )
    ax8.fill_between(years, p10, p90, alpha=0.2, color="steelblue", label="10th–90th percentile")
    ax8.plot(years, p50, color="steelblue", linewidth=2.5, label="Median projection")
    ax8.plot(years, p10, color="steelblue", linewidth=1, linestyle="--", alpha=0.6)
    ax8.plot(years, p90, color="steelblue", linewidth=1, linestyle="--", alpha=0.6)
    ax8.axhline(total_contrib, color="gray", linestyle=":", linewidth=1.5,
                label=f"Total contributed: ${total_contrib:,.0f}")
    ax8.set_title(f"Portfolio Value Projection — {user_inputs['horizon_years']}-Year Monte Carlo (1,000 simulations)")
    ax8.set_xlabel("Years")
    ax8.set_ylabel("Portfolio Value ($)")
    ax8.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax8.legend(fontsize=9)

    # ── 9. Estimated Cumulative Returns Over User Horizon ──
    ax9 = fig.add_subplot(gs[5, :])
    horizon_years = user_inputs["horizon_years"]
    initial       = user_inputs["initial"]
    w             = np.array(weights)

    # Build year-by-year cumulative return bands using the median Monte Carlo paths
    # Also plot individual per-ticker estimated cumulative return curves
    ann_r   = portfolio_metrics["ann_return"]
    ann_v   = portfolio_metrics["ann_vol"]
    year_points = np.arange(0, horizon_years + 1)

    # Per-ticker estimated cumulative return (geometric, annualized from history)
    for i, t in enumerate(tickers):
        t_ann_r = returns[t].mean() * 252
        t_cum   = [(1 + t_ann_r) ** y - 1 for y in year_points]
        ax9.plot(year_points, [v * 100 for v in t_cum],
                 label=t, color=colors[i % 10], linewidth=1.5, linestyle="--", alpha=0.75)

    # Portfolio estimated cumulative return (median, 10th, 90th from MC)
    # Convert dollar paths to % cumulative return relative to initial investment only
    # (strip out contributions to isolate return effect)
    np.random.seed(1)
    N2       = 1000
    n_days2  = horizon_years * 252
    daily_mu2  = ann_r / 252
    daily_sig2 = ann_v / np.sqrt(252)
    paths2   = np.zeros((N2, n_days2 + 1))
    paths2[:, 0] = 1.0  # start at $1, no contributions — pure return
    dr2 = np.random.normal(daily_mu2, daily_sig2, (N2, n_days2))
    for t_idx in range(1, n_days2 + 1):
        paths2[:, t_idx] = paths2[:, t_idx - 1] * (1 + dr2[:, t_idx - 1])

    # Sample at yearly intervals
    yearly_idx = [int(y * 252) for y in year_points]
    p10_r  = (np.percentile(paths2[:, yearly_idx], 10, axis=0) - 1) * 100
    p50_r  = (np.percentile(paths2[:, yearly_idx], 50, axis=0) - 1) * 100
    p90_r  = (np.percentile(paths2[:, yearly_idx], 90, axis=0) - 1) * 100

    ax9.fill_between(year_points, p10_r, p90_r, alpha=0.15, color="black",
                     label="Portfolio 10th–90th %ile")
    ax9.plot(year_points, p50_r, color="black", linewidth=2.5,
             label="Portfolio median")
    ax9.plot(year_points, p10_r, color="black", linewidth=1, linestyle=":", alpha=0.5)
    ax9.plot(year_points, p90_r, color="black", linewidth=1, linestyle=":", alpha=0.5)
    ax9.axhline(0, color="red", linewidth=1, linestyle="--", alpha=0.4)

    # Annotate final median value
    ax9.annotate(f"+{p50_r[-1]:.0f}% (median)",
                 xy=(horizon_years, p50_r[-1]),
                 xytext=(-40, 10), textcoords="offset points",
                 fontsize=9, fontweight="bold", color="black",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1))

    ax9.set_title(f"Estimated Cumulative Returns Over {horizon_years}-Year Horizon  |  Dashed = individual tickers, Black = portfolio Monte Carlo bands")
    ax9.set_xlabel("Years")
    ax9.set_ylabel("Cumulative Return (%)")
    ax9.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:+.0f}%"))
    ax9.legend(fontsize=8, ncol=min(len(tickers) + 2, 5))
    ax9.set_xticks(year_points)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n   Report saved → {output_path}")
    plt.close()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    user = get_user_inputs()
    tickers = user["tickers"]
    weights = user["weights"]

    prices, returns = fetch_data(tickers, BENCHMARK, START_DATE, END_DATE)

    # Drop any tickers that didn't load
    valid_tickers = [t for t in tickers if t in returns.columns]
    if len(valid_tickers) < len(tickers):
        missing = set(tickers) - set(valid_tickers)
        print(f"  ⚠️  Could not load data for: {missing}. Continuing with {valid_tickers}.\n")
        # Re-normalize weights for valid tickers only
        valid_idx = [tickers.index(t) for t in valid_tickers]
        weights = [weights[i] for i in valid_idx]
        total = sum(weights)
        weights = [w / total for w in weights]
        tickers = valid_tickers

    print("⚙️  Engineering features & training models...\n")
    all_results = {}
    for ticker in tickers:
        features_df = compute_features(returns, ticker)
        model, scaler, metrics, importances, y_true, y_pred, latest_pred = train_model(features_df)
        all_results[ticker] = {
            "model":       model,
            "metrics":     metrics,
            "importances": importances,
            "y_true":      y_true,
            "y_pred":      y_pred,
            "latest_pred": latest_pred,
        }
        print(f"  {ticker}:")
        print(f"    CV RMSE:      {metrics['cv_rmse_mean']:.5f} ± {metrics['cv_rmse_std']:.5f}")
        print(f"    R²:           {metrics['r2']:.3f}")
        print(f"    Best n_est:   {metrics['best_n_est']} (via early stopping)")
        print(f"    Fwd Vol Est:  {latest_pred * np.sqrt(252):.2%} annualized\n")

    print("📊 Computing portfolio-level risk metrics...")
    port_metrics = portfolio_risk_metrics(returns, tickers, weights)

    print("\n── Portfolio Summary ──────────────────────")
    print(f"  Initial investment: ${user['initial']:,.2f}")
    if user['recurring'] > 0:
        print(f"  Recurring:          ${user['recurring']:,.2f} / {user['recurring_freq']}")
    print(f"  Annualized Return:  {port_metrics['ann_return']:.2%}")
    print(f"  Annualized Vol:     {port_metrics['ann_vol']:.2%}")
    print(f"  Sharpe Ratio:       {port_metrics['sharpe']:.2f}")
    print(f"  VaR (95%, daily):   {port_metrics['var_95']:.2%}")
    print(f"  CVaR (95%, daily):  {port_metrics['cvar_95']:.2%}")
    print(f"  Max Drawdown:       {port_metrics['max_dd']:.2%}")
    print("────────────────────────────────────────────\n")

    print("🖼️  Generating risk report...")
    plot_report(prices, returns, tickers, weights, port_metrics,
                all_results, user, user["output_path"])

    print("\n✅ Done! Report saved to:", user["output_path"])

if __name__ == "__main__":
    main()