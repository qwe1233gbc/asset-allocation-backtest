"""
Generate backtest_data.json for the asset allocation backtest tool.

Sources (akshare):
  - index_us_stock_sina() for SP500 (.INX) and Nasdaq Composite (.IXIC)
  - index_zh_a_hist() for CSI Dividend LowVol (000015)

Output: backtest_data.json with monthly data, 231 pre-computed portfolios,
and 5Y/10Y rolling window annualized returns.
"""

import json
import time
import akshare as ak
import pandas as pd
import numpy as np

# ── Config ──
SP500_SYMBOL = ".INX"
NASDAQ_SYMBOL = ".IXIC"        # proxy for Nasdaq 100
DIVLOW_SYMBOL = "000015"       # CSI Dividend LowVol
DIV_SP500 = 0.018 / 12         # monthly dividend yield
DIV_NASDAQ = 0.007 / 12
DIV_DIVLOW = 0.038 / 12
OUTPUT = "backtest_data.json"


def fetch_index_data():
    """Fetch price data for all three indices."""
    print("Fetching SP500...")
    sp500 = ak.index_us_stock_sina(symbol=SP500_SYMBOL)
    sp500 = sp500[["date", "close"]].copy()
    sp500.columns = ["date", "sp500_price"]
    sp500["date"] = pd.to_datetime(sp500["date"])

    time.sleep(1)
    print("Fetching Nasdaq Composite...")
    nasdaq = ak.index_us_stock_sina(symbol=NASDAQ_SYMBOL)
    nasdaq = nasdaq[["date", "close"]].copy()
    nasdaq.columns = ["date", "nasdaq_price"]
    nasdaq["date"] = pd.to_datetime(nasdaq["date"])

    time.sleep(1)
    print("Fetching CSI Dividend LowVol...")
    divlow = ak.index_zh_a_hist(symbol=DIVLOW_SYMBOL, period="monthly", start_date="20000101", end_date="20301231")
    divlow = divlow[["日期", "收盘"]].copy()
    divlow.columns = ["date", "divlovol_price"]
    divlow["date"] = pd.to_datetime(divlow["date"])

    # Merge
    df = sp500.merge(nasdaq, on="date", how="inner")
    df = df.merge(divlow, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)

    # Monthly resample
    df = df.set_index("date").resample("ME").last().dropna().reset_index()
    print(f"Merged data: {len(df)} months, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def compute_returns(df):
    """Compute total returns (price + dividend)."""
    prices = df.copy()
    for col, div_yield in [("sp500", DIV_SP500), ("nasdaq", DIV_NASDAQ), ("divlovol", DIV_DIVLOW)]:
        price_ret = prices[f"{col}_price"].pct_change()
        prices[f"{col}_tr"] = price_ret + div_yield

    prices = prices.dropna().reset_index(drop=True)

    # Cumulative returns
    for col in ["sp500", "nasdaq", "divlovol"]:
        prices[f"{col}_cum"] = (1 + prices[f"{col}_tr"]).cumprod()

    return prices


def build_monthly_data(prices):
    """Convert to list of dicts for JSON."""
    return prices[["date", "sp500_price", "nasdaq_price", "divlovol_price",
                    "sp500_tr", "nasdaq_tr", "divlovol_tr",
                    "sp500_cum", "nasdaq_cum", "divlovol_cum"]].to_dict(orient="records")


def portfolio_return(weights, monthly_rets):
    """Compute metrics for a given allocation."""
    sp_w, nq_w, dv_w = weights
    rets = sp_w * monthly_rets["sp500_tr"] + nq_w * monthly_rets["nasdaq_tr"] + dv_w * monthly_rets["divlovol_tr"]
    cum = (1 + rets).prod()
    years = len(rets) / 12
    ann_ret = cum ** (1 / years) - 1
    ann_vol = rets.std() * np.sqrt(12)
    cum_series = (1 + rets).cumprod()
    peak = cum_series.cummax()
    mdd = ((cum_series - peak) / peak).min()
    sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
    return {"ret": round(float(ann_ret), 8), "vol": round(float(ann_vol), 8),
            "mdd": round(float(mdd), 8), "sharpe": round(float(sharpe), 8)}


def compute_all_portfolios(prices):
    """Compute 231 portfolios (5% steps, 3 assets)."""
    portfolios = {}
    monthly_rets = prices[["sp500_tr", "nasdaq_tr", "divlovol_tr"]]
    for sp in range(0, 101, 5):
        for nq in range(0, 101 - sp, 5):
            dv = 100 - sp - nq
            key = f"{sp}_{nq}_{dv}"
            metrics = portfolio_return((sp/100, nq/100, dv/100), monthly_rets)
            portfolios[key] = {"sp": sp, "nq": nq, "dv": dv, **metrics}
    print(f"Computed {len(portfolios)} portfolios")
    return portfolios


def compute_rolling_returns(prices, portfolios, window_years):
    """Compute rolling window annualized returns for each portfolio."""
    window_months = window_years * 12
    monthly_rets = prices[["sp500_tr", "nasdaq_tr", "divlovol_tr"]]
    n = len(prices)

    result = {}
    for key, alloc in portfolios.items():
        w = (alloc["sp"] / 100, alloc["nq"] / 100, alloc["dv"] / 100)
        rets = w[0] * monthly_rets["sp500_tr"] + w[1] * monthly_rets["nasdaq_tr"] + w[2] * monthly_rets["divlovol_tr"]
        windows = []
        for i in range(0, n - window_months + 1, 3):  # every 3 months
            slice_rets = rets.iloc[i:i + window_months]
            cum = (1 + slice_rets).prod()
            ann_ret = cum ** (1 / window_years) - 1
            windows.append({
                "start": str(prices["date"].iloc[i].date()),
                "end": str(prices["date"].iloc[i + window_months - 1].date()),
                "ann_ret": round(float(ann_ret), 8)
            })
        result[key] = windows

    n_windows = len(windows) if windows else 0
    print(f"Computed {window_years}Y rolling: {n_windows} windows x {len(portfolios)} portfolios")
    return result


def main():
    print("=" * 60)
    print("Asset Allocation Backtest Data Generator")
    print("=" * 60)

    # Fetch data
    prices = fetch_index_data()

    # Compute returns
    prices = compute_returns(prices)

    # Build output
    output = {
        "metadata": {
            "data_start": str(prices["date"].min().date()),
            "data_end": str(prices["date"].max().date()),
            "n_months": len(prices),
            "n_years": round(len(prices) / 12, 1),
            "sp500_div_yield": DIV_SP500 * 12,
            "nasdaq_div_yield": DIV_NASDAQ * 12,
            "divlovol_div_yield": DIV_DIVLOW * 12,
            "nasdaq_note": "Using Nasdaq Composite (.IXIC) as Nasdaq 100 proxy (correlation >0.95)",
            "methodology": "Monthly total return = price return + estimated monthly dividend yield (dividend reinvestment)"
        },
        "monthly_data": build_monthly_data(prices),
        "portfolios": compute_all_portfolios(prices),
        "rolling_windows": {}
    }

    # Rolling windows
    output["rolling_windows"]["5Y"] = compute_rolling_returns(prices, output["portfolios"], 5)
    output["rolling_windows"]["10Y"] = compute_rolling_returns(prices, output["portfolios"], 10)

    # Write
    with open(OUTPUT, "w") as f:
        json.dump(output, f)
    print(f"\nDone. Saved to: {OUTPUT}")
    print(f"File size: {len(json.dumps(output)) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
