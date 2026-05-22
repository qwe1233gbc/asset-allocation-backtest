"""
Generate enhanced backtest data with:
  1. Real fund NAV data (includes actual dividends and mgmt/custody fees)
  2. Fee model for index-derived data
  3. Hybrid approach: fund NAV where available, index+fees for earlier periods

Funds:
  - SP500: 博时标普500ETF联接A (050025), since 2012-06-14
  - Nasdaq: 广发纳斯达克100ETF联接A (270042), since 2012-08-15
  - DivLowVol: 红利低波ETF (512890), since 2018-12-19 (shorter, use index+fees for earlier)
"""

import json, time, os
import akshare as ak
import pandas as pd
import numpy as np
import urllib.request

# ── Anti-proxy patch for Chinese network environment ──
for _key in list(os.environ.keys()):
    if 'proxy' in _key.lower():
        os.environ.pop(_key, None)

# Monkey-patch akshare's requests session to disable proxy
import requests as _requests
_original_session = _requests.Session
class _NoProxySession(_original_session):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trust_env = False
_requests.Session = _NoProxySession
# Also patch any existing session classes in akshare
try:
    import requests.sessions
    requests.sessions.Session = _NoProxySession
except:
    pass

# ── Config ──
SP500_SYMBOL = ".INX"
NASDAQ_SYMBOL = ".IXIC"
DIVLOW_SYMBOL = "000015"

SP500_FUND = "050025"    # 博时标普500ETF联接A
NASDAQ_FUND = "270042"   # 广发纳斯达克100ETF联接A
DIVLOW_ETF = "512890"    # 红利低波ETF

# Fee model (Alipay QDII fund rates)
FEE_MODEL = {
    "management_fee_annual": 0.006,    # 0.6%/year
    "custody_fee_annual": 0.0025,      # 0.25%/year (QDII custody)
    "total_annual_drag": 0.0085,       # combined 0.85%/year drag on index
    "monthly_drag": 0.0085 / 12,       # ~0.071%/month
    "subscription_fee": 0.0012,        # 0.12% (Alipay 1折)
    "redemption_tiers": [              # [holding_days, fee_rate]
        [7, 0.015],                    # <7 days: 1.5%
        [365, 0.005],                  # 7d-1yr: 0.5%
        [99999, 0.0]                   # >1yr: free
    ],
    "note": "QDII index fund avg rates on Alipay. Mgmt+custody deducted daily from NAV. Subscription at 90% discount."
}

OUTPUT = "backtest_data_v2.json"


def fetch_divlow_direct():
    """Fetch CSI Dividend LowVol via direct East Money API (bypass proxy)."""
    url = ('https://push2his.eastmoney.com/api/qt/stock/kline/get'
           '?secid=1.000015'
           '&fields1=f1,f2,f3,f4,f5,f6'
           '&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'
           '&klt=101&fqt=1&beg=20000101&end=20260522')
    req = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://data.eastmoney.com/',
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    klines = data['data']['klines']
    rows = [{'date': k.split(',')[0], 'close': float(k.split(',')[2])} for k in klines]
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    return df


def fetch_index_data():
    """Fetch index price data for all three indices."""
    print("Fetching SP500 index...")
    sp500 = ak.index_us_stock_sina(symbol=SP500_SYMBOL)
    sp500 = sp500[["date", "close"]].copy()
    sp500.columns = ["date", "sp500_price"]

    time.sleep(1)
    print("Fetching Nasdaq Composite...")
    nasdaq = ak.index_us_stock_sina(symbol=NASDAQ_SYMBOL)
    nasdaq = nasdaq[["date", "close"]].copy()
    nasdaq.columns = ["date", "nasdaq_price"]

    time.sleep(1)
    print("Fetching CSI Dividend LowVol...")
    try:
        divlow = ak.index_zh_a_hist(symbol=DIVLOW_SYMBOL, period="monthly",
                                     start_date="20000101", end_date="20301231")
        divlow = divlow[["日期", "收盘"]].copy()
        divlow.columns = ["date", "divlow_price"]
        divlow["date"] = pd.to_datetime(divlow["date"])
    except Exception as e:
        print(f"  akshare failed ({e}), using direct API fallback...")
        divlow = fetch_divlow_direct()
        divlow.columns = ["date", "divlow_price"]

    # Merge
    df = sp500.merge(nasdaq, on="date", how="inner")
    df = df.merge(divlow, on="date", how="inner")
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Index data: {len(df)} days, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def fetch_fund_nav(fund_code, fund_name):
    """Fetch fund NAV history."""
    print(f"Fetching {fund_name} ({fund_code})...")
    try:
        nav = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        # Columns from akshare: 净值日期, 单位净值, 日增长率
        nav.columns = ["date", "nav", "daily_return"]
        nav["date"] = pd.to_datetime(nav["date"])
        nav = nav.sort_values("date").reset_index(drop=True)
        print(f"  {len(nav)} trading days, {nav['date'].min().date()} to {nav['date'].max().date()}")
        return nav
    except Exception as e:
        print(f"  Error: {e}")
        return None


def compute_monthly_returns(daily_df, price_cols, date_col="date"):
    """Resample daily data to monthly and compute returns."""
    df = daily_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)

    # Resample to month-end
    monthly = df.resample("ME").last().dropna()

    # Compute returns
    for col in price_cols:
        monthly[f"{col}_ret"] = monthly[col].pct_change()

    monthly = monthly.dropna().reset_index()
    monthly[date_col] = monthly["date"].dt.date
    return monthly


def build_hybrid_dataset(index_daily, sp500_fund_nav, nasdaq_fund_nav, divlow_etf_nav):
    """Build monthly dataset combining fund NAVs and index data."""
    # Step 1: Create monthly index returns (for the full period)
    idx_m = index_daily.set_index("date")
    idx_m = idx_m.resample("ME").last().dropna().reset_index()
    idx_m["date"] = pd.to_datetime(idx_m["date"])

    # Step 2: Fund NAV monthly
    for nav_df, prefix, fund_code in [
        (sp500_fund_nav, "sp500", SP500_FUND),
        (nasdaq_fund_nav, "nasdaq", NASDAQ_FUND),
        (divlow_etf_nav, "divlow", DIVLOW_ETF)
    ]:
        if nav_df is not None:
            nav_m = nav_df.set_index("date").resample("ME").last()["nav"].dropna().reset_index()
        else:
            nav_m = pd.DataFrame(columns=["date", "nav"])

        if not nav_m.empty:
            # Compute monthly total return from fund NAV
            nav_m[f"{prefix}_fund_nav"] = nav_m["nav"]
            nav_m[f"{prefix}_fund_ret"] = nav_m["nav"].pct_change()
            nav_m = nav_m.dropna()
            idx_m = idx_m.merge(
                nav_m[["date", f"{prefix}_fund_nav", f"{prefix}_fund_ret"]],
                on="date", how="left"
            )
        else:
            idx_m[f"{prefix}_fund_nav"] = np.nan
            idx_m[f"{prefix}_fund_ret"] = np.nan

    # Step 3: Compute index returns (price only)
    for col in ["sp500", "nasdaq", "divlow"]:
        idx_m[f"{col}_price_ret"] = idx_m[f"{col}_price"].pct_change()

    # Step 4: Apply fee model to index returns
    # Net index return = price return - monthly fee drag
    monthly_drag = FEE_MODEL["monthly_drag"]
    for col in ["sp500", "nasdaq", "divlow"]:
        idx_m[f"{col}_index_net_ret"] = idx_m[f"{col}_price_ret"] - monthly_drag

    # Step 5: Preferred return = fund NAV return if available, else index net return
    for col in ["sp500", "nasdaq", "divlow"]:
        idx_m[f"{col}_tr"] = idx_m[f"{col}_fund_ret"].fillna(idx_m[f"{col}_index_net_ret"])

    # Step 6: Cumulative returns
    idx_m = idx_m.dropna(subset=["sp500_tr", "nasdaq_tr", "divlow_tr"]).reset_index(drop=True)
    for col in ["sp500", "nasdaq", "divlow"]:
        idx_m[f"{col}_cum"] = (1 + idx_m[f"{col}_tr"]).cumprod()

    # Mark data source for each month
    idx_m["sp500_source"] = idx_m["sp500_fund_ret"].notna().map({True: "fund_nav", False: "index+fees"})
    idx_m["nasdaq_source"] = idx_m["nasdaq_fund_ret"].notna().map({True: "fund_nav", False: "index+fees"})
    idx_m["divlow_source"] = idx_m["divlow_fund_ret"].notna().map({True: "fund_nav", False: "index+fees"})

    # Print summary
    for col in ["sp500", "nasdaq", "divlow"]:
        n_fund = (idx_m[f"{col}_source"] == "fund_nav").sum()
        n_idx = (idx_m[f"{col}_source"] == "index+fees").sum()
        print(f"  {col}: {n_fund} months fund NAV, {n_idx} months index+fees")

    return idx_m


def build_monthly_json(prices):
    """Convert to JSON-serializable list of dicts."""
    cols = ["date", "sp500_price", "nasdaq_price", "divlow_price",
            "sp500_tr", "nasdaq_tr", "divlow_tr",
            "sp500_cum", "nasdaq_cum", "divlow_cum",
            "sp500_fund_nav", "nasdaq_fund_nav", "divlow_fund_nav",
            "sp500_fund_ret", "nasdaq_fund_ret", "divlow_fund_ret",
            "sp500_source", "nasdaq_source", "divlow_source"]
    available = [c for c in cols if c in prices.columns]
    records = prices[available].to_dict(orient="records")
    # Convert NaN to None for JSON
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                r[k] = None
    return records


def compute_portfolios(prices):
    """Compute all 231 portfolios."""
    portfolios = {}
    monthly_rets = prices[["sp500_tr", "nasdaq_tr", "divlow_tr"]]
    for sp in range(0, 101, 5):
        for nq in range(0, 101 - sp, 5):
            dv = 100 - sp - nq
            key = f"{sp}_{nq}_{dv}"
            w = (sp / 100, nq / 100, dv / 100)
            rets = w[0] * monthly_rets["sp500_tr"] + w[1] * monthly_rets["nasdaq_tr"] + w[2] * monthly_rets["divlow_tr"]
            cum = (1 + rets).prod()
            years = len(rets) / 12
            ann_ret = cum ** (1 / years) - 1
            ann_vol = rets.std() * np.sqrt(12)
            cum_series = (1 + rets).cumprod()
            mdd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min()
            sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
            portfolios[key] = {"sp": sp, "nq": nq, "dv": dv,
                               "ret": round(float(ann_ret), 8),
                               "vol": round(float(ann_vol), 8),
                               "mdd": round(float(mdd), 8),
                               "sharpe": round(float(sharpe), 8)}
    print(f"Computed {len(portfolios)} portfolios")
    return portfolios


def compute_rolling(prices, portfolios, window_years):
    """Compute rolling window returns."""
    window_months = window_years * 12
    monthly_rets = prices[["sp500_tr", "nasdaq_tr", "divlow_tr"]]
    n = len(prices)
    result = {}
    for key, alloc in portfolios.items():
        w = (alloc["sp"] / 100, alloc["nq"] / 100, alloc["dv"] / 100)
        rets = w[0] * monthly_rets["sp500_tr"] + w[1] * monthly_rets["nasdaq_tr"] + w[2] * monthly_rets["divlow_tr"]
        windows = []
        for i in range(0, n - window_months + 1, 3):
            sl = rets.iloc[i:i + window_months]
            cum = (1 + sl).prod()
            windows.append({
                "start": str(prices["date"].iloc[i].date()),
                "end": str(prices["date"].iloc[i + window_months - 1].date()),
                "ann_ret": round(float(cum ** (1 / window_years) - 1), 8)
            })
        result[key] = windows
    n_win = len(windows) if windows else 0
    print(f"  {window_years}Y rolling: {n_win} windows x {len(portfolios)} portfolios")
    return result


def main():
    print("=" * 60)
    print("Enhanced Backtest Data Generator v2")
    print("Fund NAV + Index + Fee Model")
    print("=" * 60)

    # 1. Fetch data
    index_daily = fetch_index_data()
    sp500_nav = fetch_fund_nav(SP500_FUND, "博时标普500ETF联接A")
    time.sleep(1)
    nasdaq_nav = fetch_fund_nav(NASDAQ_FUND, "广发纳斯达克100ETF联接A")
    time.sleep(1)
    divlow_nav = fetch_fund_nav(DIVLOW_ETF, "红利低波ETF")

    # 2. Build hybrid dataset
    print("\nBuilding hybrid dataset...")
    prices = build_hybrid_dataset(index_daily, sp500_nav, nasdaq_nav, divlow_nav)

    # 3. Build output
    fund_start_dates = {}
    if sp500_nav is not None and not sp500_nav.empty:
        fund_start_dates["sp500_fund_start"] = str(sp500_nav["date"].min().date())
    if nasdaq_nav is not None and not nasdaq_nav.empty:
        fund_start_dates["nasdaq_fund_start"] = str(nasdaq_nav["date"].min().date())
    if divlow_nav is not None and not divlow_nav.empty:
        fund_start_dates["divlow_fund_start"] = str(divlow_nav["date"].min().date())

    output = {
        "metadata": {
            "data_start": str(prices["date"].min().date()),
            "data_end": str(prices["date"].max().date()),
            "n_months": len(prices),
            "n_years": round(len(prices) / 12, 1),
            "fee_model": FEE_MODEL,
            "funds": {
                "sp500": {"code": SP500_FUND, "name": "博时标普500ETF联接A"},
                "nasdaq": {"code": NASDAQ_FUND, "name": "广发纳斯达克100ETF联接A"},
                "divlow": {"code": DIVLOW_ETF, "name": "红利低波ETF"},
            },
            **fund_start_dates,
            "methodology": "Hybrid: fund NAV (real dividends+fees included) where available; index+fee model (0.85%/yr drag) for earlier periods. Subscription 0.12%, redemption 0.5-1.5%."
        },
        "monthly_data": build_monthly_json(prices),
        "portfolios": compute_portfolios(prices),
        "rolling_windows": {}
    }

    # 4. Rolling windows
    print("\nComputing rolling windows...")
    output["rolling_windows"]["5Y"] = compute_rolling(prices, output["portfolios"], 5)
    output["rolling_windows"]["10Y"] = compute_rolling(prices, output["portfolios"], 10)

    # 5. Write
    with open(OUTPUT, "w") as f:
        json.dump(output, f)
    print(f"\nSaved: {OUTPUT}")
    print(f"Size: {len(json.dumps(output)) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
