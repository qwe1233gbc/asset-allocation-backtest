"""
Enhance existing backtest_data.json with:
  1. Actual fund NAV data overlays (real dividends + fees included)
  2. Fee model applied to index-only periods
  3. Hybrid monthly returns: fund NAV where available, index+fees elsewhere

Fund codes (Alipay):
  SP500:    博时标普500ETF联接A (050025), since ~2012-06
  Nasdaq:   广发纳斯达克100ETF联接A (270042), since ~2012-08
  DivLowVol: 红利低波ETF (512890), since ~2018-12
"""

import json, time
import akshare as ak
import pandas as pd
import numpy as np

# Fee model
MONTHLY_FEE_DRAG = 0.0085 / 12  # 0.85%/year = mgmt 0.6% + custody 0.25%

FUNDS = {
    "sp500": {"code": "050025", "name": "博时标普500ETF联接A"},
    "nasdaq": {"code": "270042", "name": "广发纳斯达克100ETF联接A"},
    "divlovol": {"code": "512890", "name": "红利低波ETF"},
}

INPUT = "backtest_data.json"
OUTPUT = "backtest_data_v2.json"


def fetch_fund_monthly_returns(fund_code, fund_name):
    """Pull fund NAV and compute monthly total returns."""
    print(f"Fetching {fund_name} ({fund_code})...")
    try:
        nav = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
        nav.columns = ["date", "nav", "daily_return"]
        nav["date"] = pd.to_datetime(nav["date"])
        nav = nav.sort_values("date")

        # Resample to month-end
        nav_m = nav.set_index("date").resample("ME").last().dropna().reset_index()
        nav_m["nav_change"] = nav_m["nav"].pct_change()  # total return incl dividends & fees
        nav_m = nav_m.dropna()
        nav_m["date"] = nav_m["date"].dt.date

        print(f"  {len(nav_m)} months, {nav_m['date'].min()} to {nav_m['date'].max()}")
        return dict(zip(nav_m["date"].astype(str), nav_m["nav_change"]))
    except Exception as e:
        print(f"  Error: {e}")
        return {}


def main():
    print("=" * 60)
    print("Enhancing backtest data with fund NAVs + fee model")
    print("=" * 60)

    # Load existing data
    with open(INPUT, "r") as f:
        data = json.load(f)

    monthly = pd.DataFrame(data["monthly_data"])

    # Fetch fund NAV returns
    fund_rets = {}
    for key, fund_info in FUNDS.items():
        fund_rets[key] = fetch_fund_monthly_returns(fund_info["code"], fund_info["name"])
        time.sleep(1.5)

    # Enhance monthly data
    n_fund = {k: 0 for k in FUNDS}
    n_index = {k: 0 for k in FUNDS}

    for i, row in enumerate(monthly.to_dict(orient="records")):
        date_str = str(row["date"])

        for key in FUNDS:
            idx_ret = row[f"{key}_tr"]  # original: price return + estimated dividend

            # Apply fee model to original: net = gross - fee drag
            net_idx_ret = idx_ret - MONTHLY_FEE_DRAG

            # Check if fund NAV is available for this month
            if date_str in fund_rets[key]:
                fund_ret = fund_rets[key][date_str]
                # Use fund NAV return (already net of fees and actual dividends)
                monthly.at[i, f"{key}_tr"] = round(fund_ret, 10)
                monthly.at[i, f"{key}_source"] = "fund_nav"
                # Keep original for comparison
                monthly.at[i, f"{key}_tr_original"] = round(idx_ret, 10)
                monthly.at[i, f"{key}_tr_index_net"] = round(net_idx_ret, 10)
                n_fund[key] += 1
            else:
                # Use index return with fee model
                monthly.at[i, f"{key}_tr"] = round(net_idx_ret, 10)
                monthly.at[i, f"{key}_source"] = "index_fees"
                monthly.at[i, f"{key}_tr_original"] = round(idx_ret, 10)
                monthly.at[i, f"{key}_tr_index_net"] = round(net_idx_ret, 10)
                n_index[key] += 1

    # Recompute cumulative returns
    for key in FUNDS:
        monthly[f"{key}_cum"] = (1 + monthly[f"{key}_tr"]).cumprod()

    # Report
    for key, name in [(k, v["name"]) for k, v in FUNDS.items()]:
        print(f"  {name}: {n_fund[key]} months fund NAV, {n_index[key]} months index+fees")

    # Convert back to list
    enhanced_monthly = []
    for _, row in monthly.iterrows():
        r = row.to_dict()
        # Convert numpy types
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                r[k] = int(v)
            elif isinstance(v, (np.floating,)):
                if np.isnan(v):
                    r[k] = None
                else:
                    r[k] = float(v)
            elif isinstance(v, pd.Timestamp):
                r[k] = str(v.date())
        enhanced_monthly.append(r)

    # Recompute portfolios
    print("\nRecomputing portfolios...")
    portfolios = {}
    for sp in range(0, 101, 5):
        for nq in range(0, 101 - sp, 5):
            dv = 100 - sp - nq
            key = f"{sp}_{nq}_{dv}"
            w = (sp / 100, nq / 100, dv / 100)
            rets = (w[0] * monthly["sp500_tr"] + w[1] * monthly["nasdaq_tr"] +
                    w[2] * monthly["divlovol_tr"])
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
    print(f"  {len(portfolios)} portfolios computed")

    # Rolling windows
    print("Computing rolling windows...")
    rolling = {}
    for wy in [5, 10]:
        wm = wy * 12
        n = len(monthly)
        rolling[f"{wy}Y"] = {}
        for key, alloc in portfolios.items():
            w = (alloc["sp"] / 100, alloc["nq"] / 100, alloc["dv"] / 100)
            rets = (w[0] * monthly["sp500_tr"] + w[1] * monthly["nasdaq_tr"] +
                    w[2] * monthly["divlovol_tr"])
            windows = []
            for i in range(0, n - wm + 1, 3):
                sl = rets.iloc[i:i + wm]
                c = (1 + sl).prod()
                windows.append({
                    "start": str(monthly["date"].iloc[i]),
                    "end": str(monthly["date"].iloc[i + wm - 1]),
                    "ann_ret": round(float(c ** (1 / wy) - 1), 8)
                })
            rolling[f"{wy}Y"][key] = windows
        print(f"  {wy}Y: {len(windows)} windows x {len(portfolios)} portfolios")

    # Fund start dates
    fund_starts = {}
    for key in FUNDS:
        source_dates = [d for d in enhanced_monthly if d[f"{key}_source"] == "fund_nav"]
        if source_dates:
            fund_starts[f"{key}_fund_start"] = str(source_dates[0]["date"])

    # Build output
    output = {
        "metadata": {
            **data["metadata"],
            "version": 2,
            "fee_model": {
                "management_fee_annual": 0.006,
                "custody_fee_annual": 0.0025,
                "total_annual_drag": 0.0085,
                "subscription_fee": 0.0012,
                "redemption_tiers": [[7, 0.015], [365, 0.005], [99999, 0.0]],
                "note": "指数阶段已扣减月均费率。基金净值阶段已含实际费费。定投再平衡需另外计算申购/赎回费。"
            },
            "funds": {k: v for k, v in FUNDS.items()},
            **fund_starts,
            "methodology": "Hybrid: fund NAV returns where available (real dividends+fees already included); index returns with 0.85%/yr fee drag for pre-fund periods. Subscription 0.12% and redemption 0.5-1.5% applied separately in frontend simulations.",
            "comparison_note": "Compared to v1 (index+estimated dividends), this version deducts real mgmt+custody fees and uses actual fund total returns where available."
        },
        "monthly_data": enhanced_monthly,
        "portfolios": portfolios,
        "rolling_windows": rolling
    }

    with open(OUTPUT, "w") as f:
        json.dump(output, f)
    print(f"\nSaved: {OUTPUT}")
    print(f"Size: {len(json.dumps(output)) / 1024:.1f} KB")

    # Comparison: v1 vs v2 for default 40/35/25
    v1_key = "40_35_25"
    v1_port = data["portfolios"].get(v1_key, {})
    v2_port = portfolios.get(v1_key, {})
    print(f"\n=== V1 vs V2 Comparison (40/35/25) ===")
    print(f"V1 (index+est div): ret={v1_port.get('ret', 0)*100:.2f}%, vol={v1_port.get('vol', 0)*100:.2f}%, mdd={v1_port.get('mdd', 0)*100:.2f}%")
    print(f"V2 (hybrid+fees):  ret={v2_port.get('ret', 0)*100:.2f}%, vol={v2_port.get('vol', 0)*100:.2f}%, mdd={v2_port.get('mdd', 0)*100:.2f}%")


if __name__ == "__main__":
    main()
