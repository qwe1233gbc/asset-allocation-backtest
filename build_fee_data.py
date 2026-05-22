"""
Build fee-adjusted backtest data from V1 base data.

Applies uniform fee model to all three assets:
  - SP500/Nasdaq: QDII fund fees (0.85%/yr mgmt+custody, 0.12% subscription)
  - 红利低波: domestic ETF fees (0.60%/yr mgmt+custody, 0.12% subscription)

Output: backtest_data_fee.json
"""

import json
import numpy as np
import pandas as pd

INPUT = "backtest_data.json"
OUTPUT = "backtest_data_fee.json"

# Fee models (annual, converted to monthly drag)
FEE_MODEL = {
    "sp500": {
        "name": "标普500 (QDII: 0.85%/年管理+托管)",
        "monthly_drag": 0.0085 / 12,   # 0.85%/year
        "subscription": 0.0012,         # 0.12%
    },
    "nasdaq": {
        "name": "纳斯达克100 (QDII: 0.85%/年管理+托管)",
        "monthly_drag": 0.0085 / 12,
        "subscription": 0.0012,
    },
    "divlovol": {
        "name": "红利低波 (国内ETF: 0.60%/年管理+托管)",
        "monthly_drag": 0.0060 / 12,    # 0.60%/year
        "subscription": 0.0012,
    },
}


def apply_fees(data):
    """Apply fee drag to monthly returns and recompute cumulative."""
    monthly = data["monthly_data"]
    prices = pd.DataFrame(monthly)

    for col in ["sp500", "nasdaq", "divlovol"]:
        drag = FEE_MODEL[col]["monthly_drag"]
        # Subtract fee from total return
        tr_col = f"{col}_tr"
        cum_col = f"{col}_cum"

        if tr_col in prices.columns:
            prices[tr_col] = prices[tr_col] - drag
            # Recompute cumulative
            prices[cum_col] = (1 + prices[tr_col]).cumprod()

    # Convert back
    records = prices.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and np.isnan(v):
                r[k] = None
    return records


def recompute_portfolios(records, old_portfolios):
    """Recompute portfolio metrics with fee-adjusted returns."""
    portfolios = {}
    monthly_rets = pd.DataFrame(records)[["sp500_tr", "nasdaq_tr", "divlovol_tr"]]

    for key, alloc in old_portfolios.items():
        sp_w = alloc["sp"] / 100
        nq_w = alloc["nq"] / 100
        dv_w = alloc["dv"] / 100
        rets = (sp_w * monthly_rets["sp500_tr"] +
                nq_w * monthly_rets["nasdaq_tr"] +
                dv_w * monthly_rets["divlovol_tr"])
        cum = (1 + rets).prod()
        years = len(rets) / 12
        ann_ret = cum ** (1 / years) - 1
        ann_vol = rets.std() * np.sqrt(12)
        cum_series = (1 + rets).cumprod()
        peak = cum_series.cummax()
        mdd = ((cum_series - peak) / peak).min()
        sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
        portfolios[key] = {
            "sp": alloc["sp"], "nq": alloc["nq"], "dv": alloc["dv"],
            "ret": round(float(ann_ret), 8),
            "vol": round(float(ann_vol), 8),
            "mdd": round(float(mdd), 8),
            "sharpe": round(float(sharpe), 8),
        }
    print(f"  Recomputed {len(portfolios)} portfolios")
    return portfolios


def recompute_rolling(records, portfolios, window_years):
    """Recompute rolling window returns."""
    window_months = window_years * 12
    monthly_rets = pd.DataFrame(records)[["sp500_tr", "nasdaq_tr", "divlovol_tr"]]
    n = len(records)

    result = {}
    for key, alloc in portfolios.items():
        w = (alloc["sp"] / 100, alloc["nq"] / 100, alloc["dv"] / 100)
        rets = (w[0] * monthly_rets["sp500_tr"] +
                w[1] * monthly_rets["nasdaq_tr"] +
                w[2] * monthly_rets["divlovol_tr"])
        windows = []
        for i in range(0, n - window_months + 1, 3):
            sl = rets.iloc[i:i + window_months]
            cum = (1 + sl).prod()
            windows.append({
                "start": str(records[i]["date"]),
                "end": str(records[i + window_months - 1]["date"]),
                "ann_ret": round(float(cum ** (1 / window_years) - 1), 8)
            })
        result[key] = windows
    return result


def main():
    print("=" * 60)
    print("Building fee-adjusted backtest data")
    print("=" * 60)

    with open(INPUT, "r") as f:
        base = json.load(f)

    # Apply fees
    print("\nApplying fee model...")
    for col, info in FEE_MODEL.items():
        print(f"  {info['name']}")
    monthly = apply_fees(base)

    # Recomputed portfolios
    portfolios = recompute_portfolios(monthly, base["portfolios"])

    # Show comparison
    print("\n=== 费率调整对比 ===")
    old_p = base["portfolios"]
    for k in ["100_0_0", "0_100_0", "0_0_100", "60_30_10", "40_40_20"]:
        old_ret = old_p[k]["ret"]
        new_ret = portfolios[k]["ret"]
        print(f"  {k}: {old_ret*100:.1f}% → {new_ret*100:.1f}% (Δ{new_ret-old_ret:+.1%})")

    # Rolling
    rolling = {}
    print("\nRecomputing rolling windows...")
    rolling["5Y"] = recompute_rolling(monthly, portfolios, 5)
    rolling["10Y"] = recompute_rolling(monthly, portfolios, 10)

    # Build output
    fee_info = {col: {"name": info["name"], "annual_drag": info["monthly_drag"] * 12}
                for col, info in FEE_MODEL.items()}

    output = {
        "metadata": {
            **base["metadata"],
            "fee_model": fee_info,
            "methodology": base["metadata"]["methodology"]
            + " + management/custody fee deduction (see fee_model)",
        },
        "monthly_data": monthly,
        "portfolios": portfolios,
        "rolling_windows": rolling,
    }

    with open(OUTPUT, "w") as f:
        json.dump(output, f)

    # Summary
    print(f"\n=== 费率调整后 ===")
    print(f"  SP500:   {portfolios['100_0_0']['ret']*100:.1f}%")
    print(f"  Nasdaq:  {portfolios['0_100_0']['ret']*100:.1f}%")
    print(f"  红利低波: {portfolios['0_0_100']['ret']*100:.1f}%")
    print(f"\nSaved: {OUTPUT}")
    print(f"Size: {len(json.dumps(output)) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
