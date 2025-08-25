#!/usr/bin/env python3
import os
import json
import pandas as pd

from common.plots import Plots
from common.utils import Utils
from intraday_strategy.mean_reversion_intraday_strategy import MeanReversionIntradayStrategy

# ================== CONFIG ================== #
DATA_FOLDER = "../data/2021/Cash Data April 2021/"
PAIR_JSON   = "../data/pair_trading_result.json"   # fixed path
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# =========================================== #


def align_prices(df_a, df_b):
    df = df_a.join(df_b, how="inner", lsuffix="_A", rsuffix="_B")
    df.columns = ["Close_A", "Close_B"]
    return df.dropna()


def run_pair(stock_a, stock_b):
    """Run backtest for one pair and save results."""
    f_a = os.path.join(DATA_FOLDER, f"{stock_a}.csv")
    f_b = os.path.join(DATA_FOLDER, f"{stock_b}.csv")
    if not (os.path.exists(f_a) and os.path.exists(f_b)):
        print(f"CSV missing for {stock_a}, {stock_b}")
        return None

    df_a = Utils.load_stock_csv(f_a)
    df_b = Utils.load_stock_csv(f_b)
    df = align_prices(df_a, df_b)
    if df.empty:
        print(f"No overlap for {stock_a}, {stock_b}")
        return None

    print(f"Running backtest: {stock_a} vs {stock_b}, {len(df)} rows")
    df_out, trades_df, daily_pnl = MeanReversionIntradayStrategy.apply_strategy(df)

    # --- Save per pair outputs ---
    pair_folder = os.path.join(OUTPUT_FOLDER, f"{stock_a}_{stock_b}")
    os.makedirs(pair_folder, exist_ok=True)

    df_out.to_csv(os.path.join(pair_folder, "signals.csv"))
    trades_df.to_csv(os.path.join(pair_folder, "trades.csv"), index=False)
    daily_pnl.to_csv(os.path.join(pair_folder, "daily_pnl.csv"), index=False)

    # Use your existing plotting functions (no outpath)
    Plots.plot_strategy(df_out, trades_df, stock_a, stock_b)
    Plots.plot_daily_pnl(daily_pnl, stock_a, stock_b)

    # --- Metrics summary ---
    if trades_df.empty:
        return {
            "pair": f"{stock_a}_{stock_b}",
            "trades": 0,
            "net_pnl": 0.0,
            "win_rate": 0.0,
            "daily_sharpe": 0.0,
        }
    else:
        wins = (trades_df["net_pnl"] > 0).mean()
        daily_sharpe = (
            daily_pnl["daily_net_pnl"].mean() / daily_pnl["daily_net_pnl"].std()
            if daily_pnl["daily_net_pnl"].std() > 0 else 0
        )
        return {
            "pair": f"{stock_a}_{stock_b}",
            "trades": len(trades_df),
            "net_pnl": trades_df["net_pnl"].sum(),
            "win_rate": wins,
            "daily_sharpe": daily_sharpe,
        }



def main():
    with open(PAIR_JSON, "r") as f:
        clusters = json.load(f)

    all_results = []
    for cid, cdata in clusters.items():
        pairs = cdata.get("pairs", [])
        if not pairs:
            continue
        print(f"\n=== Running {cid} ===")
        for stock_a, stock_b in pairs:
            result = run_pair(stock_a, stock_b)
            if result:
                all_results.append(result)

    # Save leaderboard
    if all_results:
        leaderboard = pd.DataFrame(all_results)
        leaderboard.sort_values("net_pnl", ascending=False, inplace=True)
        leaderboard.to_csv(os.path.join(OUTPUT_FOLDER, "pairs_leaderboard.csv"), index=False)
        print("\n=== Leaderboard saved: output/pairs_leaderboard.csv ===")
        print(leaderboard.head(10))
    else:
        print("No valid results.")


if __name__ == "__main__":
    main()
