#!/usr/bin/env python3
import os
import sys

from common.plots import Plots
from common.utils import Utils
from intraday_strategy.mean_reversion_intraday_strategy import MeanReversionIntradayStrategy

# ================== CONFIG ================== #
DATA_FOLDER = "../data/2021/Cash Data April 2021/"

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# =========================================== #


def align_prices(df_a, df_b):
    df = df_a.join(df_b, how='inner', lsuffix='_A', rsuffix='_B')
    df.columns = ['Close_A', 'Close_B']
    df = df.dropna()
    return df


def main(stock_a, stock_b):
    f_a = os.path.join(DATA_FOLDER, f"{stock_a}.csv")
    f_b = os.path.join(DATA_FOLDER, f"{stock_b}.csv")
    if not (os.path.exists(f_a) and os.path.exists(f_b)):
        print("CSV not found — check DATA_FOLDER and filenames.")
        return

    df_a = Utils.load_stock_csv(f_a)
    df_b = Utils.load_stock_csv(f_b)
    df = align_prices(df_a, df_b)

    if df.empty:
        print("No overlapping timestamps — cannot backtest.")
        return

    print(f"Loaded {len(df)} aligned rows for {stock_a} & {stock_b}. Running backtest...")
    df_out, trades_df, daily_pnl = MeanReversionIntradayStrategy.apply_strategy(df)

    Plots.plot_strategy(df_out, trades_df, stock_a, stock_b)

    # Save outputs
    trades_path = os.path.join(OUTPUT_FOLDER, f"{stock_a}_{stock_b}_trades.csv")
    daily_path = os.path.join(OUTPUT_FOLDER, f"{stock_a}_{stock_b}_daily_pnl.csv")
    df_path = os.path.join(OUTPUT_FOLDER, f"{stock_a}_{stock_b}_signals.csv")

    trades_df.to_csv(trades_path, index=False)
    daily_pnl.to_csv(daily_path, index=False)
    df_out.to_csv(df_path)
    print(f"Saved trades:      {trades_path}")
    print(f"Saved daily PnL:   {daily_path}")
    print(f"Saved signal data: {df_path}")

    # Plot daily & cumulative PnL
    Plots.plot_daily_pnl(daily_pnl, stock_a, stock_b)

    if not trades_df.empty:
        print("\n=== Summary ===")
        print(f"Trades:         {len(trades_df)}")
        print(f"Total Gross PnL {trades_df['gross_pnl'].sum():.2f}")
        print(f"Total Costs:    {trades_df['costs'].sum():.2f}")
        print(f"Total Net PnL:  {trades_df['net_pnl'].sum():.2f}")
    else:
        print("No trades were generated with current parameters.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python daily_backtest_kalman_pairs.py HDFCBANK ICICIBANK")
    else:
        main(sys.argv[1].upper(), sys.argv[2].upper())
