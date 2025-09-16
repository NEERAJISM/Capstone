#!/usr/bin/env python3
import os

from common.plots import Plots
from intraday_strategy import MeanReversionIntradayStrategy
from common import StockDataLoader
from config import config
import pandas as pd

def align_prices(df_a:pd.DataFrame, df_b:pd.DataFrame):
    df_a = df_a.set_index("datetime")
    df_b = df_b.set_index("datetime")
    df = df_a.join(df_b, how="inner", lsuffix="_A", rsuffix="_B")
    df.columns = ["Close_A", "Close_B"]
    return df.dropna()


def run_pair(stock_a:str, stock_b:str, stack_data_loader:StockDataLoader):
    """Run backtest for one pair and save results."""
    
    df_a, df_b = stack_data_loader.get_data_for_tickers()[stock_a], stack_data_loader.get_data_for_tickers()[stock_b]
    df = align_prices(df_a.to_pandas(), df_b.to_pandas())
    if df.empty:
        print(f"No overlap for {stock_a}, {stock_b}")
        return None

    print(f"Running backtest: {stock_a} vs {stock_b}, {len(df)} rows")
    df_out, trades_df, daily_pnl = MeanReversionIntradayStrategy.apply_strategy(df)

    # --- Save per pair outputs ---
    pair_folder = os.path.join(config.output_dir / "backtest_results", f"{stock_a}_{stock_b}")
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


