from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
from common import get_logger, StockDataLoader
from config import config
from pathlib import Path
import json

logger = get_logger(__name__)

plots_dir = config.data.output_dir / "backtest" / str(config.data.run_date)
plots_dir.mkdir(parents=True, exist_ok=True)


class Plots:
    @staticmethod
    def plot_clusters(json_path):

        output_dir = Path(config.data.output_dir) / "pair_selection" / config.data.run_date/"plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load clusters from JSON
        with open(json_path, "r") as f:
            result = json.load(f)

        def plot_pairs_in_cluster(cluster_name, cluster_data, start, end, base_dir):
            """Plot all pairs inside a given cluster dict (closes + spread)."""
            pairs = cluster_data.get("pairs", [])

            for i, pair_info in enumerate(pairs):
                if isinstance(pair_info, dict):
                    tickers = pair_info["tickers"]
                    beta = pair_info.get("beta", 1.0)
                else:
                    tickers = pair_info
                    beta = 1.0

                loader = StockDataLoader(
                    tickers=tickers,
                    start=start,
                    end=end,
                    select_columns=["close"],
                    base_dir=base_dir,
                    impute=True,
                )
                data_dict = loader.get_data_for_tickers()

                dfs = {}
                for t, pl_df in data_dict.items():
                    pd_df = pl_df.to_pandas().set_index("datetime")
                    dfs[t] = pd_df["close"]

                t1, t2 = tickers
                s1 = dfs[t1]
                s2 = dfs[t2]
                spread = s1 - beta * s2

                # Price plot
                plt.figure(figsize=(10, 5))
                s1.rename(t1).plot(linewidth=1)
                (beta * s2).rename(f"{t2} * β({beta:.2f})").plot(linewidth=1)
                plt.title(
                    f"{cluster_name} — Pair {i}: {t1} vs {t2}\n"
                    f"Run={config.data.run_date}, Lookback={config.pair_selection.lookback_days}, MinMR={config.pair_selection.min_mean_reversion}, VolTh={config.pair_selection.volatility_threshold}, Vol={config.pair_selection.volume_threshold}, Clusters={config.pair_selection.n_clusters_pairs}"
                )
                plt.ylabel("Close Price (INR)")
                plt.tight_layout()

                price_path = output_dir / f"{cluster_name}_pair{i}_{t1}_{t2}_price.png"
                plt.savefig(price_path, dpi=300)
                plt.close()

                # Spread plot
                plt.figure(figsize=(10, 4))
                spread.plot(linewidth=1, color="purple")
                mean, std = spread.mean(), spread.std()
                plt.axhline(mean, color="red", linestyle="--", label="Mean Spread")
                plt.axhline(mean + std, color="green", linestyle="--", label="+1σ")
                plt.axhline(mean - std, color="green", linestyle="--", label="-1σ")
                plt.axhline(mean + 2 * std, color="orange", linestyle="--", label="+2σ")
                plt.axhline(mean - 2 * std, color="orange", linestyle="--", label="-2σ")
                plt.title(
                    f"{cluster_name} — Spread {t1} - β*{t2} where β={round(beta, 3)}"
                )
                plt.ylabel("Spread (INR)")
                plt.legend()
                plt.tight_layout()

                spread_path = (
                    output_dir / f"{cluster_name}_pair{i}_{t1}_{t2}_spread.png"
                )
                plt.savefig(spread_path, dpi=300)
                plt.close()
                

        end = datetime.strptime(config.data.run_date,  "%Y-%m-%d")

        start = end - timedelta(days=config.pair_selection.lookback_days)
        for cluster_name, cluster_data in result.items():
            plot_pairs_in_cluster(
                cluster_name,
                cluster_data,
                start=start,
                end=end,
                base_dir=config.data.data_dir,
            )

    @staticmethod
    def plot_daily_pnl(daily_pnl, stock_a, stock_b):
        if daily_pnl.empty:
            logger.info("No trades -> no daily PnL plot.")
            return
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.bar(
            daily_pnl["exit_day"], daily_pnl["daily_net_pnl"], label="Daily Net PnL"
        )
        ax1.set_ylabel("Daily Net PnL")
        ax1.set_xlabel("Day")
        ax1.tick_params(axis="x", rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(
            daily_pnl["exit_day"],
            daily_pnl["cum_pnl"],
            label="Cumulative PnL",
            linewidth=2,
        )
        ax2.set_ylabel("Cumulative PnL")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        plt.title(f"Daily & Cumulative PnL — Kalman MR Pairs: {stock_a} vs {stock_b}")
        plt.tight_layout()
        out = plots_dir / f"{stock_a}_{stock_b}" / "daily_pnl.png"
        plt.savefig(out)
        plt.close()
        logger.info(f"Saved daily PnL plot: {out}")

    @staticmethod
    def plot_strategy(df, trades_df, stock_a, stock_b):
        """Visualize stock prices, spread, z-score, and trade signals."""
        if df.empty:
            logger.info("No data for plotting.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # --- Prices ---
        axes[0].plot(df.index, df["Close_A"], label=f"{stock_a}")
        axes[0].plot(df.index, df["Close_B"], label=f"{stock_b}")
        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].set_title(f"Kalman MR Pair Trading: {stock_a} vs {stock_b}")

        # --- Spread ---
        axes[1].plot(df.index, df["spread"], color="orange", label="Spread")
        axes[1].plot(
            df.index,
            df["spread_mean"],
            color="black",
            linestyle="--",
            label="Spread Mean",
        )
        axes[1].plot(
            df.index,
            df["spread_mean"] + df["spread_std"],
            color="green",
            linestyle="--",
            label="+1 STD",
        )
        axes[1].plot(
            df.index,
            df["spread_mean"] - df["spread_std"],
            color="red",
            linestyle="--",
            label="-1 STD",
        )
        axes[1].set_ylabel("Spread")
        axes[1].legend()

        # --- Z-score ---
        axes[2].plot(df.index, df["z"], label="Z-score")
        axes[2].axhline(0, color="black", linestyle="--")
        axes[2].axhline(1, color="green", linestyle="--", label="Entry (+1)")
        axes[2].axhline(-1, color="red", linestyle="--", label="Entry (-1)")
        axes[2].axhline(0.2, color="blue", linestyle=":", label="Exit band")
        axes[2].axhline(-0.2, color="blue", linestyle=":")
        axes[2].set_ylabel("Z-score")
        axes[2].legend()

        # --- Mark trades on Spread plot ---
        if not trades_df.empty:
            for _, t in trades_df.iterrows():
                entry_time = pd.to_datetime(t["entry_time"])
                exit_time = pd.to_datetime(t["exit_time"])
                # Entry marker
                axes[1].axvline(
                    entry_time,
                    color="green" if "LONG" in t["side"] else "red",
                    linestyle="--",
                    alpha=0.6,
                )
                # Exit marker
                axes[1].axvline(exit_time, color="blue", linestyle=":", alpha=0.6)

        plt.tight_layout()
        out = plots_dir / f"{stock_a}_{stock_b}" / "strategy_plot.png"
        plt.savefig(out)
        plt.close()
        logger.info(f"Saved strategy visualization: {out}")
