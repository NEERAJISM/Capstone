import matplotlib.pyplot as plt
import os
import pandas as pd

OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class Plots:

    @staticmethod
    def plot_daily_pnl(daily_pnl, stock_a, stock_b):
        if daily_pnl.empty:
            print("No trades -> no daily PnL plot.")
            return
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.bar(daily_pnl['exit_day'], daily_pnl['daily_net_pnl'], label='Daily Net PnL')
        ax1.set_ylabel('Daily Net PnL')
        ax1.set_xlabel('Day')
        ax1.tick_params(axis='x', rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(daily_pnl['exit_day'], daily_pnl['cum_pnl'], label='Cumulative PnL', linewidth=2)
        ax2.set_ylabel('Cumulative PnL')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.title(f"Daily & Cumulative PnL — Kalman MR Pairs: {stock_a} vs {stock_b}")
        plt.tight_layout()
        out = os.path.join(OUTPUT_FOLDER, f"{stock_a}_{stock_b}_daily_pnl.png")
        plt.savefig(out)
        plt.close()
        print(f"Saved daily PnL plot: {out}")

    @staticmethod
    def plot_strategy(df, trades_df, stock_a, stock_b):
        """Visualize stock prices, spread, z-score, and trade signals."""
        if df.empty:
            print("No data for plotting.")
            return

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # --- Prices ---
        axes[0].plot(df.index, df['Close_A'], label=f"{stock_a}")
        axes[0].plot(df.index, df['Close_B'], label=f"{stock_b}")
        axes[0].set_ylabel("Price")
        axes[0].legend()
        axes[0].set_title(f"Kalman MR Pair Trading: {stock_a} vs {stock_b}")

        # --- Spread ---
        axes[1].plot(df.index, df['spread'], color='orange', label="Spread")
        axes[1].plot(df.index, df['spread_mean'], color='black', linestyle='--', label="Spread Mean")
        axes[1].plot(df.index, df['spread_mean'] + df['spread_std'], color='green', linestyle='--', label="+1 STD")
        axes[1].plot(df.index, df['spread_mean'] - df['spread_std'], color='red', linestyle='--', label="-1 STD")
        axes[1].set_ylabel("Spread")
        axes[1].legend()

        # --- Z-score ---
        axes[2].plot(df.index, df['z'], label="Z-score")
        axes[2].axhline(0, color='black', linestyle='--')
        axes[2].axhline(1, color='green', linestyle='--', label="Entry (+1)")
        axes[2].axhline(-1, color='red', linestyle='--', label="Entry (-1)")
        axes[2].axhline(0.2, color='blue', linestyle=':', label="Exit band")
        axes[2].axhline(-0.2, color='blue', linestyle=':')
        axes[2].set_ylabel("Z-score")
        axes[2].legend()

        # --- Mark trades on Spread plot ---
        if not trades_df.empty:
            for _, t in trades_df.iterrows():
                entry_time = pd.to_datetime(t['entry_time'])
                exit_time = pd.to_datetime(t['exit_time'])
                # Entry marker
                axes[1].axvline(entry_time, color='green' if "LONG" in t['side'] else 'red',
                                linestyle='--', alpha=0.6)
                # Exit marker
                axes[1].axvline(exit_time, color='blue', linestyle=':', alpha=0.6)

        plt.tight_layout()
        out = os.path.join(OUTPUT_FOLDER, f"{stock_a}_{stock_b}_strategy_plot.png")
        plt.savefig(out)
        plt.close()
        print(f"Saved strategy visualization: {out}")
