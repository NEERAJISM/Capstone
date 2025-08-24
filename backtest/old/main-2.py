import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# ---------------- CONFIG ---------------- #
DATA_FOLDER = "data/2021/Cash Data April 2021/"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

STT = 0.00025     # Securities Transaction Tax
BROKERAGE = 0.0003
SLIPPAGE = 0.001  # 0.1%
CAPITAL = 1000000 # Capital for position sizing
ENTRY_Z = 1.0
EXIT_Z = 0.0
# ----------------------------------------- #

def load_stock(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['<date>'] + ' ' + df['<time>'], format='%m/%d/%Y %H:%M:%S')
    df = df[['datetime', '<close>']].copy()
    df.rename(columns={'<close>': 'Close'}, inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.set_index('datetime', inplace=True)
    return df

def run_kalman_pair_backtest(stock1, stock2):
    try:
        df1 = load_stock(os.path.join(DATA_FOLDER, f"{stock1}.csv"))
        df2 = load_stock(os.path.join(DATA_FOLDER, f"{stock2}.csv"))
    except Exception as e:
        print(f"Error loading {stock1} or {stock2}: {e}")
        return

    df = pd.merge(df1, df2, left_index=True, right_index=True, suffixes=(f'_{stock1}', f'_{stock2}'))
    df.dropna(inplace=True)
    if df.empty:
        print(f"No overlapping data between {stock1} and {stock2}, skipping.")
        return

    Y = df[f'Close_{stock1}'].values
    X = df[f'Close_{stock2}'].values
    obs_mat = np.stack([X, np.ones(len(X))], axis=1)[:, np.newaxis, :]

    delta = 1e-4
    trans_cov = delta / (1 - delta) * np.eye(2)

    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_covariance=trans_cov,
        observation_covariance=1.0,
    )
    kf.observation_matrices = obs_mat
    state_means, _ = kf.filter(Y)

    hedge_ratio = state_means[:, 0]
    intercept = state_means[:, 1]
    spread = Y - (hedge_ratio * X + intercept)
    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std

    df['hedge_ratio'] = hedge_ratio
    df['spread'] = spread
    df['z_score'] = z_score

    # Generate signals
    df['position'] = 0
    df.loc[df['z_score'] > ENTRY_Z, 'position'] = -1  # short spread
    df.loc[df['z_score'] < -ENTRY_Z, 'position'] = 1  # long spread
    df.loc[(df['z_score'] < EXIT_Z) & (df['z_score'] > -EXIT_Z), 'position'] = 0

    # ---------------- Trade Simulation ---------------- #
    trades = []
    prev_pos = 0
    prev_time = None
    equity = CAPITAL
    entry_time, entry_px_y, entry_px_x = None, None, None
    entry_qty_y, entry_qty_x, entry_side = None, None, None

    for idx, row in df.iterrows():
        date_changed = False
        if prev_time is not None and idx.date() != prev_time.date():
            date_changed = True

        pos = int(row['position'])
        px_y = row[f'Close_{stock1}']
        px_x = row[f'Close_{stock2}']
        qty_y = np.floor((CAPITAL / 2) / px_y)
        qty_x = np.floor((CAPITAL / 2) / px_x)

        if pos != prev_pos or date_changed:
            # Close trade
            if prev_pos != 0:
                exit_time = idx
                gross_profit = (entry_qty_y * (px_y - entry_px_y) +
                                entry_qty_x * (px_x - entry_px_x))
                costs = (abs(entry_qty_y) * px_y + abs(entry_qty_x) * px_x) * (STT + BROKERAGE + SLIPPAGE)
                net_profit = gross_profit - costs
                equity += net_profit

                trades.append({
                    'date': entry_time.date(),
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'side': entry_side,
                    'qty_y': entry_qty_y,
                    'qty_x': entry_qty_x,
                    'entry_px_y': entry_px_y,
                    'entry_px_x': entry_px_x,
                    'exit_px_y': px_y,
                    'exit_px_x': px_x,
                    'gross_profit': gross_profit,
                    'costs': costs,
                    'net_profit': net_profit
                })
                print(f"{exit_time} - CLOSE {entry_side} spread, Net Profit={net_profit:.2f}")

            # Open new trade
            if pos != 0 and not date_changed:
                entry_time = idx
                entry_px_y = px_y
                entry_px_x = px_x
                entry_qty_y = qty_y if pos == 1 else -qty_y
                entry_qty_x = -qty_x if pos == 1 else qty_x
                entry_side = 'BUY' if pos == 1 else 'SELL'
                print(f"{entry_time} - OPEN {entry_side} spread, QtyY={entry_qty_y}, QtyX={entry_qty_x}, PxY={entry_px_y}, PxX={entry_px_x}")

        prev_time = idx
        prev_pos = pos

    # Save trades
    trades_df = pd.DataFrame(trades)
    trades_file = os.path.join(OUTPUT_FOLDER, f"{stock1}_{stock2}_trades.csv")
    trades_df.to_csv(trades_file, index=False)
    print(f"Saved trades to {trades_file}")

    # ---------------- Plot ---------------- #
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 1, 1)
    plt.plot(df.index, hedge_ratio, label='Hedge Ratio')
    plt.ylabel('Hedge Ratio')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(df.index, spread, label='Spread', color='orange')
    plt.axhline(spread_mean, color='black', linestyle='--', label='Mean')
    plt.axhline(spread_mean + spread_std, color='green', linestyle='--', label='+1 STD')
    plt.axhline(spread_mean - spread_std, color='red', linestyle='--', label='-1 STD')
    plt.ylabel('Spread')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df.index, z_score, label='Z-Score')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(ENTRY_Z, color='green', linestyle='--', label='Entry +1')
    plt.axhline(-ENTRY_Z, color='red', linestyle='--', label='Entry -1')
    plt.ylabel('Z-Score')
    plt.legend()

    plt.tight_layout()
    plot_file = os.path.join(OUTPUT_FOLDER, f"{stock1}_{stock2}_kalman_plot.png")
    plt.savefig(plot_file)
    plt.close()
    print(f"Saved plot: {plot_file}")

if __name__ == "__main__":
    run_kalman_pair_backtest("ICICIBANK", "HDFCBANK")
