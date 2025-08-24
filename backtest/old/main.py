#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import coint
from datetime import datetime

# -------------------------
# CONFIG
# -------------------------
DATA_FOLDER = "data/2021/Cash Data April 2021/"
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Backtest / strategy params
ROLLING_WINDOW = 60         # rolling window for mean/std of spread (in minutes)
Z_ENTRY = 1.0               # z to enter
Z_EXIT = 0.2                # z to exit (close)
CAPITAL = 100000.0

# TODO - Kelly's criteria for per trade risk per pair

PER_TRADE_RISK = 0.02       # fraction of capital used per trade
STT_PCT = 0.00025           # example: 0.025% -> 0.00025
SLIPPAGE_PCT = 0.001        # 0.1%
BROKERAGE_PCT = 0.0003      # 0.03%

# -------------------------
# Helpers
# -------------------------
def load_stock(filepath):
    df = pd.read_csv(filepath)
    # adjust these column names if your CSV differs
    if '<date>' in df.columns and '<time>' in df.columns:
        df['datetime'] = pd.to_datetime(df['<date>'] + ' ' + df['<time>'], format='%m/%d/%Y %H:%M:%S', errors='coerce')
    else:
        # try common column names
        for col in ['datetime', 'timestamp', 'time', 'Date', 'DateTime']:
            if col in df.columns:
                df['datetime'] = pd.to_datetime(df[col], errors='coerce')
                break
    # find close column
    close_col = None
    for c in ['<close>', 'close', 'Close', 'LAST', 'last', 'Adj Close']:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise ValueError(f"No close column found in {filepath}. Columns: {df.columns.tolist()}")
    df = df[['datetime', close_col]].copy()
    df.rename(columns={close_col: 'Close'}, inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    df.set_index('datetime', inplace=True)
    df = df.sort_index()
    return df

def align_data(df1, df2):
    df = df1.join(df2, how='inner', lsuffix='_1', rsuffix='_2')
    df.columns = ['Close_x', 'Close_y']
    df.dropna(inplace=True)
    return df

# -------------------------
# Cointegration test
# -------------------------
def check_cointegration(x, y, p_thresh=0.05):
    try:
        score, pvalue, _ = coint(y, x)  # y ~ beta * x + const
        return pvalue, pvalue < p_thresh
    except Exception as e:
        print("Cointegration test failed:", e)
        return None, False

# -------------------------
# Kalman filter
# -------------------------
def run_kalman(x, y, delta=1e-4, obs_cov=1.0):
    # observation matrix for each time step: [x_t, 1]
    n = len(x)
    obs_mat = np.stack([x, np.ones(n)], axis=1)[:, np.newaxis, :]  # shape (n,1,2)
    trans_cov = delta / (1 - delta) * np.eye(2)

    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2,2)),
        transition_covariance=trans_cov,
        observation_covariance=obs_cov
    )
    # assign observation matrices
    kf.observation_matrices = obs_mat
    state_means, state_covs = kf.filter(y)
    betas = state_means[:, 0]
    intercepts = state_means[:, 1]
    return betas, intercepts

# -------------------------
# Signals & Backtest
# -------------------------
def generate_signals_and_backtest(df, betas, intercepts,
                                  rolling_window=ROLLING_WINDOW,
                                  z_entry=Z_ENTRY, z_exit=Z_EXIT,
                                  capital=CAPITAL, per_trade_risk=PER_TRADE_RISK,
                                  stt_pct=STT_PCT, slippage_pct=SLIPPAGE_PCT, brokerage_pct=BROKERAGE_PCT):
    df = df.copy()
    x = df['Close_x'].values
    y = df['Close_y'].values
    spread = y - (betas * x + intercepts)
    df['beta'] = betas
    df['intercept'] = intercepts
    df['spread'] = spread
    df['spread_mean'] = df['spread'].rolling(window=rolling_window, min_periods=1).mean()
    df['spread_std'] = df['spread'].rolling(window=rolling_window, min_periods=1).std().replace(0, np.nan).fillna(method='bfill')
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    # generate positions (1 long spread: long y short x; -1 short spread: short y long x)
    pos = 0
    positions = []
    for z in df['zscore'].values:
        if pos == 0:
            if np.isnan(z):
                positions.append(0)
                continue
            if z > z_entry:
                pos = -1
            elif z < -z_entry:
                pos = 1
        elif pos == 1:
            if abs(z) < z_exit:
                pos = 0
        elif pos == -1:
            if abs(z) < z_exit:
                pos = 0
        positions.append(pos)
    df['position'] = positions

    # Backtest sim
    df['pos_prev'] = df['position'].shift(1).fillna(0)
    df['trade'] = df['position'] - df['pos_prev']

    notional = capital * per_trade_risk

    current_size_x = 0.0
    current_size_y = 0.0
    size_x_hist = []
    size_y_hist = []
    pnl_hist = []
    cash = capital
    cumulative_costs = 0.0
    equity_hist = []

    # iterate rows
    prev_px_x = df['Close_x'].iloc[0]
    prev_px_y = df['Close_y'].iloc[0]
    prev_pos = 0
    cum_pnl = 0.0

    for idx, row in df.iterrows():
        px_x = row['Close_x']
        px_y = row['Close_y']
        pos = int(row['position'])
        beta = float(row['beta']) if not np.isnan(row['beta']) else 1.0

        # if trade/change
        if pos != prev_pos:
            # close previous sizes -> incur turnover cost
            turnover_close = abs(current_size_x * prev_px_x) + abs(current_size_y * prev_px_y)
            cost_close = turnover_close * (stt_pct + slippage_pct + brokerage_pct)
            cumulative_costs += cost_close
            cash -= cost_close

            # open new position if pos != 0
            if pos == 0:
                current_size_x = 0.0
                current_size_y = 0.0
            else:
                # determine new sizes: dollar-neutral approximate (adjust x by beta)
                size_y_new = notional / px_y if px_y > 0 else 0.0
                size_x_new = (notional * abs(beta)) / px_x if px_x > 0 else 0.0
                if pos == 1:
                    current_size_y = size_y_new
                    current_size_x = -size_x_new
                elif pos == -1:
                    current_size_y = -size_y_new
                    current_size_x = size_x_new
                # entry costs
                turnover_open = abs(current_size_x * px_x) + abs(current_size_y * px_y)
                cost_open = turnover_open * (stt_pct + slippage_pct + brokerage_pct)
                cumulative_costs += cost_open
                cash -= cost_open

        # mark-to-market P&L step (from prev price to current)
        pnl_step = current_size_x * (px_x - prev_px_x) + current_size_y * (px_y - prev_px_y)
        cum_pnl += pnl_step
        cash += pnl_step

        size_x_hist.append(current_size_x)
        size_y_hist.append(current_size_y)
        pnl_hist.append(cum_pnl)
        equity_hist.append(cash - 0.0)  # cash already accounts for costs and pnl

        prev_px_x = px_x
        prev_px_y = px_y
        prev_pos = pos

    df['size_x'] = size_x_hist
    df['size_y'] = size_y_hist
    df['cum_pnl'] = pnl_hist
    df['equity'] = equity_hist
    df['cum_costs'] = cumulative_costs

    return df

# -------------------------
# Plotting
# -------------------------
def plot_and_save(df, stock_x, stock_y, out_folder=OUTPUT_FOLDER):
    title = f"{stock_y} vs {stock_x} - Kalman Pair Trading"
    plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(4,1,1)
    df['Close_x'].plot(ax=ax1, label=stock_x)
    df['Close_y'].plot(ax=ax1, label=stock_y, alpha=0.8)
    ax1.set_title('Prices')
    ax1.legend()

    ax2 = plt.subplot(4,1,2, sharex=ax1)
    df['spread'].plot(ax=ax2, label='Spread')
    df['spread_mean'].plot(ax=ax2, linestyle='--', label='Rolling Mean')
    ax2.axhline(0, color='k', linestyle=':')
    ax2.set_title('Spread & Rolling Mean')
    ax2.legend()

    ax3 = plt.subplot(4,1,3, sharex=ax1)
    df['zscore'].plot(ax=ax3, label='Z-score')
    ax3.axhline(Z_ENTRY, color='r', linestyle='--', label='Entry Threshold')
    ax3.axhline(-Z_ENTRY, color='g', linestyle='--', label='Entry Threshold -')
    ax3.axhline(0, color='k', linestyle=':')
    ax3.set_title('Z-score')
    ax3.legend()

    ax4 = plt.subplot(4,1,4, sharex=ax1)
    df['equity'].plot(ax=ax4, label='Equity')
    ax4.set_title('Equity Curve')
    ax4.legend()

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    filename = f"{stock_x}_{stock_y}_kalman_backtest.png"
    filepath = os.path.join(out_folder, filename)
    plt.savefig(filepath)
    plt.close()
    print("Saved plot ->", filepath)

# -------------------------
# Main
# -------------------------
def main(stock_x, stock_y, data_folder=DATA_FOLDER):
    fx = os.path.join(data_folder, f"{stock_x}.csv")
    fy = os.path.join(data_folder, f"{stock_y}.csv")
    if not os.path.exists(fx) or not os.path.exists(fy):
        print("CSV files not found. Check data folder and filenames.")
        return

    df1 = load_stock(fx)
    df2 = load_stock(fy)
    df = align_data(df1, df2)
    if df.empty:
        print("No overlapping timestamps between the two files.")
        return
    print(f"Loaded {len(df)} rows of aligned data")

    # cointegration check
    pvalue, is_coin = check_cointegration(df['Close_x'], df['Close_y'])
    print(f"Cointegration p-value: {pvalue:.6f}  -> cointegrated: {is_coin} (p<{0.05})")

    # run kalman
    betas, intercepts = run_kalman(df['Close_x'].values, df['Close_y'].values)
    df_bt = generate_signals_and_backtest(df, betas, intercepts,
                                         rolling_window=ROLLING_WINDOW,
                                         z_entry=Z_ENTRY, z_exit=Z_EXIT,
                                         capital=CAPITAL, per_trade_risk=PER_TRADE_RISK,
                                         stt_pct=STT_PCT, slippage_pct=SLIPPAGE_PCT,
                                         brokerage_pct=BROKERAGE_PCT)

    plot_and_save(df_bt, stock_x, stock_y)

    # save CSV results
    out_csv = os.path.join(OUTPUT_FOLDER, f"{stock_x}_{stock_y}_backtest.csv")
    df_bt.to_csv(out_csv)
    print("Saved backtest CSV ->", out_csv)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python kalman_pair_backtest_improved.py STOCK_X STOCK_Y")
    else:
        sx = sys.argv[1].upper()
        sy = sys.argv[2].upper()
        main(sx, sy)
