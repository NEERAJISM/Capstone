#!/usr/bin/env python3
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# ================== CONFIG ================== #
DATA_FOLDER = "../data/2021/Cash Data April 2021/"   # adjust path as needed
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Capital, risk, and cost assumptions
CAPITAL = 100000.0
PER_TRADE_RISK = 0.02       # fraction of capital used per trade (sizing)
STT_PCT = 0.00025           # 0.025% (approx; adjust your broker’s rule if needed)
SLIPPAGE_PCT = 0.001        # 0.1%
BROKERAGE_PCT = 0.0003      # 0.03%

# Signal params
ROLLING_WINDOW = 60         # minutes for rolling mean/std for z-score
Z_ENTRY = 2.0
Z_EXIT  = 0.5               # close when |z| < Z_EXIT

# Kalman params
DELTA = 5e-4
OBS_COV = 1.0
# =========================================== #

def load_stock_csv(filepath):
    """Load CSV with columns <date>, <time>, <close> -> DataFrame indexed by datetime with column Close."""
    df = pd.read_csv(filepath)
    if '<date>' in df.columns and '<time>' in df.columns:
        df['datetime'] = pd.to_datetime(df['<date>'] + ' ' + df['<time>'],
                                        format='%m/%d/%Y %H:%M:%S', errors='coerce')
    else:
        # Fallback for other schemas
        for c in ['datetime', 'timestamp', 'DateTime', 'Date', 'Time']:
            if c in df.columns:
                df['datetime'] = pd.to_datetime(df[c], errors='coerce')
                break
    # find close col
    ccol = None
    for c in ['<close>', 'Close', 'close', 'LAST', 'Adj Close']:
        if c in df.columns:
            ccol = c; break
    if ccol is None:
        raise ValueError(f"Close column not found in {filepath}. Columns={df.columns.tolist()}")

    df = df[['datetime', ccol]].copy()
    df.rename(columns={ccol: 'Close'}, inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['datetime', 'Close'], inplace=True)
    df.set_index('datetime', inplace=True)
    df.sort_index(inplace=True)
    return df

def align_prices(df_a, df_b):
    df = df_a.join(df_b, how='inner', lsuffix='_A', rsuffix='_B')
    df.columns = ['Close_A', 'Close_B']
    df = df.dropna()
    return df

def kalman_hedge(x, y, delta=DELTA, obs_cov=OBS_COV):
    """Time-varying hedge ratio and intercept using a 2D state Kalman filter."""
    n = len(x)
    obs_mats = np.stack([x, np.ones(n)], axis=1)[:, np.newaxis, :]  # (n,1,2)
    trans_cov = delta / (1.0 - delta) * np.eye(2)

    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        transition_covariance=trans_cov,
        observation_covariance=obs_cov,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
    )
    kf.observation_matrices = obs_mats
    state_means, _ = kf.filter(y)
    beta = state_means[:, 0]
    intercept = state_means[:, 1]
    return beta, intercept

def apply_strategy(df):
    """Generate signals, run intraday-only backtest, and return trades + daily PnL."""
    # Kalman dynamic hedge
    beta, alpha = kalman_hedge(df['Close_A'].values, df['Close_B'].values)
    df = df.copy()
    df['beta'] = beta
    df['alpha'] = alpha
    df['spread'] = df['Close_B'] - (df['beta'] * df['Close_A'] + df['alpha'])

    # Rolling z-score (adaptive mean/std)
    df['spread_mean'] = df['spread'].rolling(ROLLING_WINDOW, min_periods=1).mean()
    df['spread_std']  = df['spread'].rolling(ROLLING_WINDOW, min_periods=1).std().replace(0, np.nan).bfill()
    df['z'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    # Backtest state
    position = 0           # 0 flat, +1 long spread (long B/short A), -1 short spread
    entry = None           # dict with entry details
    trades = []
    notional = CAPITAL * PER_TRADE_RISK

    for i in range(len(df)):
        ts   = df.index[i]
        pxA  = float(df['Close_A'].iloc[i])
        pxB  = float(df['Close_B'].iloc[i])
        z    = float(df['z'].iloc[i]) if np.isfinite(df['z'].iloc[i]) else np.nan
        b    = float(df['beta'].iloc[i]) if np.isfinite(df['beta'].iloc[i]) else 1.0

        # detect day change: if first bar of a new day, force previous position to close at this bar
        if i > 0 and ts.date() != df.index[i-1].date() and position != 0 and entry is not None:
            # EOD close at current prices
            sizeA, sizeB = entry['sizeA'], entry['sizeB']
            gross = sizeB * (pxB - entry['entryB']) + sizeA * (pxA - entry['entryA'])
            turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
            costs = entry['open_cost'] + turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
            net = gross - costs
            trades.append({
                'entry_time': entry['entry_time'], 'exit_time': ts,
                'side': 'EOD_LONG' if position == 1 else 'EOD_SHORT',
                'entryA': entry['entryA'], 'entryB': entry['entryB'],
                'exitA': pxA, 'exitB': pxB,
                'sizeA': sizeA, 'sizeB': sizeB,
                'z_entry': entry['z_entry'], 'z_exit': np.nan,
                'gross_pnl': gross, 'costs': costs, 'net_pnl': net
            })
            position, entry = 0, None

        # ENTRY
        if position == 0 and np.isfinite(z):
            # Short spread: spread too high (z > Z_ENTRY) => short B, long A
            if z > Z_ENTRY:
                sizeB = -(notional / pxB)             # short B
                sizeA = +(abs(b) * notional / pxA)    # long A, scaled by |beta|
                turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                cost_open = turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                entry = dict(entry_time=ts, entryA=pxA, entryB=pxB, sizeA=sizeA, sizeB=sizeB,
                             z_entry=z, open_cost=cost_open)
                position = -1
                print(f"{ts} OPEN SHORT spread | qtyA={int(sizeA)} long A, qtyB={int(sizeB)} short B | z={z:.2f} cost={cost_open:.2f}")

            # Long spread: spread too low (z < -Z_ENTRY) => long B, short A
            elif z < -Z_ENTRY:
                sizeB = +(notional / pxB)             # long B
                sizeA = -(abs(b) * notional / pxA)    # short A, scaled by |beta|
                turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                cost_open = turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                entry = dict(entry_time=ts, entryA=pxA, entryB=pxB, sizeA=sizeA, sizeB=sizeB,
                             z_entry=z, open_cost=cost_open)
                position = 1
                print(f"{ts} OPEN LONG  spread | qtyA={int(sizeA)} short A, qtyB={int(sizeB)} long B | z={z:.2f} cost={cost_open:.2f}")

        # EXIT
        elif position != 0 and np.isfinite(z) and abs(z) < Z_EXIT:
            sizeA, sizeB = entry['sizeA'], entry['sizeB']
            gross = sizeB * (pxB - entry['entryB']) + sizeA * (pxA - entry['entryA'])
            turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
            costs = entry['open_cost'] + turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
            net = gross - costs
            trades.append({
                'entry_time': entry['entry_time'], 'exit_time': ts,
                'side': 'LONG_SPREAD' if position == 1 else 'SHORT_SPREAD',
                'entryA': entry['entryA'], 'entryB': entry['entryB'],
                'exitA': pxA, 'exitB': pxB,
                'sizeA': sizeA, 'sizeB': sizeB,
                'z_entry': entry['z_entry'], 'z_exit': z,
                'gross_pnl': gross, 'costs': costs, 'net_pnl': net
            })
            print(f"{ts} CLOSE {'LONG' if position==1 else 'SHORT'} spread | net={net:.2f} (gross={gross:.2f} costs={costs:.2f})")
            position, entry = 0, None

    trades_df = pd.DataFrame(trades)
    # daily PnL (intraday-only because of forced EOD exit)
    if not trades_df.empty:
        trades_df['exit_day'] = pd.to_datetime(trades_df['exit_time']).dt.date
        daily_pnl = trades_df.groupby('exit_day')['net_pnl'].sum().reset_index()
        daily_pnl.rename(columns={'net_pnl': 'daily_net_pnl'}, inplace=True)
        daily_pnl['cum_pnl'] = daily_pnl['daily_net_pnl'].cumsum()
    else:
        daily_pnl = pd.DataFrame(columns=['exit_day', 'daily_net_pnl', 'cum_pnl'])
    return df, trades_df, daily_pnl

def plot_daily_pnl(daily_pnl, stockA, stockB):
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
    plt.title(f"Daily & Cumulative PnL — Kalman MR Pairs: {stockA} vs {stockB}")
    plt.tight_layout()
    out = os.path.join(OUTPUT_FOLDER, f"{stockA}_{stockB}_daily_pnl.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved daily PnL plot: {out}")

def plot_strategy(df, trades_df, stockA, stockB):
    """Visualize stock prices, spread, z-score, and trade signals."""
    if df.empty:
        print("No data for plotting.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # --- Prices ---
    axes[0].plot(df.index, df['Close_A'], label=f"{stockA}")
    axes[0].plot(df.index, df['Close_B'], label=f"{stockB}")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].set_title(f"Kalman MR Pair Trading: {stockA} vs {stockB}")

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
    out = os.path.join(OUTPUT_FOLDER, f"{stockA}_{stockB}_strategy_plot.png")
    plt.savefig(out)
    plt.close()
    print(f"Saved strategy visualization: {out}")


def main(stockA, stockB):
    fA = os.path.join(DATA_FOLDER, f"{stockA}.csv")
    fB = os.path.join(DATA_FOLDER, f"{stockB}.csv")
    if not (os.path.exists(fA) and os.path.exists(fB)):
        print("CSV not found — check DATA_FOLDER and filenames.")
        return

    dfA = load_stock_csv(fA)
    dfB = load_stock_csv(fB)
    df = align_prices(dfA, dfB)

    if df.empty:
        print("No overlapping timestamps — cannot backtest.")
        return

    print(f"Loaded {len(df)} aligned rows for {stockA} & {stockB}. Running backtest...")
    df_out, trades_df, daily_pnl = apply_strategy(df)

    plot_strategy(df_out, trades_df, stockA, stockB)

    # Save outputs
    trades_path = os.path.join(OUTPUT_FOLDER, f"{stockA}_{stockB}_trades.csv")
    daily_path  = os.path.join(OUTPUT_FOLDER, f"{stockA}_{stockB}_daily_pnl.csv")
    df_path     = os.path.join(OUTPUT_FOLDER, f"{stockA}_{stockB}_signals.csv")

    trades_df.to_csv(trades_path, index=False)
    daily_pnl.to_csv(daily_path, index=False)
    df_out.to_csv(df_path)
    print(f"Saved trades:      {trades_path}")
    print(f"Saved daily PnL:   {daily_path}")
    print(f"Saved signal data: {df_path}")

    # Plot daily & cumulative PnL
    plot_daily_pnl(daily_pnl, stockA, stockB)

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
