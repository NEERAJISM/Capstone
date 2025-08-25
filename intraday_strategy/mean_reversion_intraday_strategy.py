import numpy as np
import pandas as pd

from intraday_strategy.kalman_filter.kalman_filter import Kalman

# ================== CONFIG ================== #

# Capital, risk, and cost assumptions
CAPITAL = 100000.0
PER_TRADE_RISK = 0.02  # fraction of capital used per trade (sizing)
STT_PCT = 0.00025  # 0.025% (approx; adjust your broker’s rule if needed)
SLIPPAGE_PCT = 0.001  # 0.1%
BROKERAGE_PCT = 0.0003  # 0.03%

# Signal params
ROLLING_WINDOW = 60  # minutes for rolling mean/std for z-score
Z_ENTRY = 2.0
Z_EXIT = 0.5  # close when |z| < Z_EXIT


# =========================================== #

class MeanReversionIntradayStrategy:

    @staticmethod
    def apply_strategy(df):
        """Generate signals, run intraday-only backtest, and return trades + daily PnL."""
        # Kalman dynamic hedge
        beta, alpha = Kalman.kalman_hedge(df['Close_A'].values, df['Close_B'].values)
        df = df.copy()
        df['beta'] = beta
        df['alpha'] = alpha
        df['spread'] = df['Close_B'] - (df['beta'] * df['Close_A'] + df['alpha'])

        # Rolling z-score (adaptive mean/std)
        df['spread_mean'] = df['spread'].rolling(ROLLING_WINDOW, min_periods=1).mean()
        df['spread_std'] = df['spread'].rolling(ROLLING_WINDOW, min_periods=1).std().replace(0, np.nan).bfill()
        df['z'] = (df['spread'] - df['spread_mean']) / df['spread_std']

        # Backtest state
        position = 0  # 0 flat, +1 long spread (long B/short A), -1 short spread
        entry = None  # dict with entry details
        trades = []
        notional = CAPITAL * PER_TRADE_RISK

        for i in range(len(df)):
            ts = df.index[i]
            pxA = float(df['Close_A'].iloc[i])
            pxB = float(df['Close_B'].iloc[i])
            z = float(df['z'].iloc[i]) if np.isfinite(df['z'].iloc[i]) else np.nan
            b = float(df['beta'].iloc[i]) if np.isfinite(df['beta'].iloc[i]) else 1.0

            # detect day change: if first bar of a new day, force previous position to close at this bar
            if i > 0 and ts.date() != df.index[i - 1].date() and position != 0 and entry is not None:
                # EOD close at current prices
                size_a, size_b = entry['sizeA'], entry['sizeB']
                gross = size_b * (pxB - entry['entryB']) + size_a * (pxA - entry['entryA'])
                turnover = abs(size_a) * pxA + abs(size_b) * pxB
                costs = entry['open_cost'] + turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                net = gross - costs
                trades.append({
                    'entry_time': entry['entry_time'], 'exit_time': ts,
                    'side': 'EOD_LONG' if position == 1 else 'EOD_SHORT',
                    'entryA': entry['entryA'], 'entryB': entry['entryB'],
                    'exitA': pxA, 'exitB': pxB,
                    'sizeA': size_a, 'sizeB': size_b,
                    'z_entry': entry['z_entry'], 'z_exit': np.nan,
                    'gross_pnl': gross, 'costs': costs, 'net_pnl': net
                })
                position, entry = 0, None

            # ENTRY
            if position == 0 and np.isfinite(z):
                # Short spread: spread too high (z > Z_ENTRY) => short B, long A
                if z > Z_ENTRY:
                    size_b = -(notional / pxB)  # short B
                    size_a = +(abs(b) * notional / pxA)  # long A, scaled by |beta|
                    turnover = abs(size_a) * pxA + abs(size_b) * pxB
                    cost_open = turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                    entry = dict(entry_time=ts, entryA=pxA, entryB=pxB, sizeA=size_a, sizeB=size_b,
                                 z_entry=z, open_cost=cost_open)
                    position = -1
                    print(
                        f"{ts} OPEN SHORT spread | qtyA={int(size_a)} long A, qtyB={int(size_b)} short B | z={z:.2f} cost={cost_open:.2f}")

                # Long spread: spread too low (z < -Z_ENTRY) => long B, short A
                elif z < -Z_ENTRY:
                    size_b = +(notional / pxB)  # long B
                    size_a = -(abs(b) * notional / pxA)  # short A, scaled by |beta|
                    turnover = abs(size_a) * pxA + abs(size_b) * pxB
                    cost_open = turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                    entry = dict(entry_time=ts, entryA=pxA, entryB=pxB, sizeA=size_a, sizeB=size_b,
                                 z_entry=z, open_cost=cost_open)
                    position = 1
                    print(
                        f"{ts} OPEN LONG  spread | qtyA={int(size_a)} short A, qtyB={int(size_b)} long B | z={z:.2f} cost={cost_open:.2f}")

            # EXIT
            elif position != 0 and np.isfinite(z) and abs(z) < Z_EXIT:
                size_a, size_b = entry['sizeA'], entry['sizeB']
                gross = size_b * (pxB - entry['entryB']) + size_a * (pxA - entry['entryA'])
                turnover = abs(size_a) * pxA + abs(size_b) * pxB
                costs = entry['open_cost'] + turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                net = gross - costs
                trades.append({
                    'entry_time': entry['entry_time'], 'exit_time': ts,
                    'side': 'LONG_SPREAD' if position == 1 else 'SHORT_SPREAD',
                    'entryA': entry['entryA'], 'entryB': entry['entryB'],
                    'exitA': pxA, 'exitB': pxB,
                    'sizeA': size_a, 'sizeB': size_b,
                    'z_entry': entry['z_entry'], 'z_exit': z,
                    'gross_pnl': gross, 'costs': costs, 'net_pnl': net
                })
                print(
                    f"{ts} CLOSE {'LONG' if position == 1 else 'SHORT'} spread | net={net:.2f} (gross={gross:.2f} costs={costs:.2f})")
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
