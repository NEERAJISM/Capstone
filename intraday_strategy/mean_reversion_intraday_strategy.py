import numpy as np
import pandas as pd
from intraday_strategy.kalman_filter.kalman_filter import Kalman

# ================== CONFIG ================== #
CAPITAL = 100000.0
PER_TRADE_RISK = 0.02       # fraction of capital used per trade
STT_PCT = 0.00025           # 0.025%
SLIPPAGE_PCT = 0.001        # 0.1%
BROKERAGE_PCT = 0.0003      # 0.03%

ROLLING_WINDOW = 60         # lookback (minutes)
Z_ENTRY = 2.0
Z_EXIT = 0.5

# Trading hours restrictions
START_TIME = pd.to_datetime("10:15:00").time()
END_TIME   = pd.to_datetime("14:30:00").time()
EOD_TIME   = pd.to_datetime("15:30:00").time()
# =========================================== #

class MeanReversionIntradayStrategy:

    @staticmethod
    def apply_strategy(df):
        """Apply Kalman-filter mean reversion strategy with intraday restrictions."""
        # --- Kalman filter hedge ratio ---
        beta, alpha = Kalman.kalman_hedge(df['Close_A'].values, df['Close_B'].values)
        df = df.copy()
        df['beta'] = beta
        df['alpha'] = alpha
        df['spread'] = df['Close_B'] - (df['beta'] * df['Close_A'] + df['alpha'])

        # --- Z-score ---
        df['spread_mean'] = df['spread'].rolling(ROLLING_WINDOW, min_periods=1).mean()
        df['spread_std']  = df['spread'].rolling(ROLLING_WINDOW, min_periods=1).std().replace(0, np.nan).bfill()
        df['z'] = (df['spread'] - df['spread_mean']) / df['spread_std']

        # --- Backtest loop ---
        position = 0
        entry = None
        trades = []
        notional = CAPITAL * PER_TRADE_RISK

        for i in range(len(df)):
            ts = df.index[i]
            pxA, pxB = float(df['Close_A'].iloc[i]), float(df['Close_B'].iloc[i])
            z = float(df['z'].iloc[i]) if np.isfinite(df['z'].iloc[i]) else np.nan
            b = float(df['beta'].iloc[i]) if np.isfinite(df['beta'].iloc[i]) else 1.0
            t = ts.time()

            # --- Force-close at new day open ---
            if i > 0 and ts.date() != df.index[i - 1].date() and position != 0 and entry:
                sizeA, sizeB = entry['sizeA'], entry['sizeB']
                gross = sizeB * (pxB - entry['entryB']) + sizeA * (pxA - entry['entryA'])
                turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                costs = entry['open_cost'] + turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                net = gross - costs
                trades.append({
                    'entry_time': entry['entry_time'], 'exit_time': ts,
                    'side': 'EOD_CLOSE',
                    'entryA': entry['entryA'], 'entryB': entry['entryB'],
                    'exitA': pxA, 'exitB': pxB,
                    'sizeA': sizeA, 'sizeB': sizeB,
                    'z_entry': entry['z_entry'], 'z_exit': np.nan,
                    'gross_pnl': gross, 'costs': costs, 'net_pnl': net
                })
                print(f"{ts} [EOD CLOSE] Net={net:.2f}")
                position, entry = 0, None

            # --- Skip outside trading window (but allow 14:30 force-close) ---
            if not (START_TIME <= t <= END_TIME):
                # Force close at END_TIME
                if position != 0 and entry and t > END_TIME:
                    sizeA, sizeB = entry['sizeA'], entry['sizeB']
                    gross = sizeB * (pxB - entry['entryB']) + sizeA * (pxA - entry['entryA'])
                    turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                    costs = entry['open_cost'] + turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                    net = gross - costs
                    trades.append({
                        'entry_time': entry['entry_time'], 'exit_time': ts,
                        'side': 'FORCE_CLOSE',
                        'entryA': entry['entryA'], 'entryB': entry['entryB'],
                        'exitA': pxA, 'exitB': pxB,
                        'sizeA': sizeA, 'sizeB': sizeB,
                        'z_entry': entry['z_entry'], 'z_exit': z,
                        'gross_pnl': gross, 'costs': costs, 'net_pnl': net
                    })
                    print(f"{ts} [FORCE CLOSE 14:30] Net={net:.2f}")
                    position, entry = 0, None
                continue

            # --- ENTRY ---
            if position == 0 and np.isfinite(z):
                if z > Z_ENTRY:  # Short spread
                    sizeB = -(notional / pxB)
                    sizeA = +(abs(b) * notional / pxA)
                    turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                    cost_open = turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                    entry = dict(entry_time=ts, entryA=pxA, entryB=pxB, sizeA=sizeA, sizeB=sizeB,
                                 z_entry=z, open_cost=cost_open)
                    position = -1
                    print(f"{ts} [OPEN SHORT] qtyA={int(sizeA)}, qtyB={int(sizeB)}, z={z:.2f}")

                elif z < -Z_ENTRY:  # Long spread
                    sizeB = +(notional / pxB)
                    sizeA = -(abs(b) * notional / pxA)
                    turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                    cost_open = turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                    entry = dict(entry_time=ts, entryA=pxA, entryB=pxB, sizeA=sizeA, sizeB=sizeB,
                                 z_entry=z, open_cost=cost_open)
                    position = 1
                    print(f"{ts} [OPEN LONG] qtyA={int(sizeA)}, qtyB={int(sizeB)}, z={z:.2f}")

            # --- EXIT ---
            elif position != 0 and np.isfinite(z) and abs(z) < Z_EXIT:
                sizeA, sizeB = entry['sizeA'], entry['sizeB']
                gross = sizeB * (pxB - entry['entryB']) + sizeA * (pxA - entry['entryA'])
                turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                costs = entry['open_cost'] + turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                net = gross - costs
                trades.append({
                    'entry_time': entry['entry_time'], 'exit_time': ts,
                    'side': 'EXIT_SIGNAL',
                    'entryA': entry['entryA'], 'entryB': entry['entryB'],
                    'exitA': pxA, 'exitB': pxB,
                    'sizeA': sizeA, 'sizeB': sizeB,
                    'z_entry': entry['z_entry'], 'z_exit': z,
                    'gross_pnl': gross, 'costs': costs, 'net_pnl': net
                })
                print(f"{ts} [EXIT SIGNAL] Net={net:.2f}, z={z:.2f}")
                position, entry = 0, None

        # --- Results ---
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['exit_day'] = pd.to_datetime(trades_df['exit_time']).dt.date
            daily_pnl = trades_df.groupby('exit_day')['net_pnl'].sum().reset_index()
            daily_pnl.rename(columns={'net_pnl': 'daily_net_pnl'}, inplace=True)
            daily_pnl['cum_pnl'] = daily_pnl['daily_net_pnl'].cumsum()
        else:
            daily_pnl = pd.DataFrame(columns=['exit_day', 'daily_net_pnl', 'cum_pnl'])
        return df, trades_df, daily_pnl
