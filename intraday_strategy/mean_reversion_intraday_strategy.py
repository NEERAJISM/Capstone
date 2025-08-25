import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from intraday_strategy.kalman_filter.kalman_filter import Kalman

# ================== CONFIG ================== #
CAPITAL = 100000.0
PER_TRADE_RISK = 0.02
STT_PCT = 0.00025
SLIPPAGE_PCT = 0.001
BROKERAGE_PCT = 0.0003

ROLLING_WINDOW = 60
Z_ENTRY = 2.5
Z_EXIT = 0.2
MIN_HOLD_BARS = 5
COOLDOWN_BARS = 15
VOL_FILTER = 0.0005

START_TIME = pd.to_datetime("10:15:00").time()
END_TIME   = pd.to_datetime("14:30:00").time()
# =========================================== #

class MeanReversionIntradayStrategy:

    @staticmethod
    def apply_strategy(df, stockA="A", stockB="B", plot=True):
        """Apply improved Kalman-filter mean reversion intraday strategy with plotting."""
        beta, alpha = Kalman.kalman_hedge(df['Close_A'].values, df['Close_B'].values)
        df = df.copy()
        df['beta'] = beta
        df['alpha'] = alpha
        df['spread'] = df['Close_B'] - (df['beta'] * df['Close_A'] + df['alpha'])

        df['spread_mean'] = df['spread'].rolling(ROLLING_WINDOW, min_periods=1).mean()
        df['spread_std']  = df['spread'].rolling(ROLLING_WINDOW, min_periods=1).std().replace(0, np.nan).bfill()
        df['z'] = (df['spread'] - df['spread_mean']) / df['spread_std']

        position, entry, trades = 0, None, []
        notional = CAPITAL * PER_TRADE_RISK
        last_exit_index = -COOLDOWN_BARS

        for i in range(len(df)):
            ts = df.index[i]
            pxA, pxB = float(df['Close_A'].iloc[i]), float(df['Close_B'].iloc[i])
            z = float(df['z'].iloc[i]) if np.isfinite(df['z'].iloc[i]) else np.nan
            b = float(df['beta'].iloc[i]) if np.isfinite(df['beta'].iloc[i]) else 1.0
            t = ts.time()

            # Close at EOD
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
                position, entry = 0, None
                last_exit_index = i

            if not (START_TIME <= t <= END_TIME):
                continue

            # ENTRY
            if position == 0 and np.isfinite(z) and (i - last_exit_index) >= COOLDOWN_BARS:
                if df['spread_std'].iloc[i] < VOL_FILTER * np.mean([pxA, pxB]):
                    continue
                if z > Z_ENTRY:   # Short spread
                    sizeB = -round(notional / pxB)
                    sizeA = +round(abs(b) * notional / pxA)
                    turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                    cost_open = turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                    entry = dict(i_entry=i, entry_time=ts, entryA=pxA, entryB=pxB,
                                 sizeA=sizeA, sizeB=sizeB, z_entry=z, open_cost=cost_open)
                    position = -1
                elif z < -Z_ENTRY:  # Long spread
                    sizeB = +round(notional / pxB)
                    sizeA = -round(abs(b) * notional / pxA)
                    turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                    cost_open = turnover * (STT_PCT + SLIPPAGE_PCT + BROKERAGE_PCT)
                    entry = dict(i_entry=i, entry_time=ts, entryA=pxA, entryB=pxB,
                                 sizeA=sizeA, sizeB=sizeB, z_entry=z, open_cost=cost_open)
                    position = 1

            # EXIT
            elif position != 0 and np.isfinite(z) and abs(z) < Z_EXIT:
                if (i - entry['i_entry']) >= MIN_HOLD_BARS:
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
                    position, entry = 0, None
                    last_exit_index = i

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df['exit_day'] = pd.to_datetime(trades_df['exit_time']).dt.date
            daily_pnl = trades_df.groupby('exit_day')['net_pnl'].sum().reset_index()
            daily_pnl.rename(columns={'net_pnl': 'daily_net_pnl'}, inplace=True)
            daily_pnl['cum_pnl'] = daily_pnl['daily_net_pnl'].cumsum()
        else:
            daily_pnl = pd.DataFrame(columns=['exit_day', 'daily_net_pnl', 'cum_pnl'])

        # --- Plotting ---
        if plot and not trades_df.empty:
            plt.figure(figsize=(14, 10))

            # Prices
            ax1 = plt.subplot(3, 1, 1)
            df['Close_A'].plot(ax=ax1, label=stockA)
            df['Close_B'].plot(ax=ax1, label=stockB)
            for _, tr in trades_df.iterrows():
                ax1.axvline(tr['entry_time'], color='g' if tr['side'] in ['EXIT_SIGNAL'] else 'r', linestyle='--')
            ax1.set_ylabel("Prices")
            ax1.legend()

            # Spread
            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            df['spread'].plot(ax=ax2, color='orange', label='Spread')
            df['spread_mean'].plot(ax=ax2, color='black', linestyle='--', label='Mean')
            ax2.fill_between(df.index, df['spread_mean']+df['spread_std'], df['spread_mean']-df['spread_std'],
                             color='gray', alpha=0.2, label='±1σ')
            ax2.set_ylabel("Spread")
            ax2.legend()

            # Z-score
            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            df['z'].plot(ax=ax3, label="Z-score")
            ax3.axhline(Z_ENTRY, color='r', linestyle='--', label="Entry Threshold")
            ax3.axhline(-Z_ENTRY, color='r', linestyle='--')
            ax3.axhline(Z_EXIT, color='g', linestyle='--', label="Exit Band")
            ax3.axhline(-Z_EXIT, color='g', linestyle='--')
            ax3.legend()
            ax3.set_ylabel("Z-score")

            plt.tight_layout()
            plt.show()

            # Daily PnL
            plt.figure(figsize=(12,4))
            plt.bar(daily_pnl['exit_day'], daily_pnl['daily_net_pnl'], label="Daily PnL")
            plt.plot(daily_pnl['exit_day'], daily_pnl['cum_pnl'], color='orange', label="Cumulative PnL")
            plt.legend()
            plt.title("Strategy PnL")
            plt.show()

        return df, trades_df, daily_pnl
