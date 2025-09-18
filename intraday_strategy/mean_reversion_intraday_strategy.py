import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from intraday_strategy.kalman_filter.kalman_filter import Kalman
from config import config


class MeanReversionIntradayStrategy:
    @staticmethod
    def apply_strategy(df):
        """Apply improved Kalman-filter mean reversion intraday strategy with plotting."""
        beta, alpha = Kalman.kalman_hedge(df["Close_A"].values, df["Close_B"].values)
        df = df.copy()
        df["beta"] = beta
        df["alpha"] = alpha
        df["spread"] = df["Close_B"] - (df["beta"] * df["Close_A"] + df["alpha"])

        df["spread_mean"] = df["spread"].rolling(config.strategy.rolling_window, min_periods=1).mean()
        df["spread_std"] = (
            df["spread"]
            .rolling(config.strategy.rolling_window, min_periods=1)
            .std()
            .replace(0, np.nan)
            .bfill()
        )
        df["z"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

        position, entry, trades = 0, None, []
        notional = config.strategy.capital * config.strategy.per_trade_risk
        last_exit_index = -config.strategy.cooldown_bars

        for i in range(len(df)):
            ts = df.index[i]
            pxA, pxB = float(df["Close_A"].iloc[i]), float(df["Close_B"].iloc[i])
            z = float(df["z"].iloc[i]) if np.isfinite(df["z"].iloc[i]) else np.nan
            b = float(df["beta"].iloc[i]) if np.isfinite(df["beta"].iloc[i]) else 1.0
            t = ts.time()

            # Close at EOD
            if (
                i > 0
                and ts.date() != df.index[i - 1].date()
                and position != 0
                and entry
            ):
                sizeA, sizeB = entry["sizeA"], entry["sizeB"]
                gross = sizeB * (pxB - entry["entryB"]) + sizeA * (
                    pxA - entry["entryA"]
                )
                turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                costs = entry["open_cost"] + turnover * (
                    config.strategy.stt_pct + config.strategy.slippage_pct + config.strategy.brokerage_pct
                )
                net = gross - costs
                trades.append(
                    {
                        "entry_time": entry["entry_time"],
                        "exit_time": ts,
                        "side": "EOD_CLOSE",
                        "entryA": round(entry["entryA"], 4),
                        "entryB": round(entry["entryB"], 4),
                        "exitA": round(pxA, 4),
                        "exitB": round(pxB, 4),
                        "sizeA": sizeA,
                        "sizeB": sizeB,
                        "z_entry": round(entry["z_entry"], 4),
                        "z_exit": round(z, 4),
                        "gross_pnl": round(gross, 4),
                        "costs": round(costs, 4),
                        "net_pnl": round(net, 4),
                    }
                )
                position, entry = 0, None
                last_exit_index = i

            if t < config.strategy.start_time or t > config.strategy.end_time:
                continue

            # ENTRY
            if (
                position == 0
                and np.isfinite(z)
                and (i - last_exit_index) >= config.strategy.cooldown_bars
            ):
                if df["spread_std"].iloc[i] < config.strategy.vol_filter * np.mean([pxA, pxB]):
                    continue
                if z > config.strategy.z_entry:  # Short spread
                    sizeB = -round(notional / pxB)
                    sizeA = +round(abs(b) * notional / pxA)
                    turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                    cost_open = turnover * (config.strategy.stt_pct + config.strategy.slippage_pct + config.strategy.brokerage_pct)
                    entry = dict(
                        i_entry=i,
                        entry_time=ts,
                        entryA=pxA,
                        entryB=pxB,
                        sizeA=sizeA,
                        sizeB=sizeB,
                        z_entry=z,
                        open_cost=cost_open,
                    )
                    position = -1
                elif z < -config.strategy.z_entry:  # Long spread
                    sizeB = +round(notional / pxB)
                    sizeA = -round(abs(b) * notional / pxA)
                    turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                    cost_open = turnover * (config.strategy.stt_pct + config.strategy.slippage_pct + config.strategy.brokerage_pct)
                    entry = dict(
                        i_entry=i,
                        entry_time=ts,
                        entryA=pxA,
                        entryB=pxB,
                        sizeA=sizeA,
                        sizeB=sizeB,
                        z_entry=z,
                        open_cost=cost_open,
                    )
                    position = 1

            # EXIT
            elif position != 0 and np.isfinite(z) and abs(z) < config.strategy.z_exit:
                if (i - entry["i_entry"]) >= config.strategy.min_hold_bars:
                    sizeA, sizeB = entry["sizeA"], entry["sizeB"]
                    gross = sizeB * (pxB - entry["entryB"]) + sizeA * (
                        pxA - entry["entryA"]
                    )
                    turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                    costs = entry["open_cost"] + turnover * (
                        config.strategy.stt_pct + config.strategy.slippage_pct + config.strategy.brokerage_pct
                    )
                    net = gross - costs
                    trades.append(
                        {
                            "entry_time": entry["entry_time"],
                            "exit_time": ts,
                            "side": "EXIT_SIGNAL",
                            "entryA": round(entry["entryA"], 4),
                            "entryB": round(entry["entryB"], 4),
                            "exitA": round(pxA, 4),
                            "exitB": round(pxB, 4),
                            "sizeA": sizeA,
                            "sizeB": sizeB,
                            "z_entry": round(entry["z_entry"], 4),
                            "z_exit": round(z, 4),
                            "gross_pnl": round(gross, 4),
                            "costs": round(costs, 4),
                            "net_pnl": round(net, 4),
                        }
                    )
                    position, entry = 0, None
                    last_exit_index = i

        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df["exit_day"] = pd.to_datetime(trades_df["exit_time"]).dt.date
            daily_pnl = trades_df.groupby("exit_day")["net_pnl"].sum().reset_index()
            daily_pnl.rename(columns={"net_pnl": "daily_net_pnl"}, inplace=True)
            daily_pnl["cum_pnl"] = daily_pnl["daily_net_pnl"].cumsum()
        else:
            daily_pnl = pd.DataFrame(columns=["exit_day", "daily_net_pnl", "cum_pnl"])

        return df, trades_df, daily_pnl
