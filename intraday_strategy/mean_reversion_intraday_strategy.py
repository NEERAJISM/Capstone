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

        df["spread_mean"] = (
            df["spread"].rolling(config.strategy.rolling_window, min_periods=1).mean()
        )
        df["spread_std"] = (
            df["spread"]
            .rolling(config.strategy.rolling_window, min_periods=1)
            .std()
            .replace(0, np.nan)
            .bfill()
        )
        df["z"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]

        # helper for safe rounding
        def _r(x):
            try:
                return float(np.round(x, 4))
            except Exception:
                return np.nan

        position, entry, trades = 0, None, []
        notional = config.strategy.capital * config.strategy.per_trade_risk
        last_exit_index = -config.strategy.cooldown_bars

        # Entry-funnel diagnostics: count why bars fail to produce an entry.
        funnel = {
            "in_window": 0,
            "flat_and_ready": 0,   # position 0, finite z, past cooldown
            "rej_regime": 0,       # blocked by both-Sideways gate
            "rej_vol_filter": 0,   # blocked by spread_std vol filter
            "z_below_entry": 0,    # gates passed but |z| < z_entry
            "entries": 0,
        }

        for i in range(len(df)):
            ts = df.index[i]
            pxA, pxB = float(df["Close_A"].iloc[i]), float(df["Close_B"].iloc[i])
            z = float(df["z"].iloc[i]) if np.isfinite(df["z"].iloc[i]) else np.nan
            b = float(df["beta"].iloc[i]) if np.isfinite(df["beta"].iloc[i]) else 1.0
            t = ts.time()

            # --- EOD close (force)
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
                    config.strategy.stt_pct
                    + config.strategy.slippage_pct
                    + config.strategy.brokerage_pct
                )
                net = gross - costs
                trades.append(
                    {
                        "entry_time": entry["entry_time"],
                        "exit_time": ts,
                        "side": "EOD_CLOSE",
                        "entryA": _r(entry["entryA"]),
                        "entryB": _r(entry["entryB"]),
                        "exitA": _r(pxA),
                        "exitB": _r(pxB),
                        "sizeA": int(entry["sizeA"]),
                        "sizeB": int(entry["sizeB"]),
                        "z_entry": _r(entry["z_entry"]),
                        "z_exit": np.nan,
                        "gross_pnl": _r(gross),
                        "costs": _r(costs),
                        "net_pnl": _r(net),
                    }
                )
                position, entry = 0, None
                last_exit_index = i

            # skip outside trading window
            if t < config.strategy.start_time or t > config.strategy.end_time:
                continue
            funnel["in_window"] += 1

            # ENTRY
            if (
                position == 0
                and np.isfinite(z)
                and (i - last_exit_index) >= config.strategy.cooldown_bars
            ):
                funnel["flat_and_ready"] += 1
                if (
                    config.strategy.use_regime_filter
                    and "regime_A" in df.columns
                    and "regime_B" in df.columns
                ):
                    if (
                        df["regime_A"].iloc[i] != "Sideways"
                        or df["regime_B"].iloc[i] != "Sideways"
                    ):
                        funnel["rej_regime"] += 1
                        continue  # only enter if both are Sideways

                if df["spread_std"].iloc[i] < config.strategy.vol_filter * np.mean(
                    [pxA, pxB]
                ):
                    funnel["rej_vol_filter"] += 1
                    continue
                if abs(z) <= config.strategy.z_entry:
                    funnel["z_below_entry"] += 1
                if z > config.strategy.z_entry:  # Short spread
                    sizeB = -round(notional / pxB)
                    sizeA = +round(abs(b) * notional / pxA)
                    turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                    cost_open = turnover * (
                        config.strategy.stt_pct
                        + config.strategy.slippage_pct
                        + config.strategy.brokerage_pct
                    )
                    entry = dict(
                        i_entry=i,
                        entry_time=ts,
                        entryA=_r(pxA),
                        entryB=_r(pxB),
                        sizeA=int(sizeA),
                        sizeB=int(sizeB),
                        z_entry=_r(z),
                        open_cost=_r(cost_open),
                    )
                    position = -1
                    funnel["entries"] += 1
                    print(
                        f"{ts} [OPEN SHORT] qtyA={entry['sizeA']}, qtyB={entry['sizeB']}, z={entry['z_entry']:.4f}, cost={entry['open_cost']:.4f}"
                    )

                elif z < -config.strategy.z_entry:  # Long spread
                    sizeB = +round(notional / pxB)
                    sizeA = -round(abs(b) * notional / pxA)
                    turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                    cost_open = turnover * (
                        config.strategy.stt_pct
                        + config.strategy.slippage_pct
                        + config.strategy.brokerage_pct
                    )
                    entry = dict(
                        i_entry=i,
                        entry_time=ts,
                        entryA=_r(pxA),
                        entryB=_r(pxB),
                        sizeA=int(sizeA),
                        sizeB=int(sizeB),
                        z_entry=_r(z),
                        open_cost=_r(cost_open),
                    )
                    position = 1
                    funnel["entries"] += 1
                    print(
                        f"{ts} [OPEN LONG] qtyA={entry['sizeA']}, qtyB={entry['sizeB']}, z={entry['z_entry']:.4f}, cost={entry['open_cost']:.4f}"
                    )

            # EXIT
            elif position != 0 and np.isfinite(z) and abs(z) < config.strategy.z_exit:
                if (i - entry["i_entry"]) >= config.strategy.min_hold_bars:
                    sizeA, sizeB = entry["sizeA"], entry["sizeB"]
                    gross = sizeB * (pxB - entry["entryB"]) + sizeA * (
                        pxA - entry["entryA"]
                    )
                    turnover = abs(sizeA) * pxA + abs(sizeB) * pxB
                    costs = entry["open_cost"] + turnover * (
                        config.strategy.stt_pct
                        + config.strategy.slippage_pct
                        + config.strategy.brokerage_pct
                    )
                    net = gross - costs
                    trades.append(
                        {
                            "entry_time": entry["entry_time"],
                            "exit_time": ts,
                            "side": "EXIT_SIGNAL",
                            "entryA": _r(entry["entryA"]),
                            "entryB": _r(entry["entryB"]),
                            "exitA": _r(pxA),
                            "exitB": _r(pxB),
                            "sizeA": int(sizeA),
                            "sizeB": int(sizeB),
                            "z_entry": _r(entry["z_entry"]),
                            "z_exit": _r(z),
                            "gross_pnl": _r(gross),
                            "costs": _r(costs),
                            "net_pnl": _r(net),
                        }
                    )
                    print(f"{ts} [EXIT SIGNAL] Net={_r(net):.4f}, z={_r(z):.4f}")
                    position, entry = 0, None
                    last_exit_index = i

        # Entry-funnel diagnostic: shows exactly which gate kills entries.
        print(
            "[FUNNEL] in_window={in_window} flat_ready={flat_and_ready} "
            "rej_regime={rej_regime} rej_vol_filter={rej_vol_filter} "
            "z_below_entry={z_below_entry} entries={entries}".format(**funnel)
        )

        # create dataframe and coerce/round numeric cols robustly
        trades_df = pd.DataFrame(trades)

        if not trades_df.empty:
            numeric_cols = [
                "entryA",
                "entryB",
                "exitA",
                "exitB",
                "z_entry",
                "z_exit",
                "gross_pnl",
                "costs",
                "net_pnl",
            ]
            for col in numeric_cols:
                if col in trades_df.columns:
                    trades_df[col] = pd.to_numeric(
                        trades_df[col], errors="coerce"
                    ).round(4)

            trades_df["exit_day"] = pd.to_datetime(trades_df["exit_time"]).dt.date
            daily_pnl = trades_df.groupby("exit_day")["net_pnl"].sum().reset_index()
            daily_pnl.rename(columns={"net_pnl": "daily_net_pnl"}, inplace=True)
            daily_pnl["cum_pnl"] = daily_pnl["daily_net_pnl"].cumsum().round(4)
        else:
            daily_pnl = pd.DataFrame(columns=["exit_day", "daily_net_pnl", "cum_pnl"])

        return df, trades_df, daily_pnl
