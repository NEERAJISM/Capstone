from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from common import StockDataLoader, get_logger, filter_market_hours
from config import config

logger = get_logger(__name__)


def _compute_rolling_features(
    close: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute log_return, volatility, slope arrays (NaN-padded for first window-1 rows)."""
    n = len(close)
    log_ret = np.full(n, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])

    volatility = np.full(n, np.nan)
    slope = np.full(n, np.nan)
    for i in range(window - 1, n):
        lr_slice = log_ret[i - window + 1 : i + 1]
        valid = lr_slice[~np.isnan(lr_slice)]
        if len(valid) > 1:
            volatility[i] = np.std(valid, ddof=1) * np.sqrt(window)
        c_slice = close[i - window + 1 : i + 1]
        slope[i] = np.polyfit(range(len(c_slice)), c_slice, 1)[0]

    return log_ret, volatility, slope


def load_hmm_data(
    base_path, trading_date, ticker
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (training_df, trading_df) as Polars DataFrames with 'datetime', 'close', 'ticker'."""
    if isinstance(trading_date, str):
        trading_date = datetime.strptime(trading_date, "%Y-%m-%d")

    lookback_start = trading_date - relativedelta(
        months=config.regime_detection.lookback_months
    )
    # Load through end of trading day so intraday bars are included
    trading_day_end = trading_date.replace(hour=15, minute=30, second=0)

    loader = StockDataLoader(
        base_dir=config.data.data_dir,
        start=lookback_start.strftime("%Y-%m-%d %H:%M:%S"),
        end=trading_day_end.strftime("%Y-%m-%d %H:%M:%S"),
        tickers=[ticker],
        select_columns=["close"],
        impute=True,
    )
    data = loader.get_data_for_tickers()[ticker]
    data = data.with_columns(pl.lit(ticker).alias("ticker"))

    # Drop imputed overnight/weekend flat bars before any modeling.
    data = filter_market_hours(data)

    trading_date_date = trading_date.date()
    training_data = data.filter(pl.col("datetime").dt.date() < trading_date_date)
    trading_data = data.filter(pl.col("datetime").dt.date() == trading_date_date)

    return training_data, trading_data


def detect_regimes_train_test_rolling(
    data: pl.DataFrame,
    train_start: datetime,
    train_end: datetime,
    test_day: datetime,
    lookback: int = 60,
    rolling_window: int = 15,
) -> pl.DataFrame:
    # Slice training window
    train_df = data.filter(
        (pl.col("datetime") >= train_start) & (pl.col("datetime") <= train_end)
    ).sort("datetime")

    if train_df.is_empty():
        raise ValueError(f"Training data empty for range {train_start} -> {train_end}")

    train_close = train_df["close"].to_numpy().astype(float)
    log_ret, volatility, slope = _compute_rolling_features(train_close, rolling_window)

    valid_mask = ~(np.isnan(log_ret) | np.isnan(volatility) | np.isnan(slope))
    features_train = np.stack([log_ret, volatility, slope], axis=1)[valid_mask]

    if features_train.shape[0] == 0:
        raise ValueError(
            "Training features all-NaN after rolling. Increase lookback or check data."
        )

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_train)

    try:
        # Diagonal covariance is far more stable than "full" on intraday
        # features (fewer params, no near-singular full covariance), which
        # removes the "Model is not converging" collapse seen with "full".
        hmm = GaussianHMM(
            n_components=2,
            covariance_type="diag",
            n_iter=500,
            tol=1e-4,
            random_state=42,
        )
        hmm.fit(features_scaled)
    except Exception as e:
        logger.exception("HMM training failed: %s", e)
        raise

    if not hmm.monitor_.converged:
        logger.warning("HMM did not fully converge; using best-effort fit.")

    # Label states by a combined trending signature in scaled feature space:
    # trending = higher directional movement (|slope|) AND higher volatility.
    # Using both (not slope alone) makes the label assignment stable when one
    # feature barely separates the two states.
    train_states = hmm.predict(features_scaled)

    def _trending_score(state: int) -> float:
        rows = features_scaled[train_states == state]
        if len(rows) == 0:
            return -np.inf
        # cols: 0=log_return, 1=volatility, 2=slope
        return float(np.abs(rows[:, 2]).mean() + rows[:, 1].mean())

    trending_regime = 0 if _trending_score(0) >= _trending_score(1) else 1
    sideways_regime = 1 - trending_regime

    # Test data: single trading day
    test_day_date = test_day.date() if isinstance(test_day, datetime) else test_day
    test_df = data.filter(
        pl.col("datetime").dt.date() == test_day_date
    ).sort("datetime")

    if test_df.is_empty():
        logger.warning(
            "No test data for %s. Available: %s -> %s",
            test_day_date,
            data["datetime"].min(),
            data["datetime"].max(),
        )
        return pl.DataFrame()

    if len(test_df) <= lookback:
        logger.warning(
            "Test day has %d rows <= lookback (%d).", len(test_df), lookback
        )
        return pl.DataFrame()

    test_close = test_df["close"].to_numpy().astype(float)
    test_timestamps = test_df["datetime"].to_list()
    test_ticker = test_df["ticker"][0] if "ticker" in test_df.columns else None

    # Compute the test-day features once and decode the whole session as a
    # single coherent state path (Viterbi). This replaces the previous
    # per-minute isolated-window decode, which refit the scaler on each 60-bar
    # slice and took only its last point -> unstable, all-Trending labels, slow.
    # NOTE: full-day Viterbi uses the transition structure across the day, i.e.
    # mild within-session smoothing; acceptable for a regime *filter*. For a
    # strictly causal estimate, swap to forward filtering later.
    t_log, t_vol, t_slope = _compute_rolling_features(test_close, rolling_window)
    test_valid = ~(np.isnan(t_log) | np.isnan(t_vol) | np.isnan(t_slope))
    if not np.any(test_valid):
        logger.warning("Test-day features all-NaN after rolling. Nothing to decode.")
        return pl.DataFrame()

    feat = np.stack([t_log, t_vol, t_slope], axis=1)[test_valid]
    feat_scaled = scaler.transform(feat)

    regime_path = hmm.predict(feat_scaled)
    probs = hmm.predict_proba(feat_scaled)
    valid_idx = np.where(test_valid)[0]

    results = []
    for k, i in enumerate(valid_idx):
        regime = int(regime_path[k])
        results.append(
            {
                "ticker": test_ticker,
                "timestamp": test_timestamps[i],
                "close": float(test_close[i]),
                "regime": regime,
                "prob_sideways": float(probs[k, sideways_regime]),
                "prob_trending": float(probs[k, trending_regime]),
                "regime_label": "Trending" if regime == trending_regime else "Sideways",
            }
        )

    if not results:
        logger.warning("No regime results produced. Check rolling_window and intraday rows.")
        return pl.DataFrame()

    return pl.DataFrame(results)


def save_regime_detected(ticker: str) -> pl.DataFrame:
    trading_date = datetime.strptime(config.data.run_date, "%Y-%m-%d")

    training_data, trading_data = load_hmm_data(
        base_path=config.data.data_dir,
        trading_date=trading_date,
        ticker=ticker,
    )

    logger.info("Training rows: %d", len(training_data))
    logger.info("Trading rows:  %d", len(trading_data))

    if training_data.is_empty():
        logger.error("No training data for %s. Aborting.", ticker)
        return pl.DataFrame()

    if trading_data.is_empty():
        logger.error(
            "No trading data for %s on %s. Aborting.", ticker, trading_date.date()
        )
        return pl.DataFrame()

    all_data = pl.concat([training_data, trading_data]).sort("datetime")
    train_start = training_data["datetime"].min()
    train_end = training_data["datetime"].max()

    result = detect_regimes_train_test_rolling(
        all_data,
        train_start,
        train_end,
        trading_date,
        lookback=config.regime_detection.lookback_minutes,
    )

    if result.is_empty():
        logger.warning("No regime results for %s. Nothing to save.", ticker)
        return result

    regime_dir = (
        config.data.output_dir / "regime_detection" / str(config.data.run_date)
    )
    regime_dir.mkdir(parents=True, exist_ok=True)
    out_path = regime_dir / f"{ticker}_regimes_{trading_date.date()}.csv"
    result.write_csv(str(out_path))
    logger.info("Saved regimes to %s", out_path)
    return result
