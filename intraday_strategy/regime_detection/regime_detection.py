from functools import lru_cache
import os
import pandas as pd
from datetime import datetime, timedelta
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import numpy as np
from common import get_logger, StockDataLoader
from config import config
from dateutil.relativedelta import relativedelta

logger = get_logger(__name__)

def load_hmm_data(base_path, trading_date, ticker):
    trading_date = pd.to_datetime(trading_date).to_pydatetime()
    given_months_back = trading_date - relativedelta(
        months=config.regime_detection.lookback_months
    )
    loader = StockDataLoader(
        base_dir=config.data.data_dir,
        start=given_months_back,
        end=trading_date,
        tickers=[ticker],
        select_columns=["close"],
        impute=True,
    )
    data = loader.get_data_for_tickers()[ticker].to_pandas()

    # FIX: reset index so 'datetime' is a column
    data = data.reset_index()
    data["ticker"] = ticker

    # Training = last 3 months up to *day before trading_date*
    training_data = data[data["datetime"] < trading_date]

    # Trading = only the trading_date
    trading_data = data[data["datetime"].dt.date == trading_date.date()]

    return training_data, trading_data


def detect_regimes_train_test_rolling(
    data, train_start, train_end, test_day, lookback=60
):
    # 1. Training set
    train_data = data[(data.index >= train_start) & (data.index <= train_end)].copy()
    if train_data.empty:
        raise ValueError("Training data is empty. Check train_start/train_end dates.")

    # Training features
    train_data["log_return"] = np.log(
        train_data["close"] / train_data["close"].shift(1)
    )
    window = 15
    train_data["volatility"] = train_data["log_return"].rolling(window).std() * np.sqrt(
        window
    )
    train_data["slope"] = (
        train_data["close"]
        .rolling(window)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    )
    train_data.dropna(inplace=True)

    features_train = train_data[["log_return", "volatility", "slope"]].values
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)

    # Train HMM
    hmm_model = GaussianHMM(
        n_components=2, covariance_type="full", n_iter=200, random_state=42
    )
    hmm_model.fit(features_train_scaled)

    # Identify regimes (trending vs sideways)
    train_data["regime"] = hmm_model.predict(features_train_scaled)
    regime_stats = train_data.groupby("regime")[["volatility", "slope"]].mean()
    trending_regime = regime_stats["slope"].abs().idxmax()
    sideways_regime = 1 - trending_regime

    # 2. Test set
    test_data = data[data.index >= test_day].copy()
    if test_data.empty:
        raise ValueError("No data found for test_day. Check your date filter.")

    results = []

    # Rolling prediction for each minute
    for i in range(lookback, len(test_data)):
        window_slice = test_data.iloc[
            i - lookback : i + 1
        ].copy()  # last 60 minutes + current
        window_slice["log_return"] = np.log(
            window_slice["close"] / window_slice["close"].shift(1)
        )
        window_slice["volatility"] = window_slice["log_return"].rolling(
            window
        ).std() * np.sqrt(window)
        window_slice["slope"] = (
            window_slice["close"]
            .rolling(window)
            .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        )
        window_slice.dropna(inplace=True)

        if len(window_slice) == 0:
            continue

        features_window = window_slice[["log_return", "volatility", "slope"]].values
        features_window_scaled = scaler.transform(features_window)

        # Predict regime for the latest point in this window
        probs = hmm_model.predict_proba(features_window_scaled)[-1]
        regime = hmm_model.predict(features_window_scaled)[-1]

        results.append(
            {
                "ticker": test_data["ticker"].iloc[i],
                "timestamp": test_data.index[i],
                "close": test_data["close"].iloc[i],
                "regime": regime,
                "prob_sideways": probs[sideways_regime],
                "prob_trending": probs[trending_regime],
                "regime_label": "Trending" if regime == trending_regime else "Sideways",
            }
        )
def detect_regimes_train_test_rolling(
    data, train_start, train_end, test_day, lookback=60, rolling_window=15
):
    # Ensure timestamps are Timestamps
    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end)
    test_day = pd.to_datetime(test_day)

    # Slice train_data using .loc (inclusive)
    train_data = data.loc[train_start:train_end].copy()
    if train_data.empty:
        raise ValueError(f"Training data empty for range {train_start} → {train_end}")

    # Build training features
    train_data["log_return"] = np.log(train_data["close"] / train_data["close"].shift(1))
    train_data["volatility"] = train_data["log_return"].rolling(rolling_window).std() * np.sqrt(rolling_window)
    train_data["slope"] = train_data["close"].rolling(rolling_window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    train_data = train_data.dropna()
    if train_data.empty:
        raise ValueError("Training data becomes empty after computing features + dropna(). Increase lookback / check raw data.")

    features_train = train_data[["log_return", "volatility", "slope"]].values
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)

    # Fit HMM (catch exceptions)
    try:
        hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=42)
        hmm_model.fit(features_train_scaled)
    except Exception as e:
        logger.exception("HMM training failed: %s", e)
        raise

    # identify regimes
    train_data["regime"] = hmm_model.predict(features_train_scaled)
    regime_stats = train_data.groupby("regime")[["volatility", "slope"]].mean()
    trending_regime = regime_stats["slope"].abs().idxmax()
    sideways_regime = 1 - trending_regime

    # Prepare test_data: explicitly get the single day's rows
    day_start = test_day.normalize()
    day_end = day_start + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    test_data = data.loc[day_start:day_end].copy()

    if test_data.empty:
        logger.warning("No test data found for %s. Data available range: %s → %s", test_day.date(), data.index.min(), data.index.max())
        return pd.DataFrame()  # return empty DataFrame rather than exit

    # If test_data is shorter than lookback -> cannot compute rolling predictions
    if len(test_data) <= lookback:
        logger.warning(
            "Test day has %d rows which is <= lookback (%d). Need more intraday rows to compute rolling predictions.",
            len(test_data),
            lookback,
        )
        return pd.DataFrame()

    results = []
    # For each minute index >= lookback produce features on the last `rolling_window` minutes (or up to lookback)
    for i in range(lookback, len(test_data)):
        window_slice = test_data.iloc[max(0, i - lookback): i + 1].copy()
        # compute features for this window_slice; use the same rolling_window (or smaller if not enough)
        w = min(rolling_window, max(2, len(window_slice) - 1))

        window_slice["log_return"] = np.log(window_slice["close"] / window_slice["close"].shift(1))
        window_slice["volatility"] = window_slice["log_return"].rolling(w).std() * np.sqrt(w)
        window_slice["slope"] = window_slice["close"].rolling(w).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])

        window_slice = window_slice.dropna()
        if window_slice.empty:
            continue

        features_window = window_slice[["log_return", "volatility", "slope"]].values
        # scaler expects same number of features and at least one row
        if features_window.shape[0] == 0:
            continue

        features_window_scaled = scaler.transform(features_window)

        probs = hmm_model.predict_proba(features_window_scaled)[-1]
        regime = hmm_model.predict(features_window_scaled)[-1]

        ts = test_data.index[i]
        results.append(
            {
                "ticker": test_data["ticker"].iloc[i] if "ticker" in test_data.columns else None,
                "timestamp": ts,
                "close": test_data["close"].iloc[i],
                "regime": int(regime),
                "prob_sideways": float(probs[sideways_regime]),
                "prob_trending": float(probs[trending_regime]),
                "regime_label": "Trending" if regime == trending_regime else "Sideways",
            }
        )

    if not results:
        logger.warning("No regimes were appended during rolling loop. Check lookback/window and available intraday rows.")
        return pd.DataFrame()

    result_df = pd.DataFrame(results).set_index("timestamp")
    return result_df


# Save Every minute regime detected  for the ticker
def save_regime_detected(ticker):
    trading_date = pd.to_datetime(config.data.run_date)
    base_path = config.data.data_dir

    training_data, trading_data = load_hmm_data(
        base_path=base_path, trading_date=trading_date, ticker=ticker
    )

    logger.info("Training data shape: %s", getattr(training_data, "shape", None))
    logger.info("Trading data shape: %s", getattr(trading_data, "shape", None))

    if training_data.empty:
        logger.error("No training data for %s. Aborting.", ticker)
        return pd.DataFrame()

    if trading_data.empty:
        logger.error("No trading data for %s on %s. Aborting.", ticker, trading_date.date())
        return pd.DataFrame()

    final_df = pd.concat([training_data, trading_data], ignore_index=True)
    final_df["datetime"] = pd.to_datetime(final_df["datetime"])
    final_df = final_df.sort_values(["datetime"]).set_index("datetime")

    train_start = training_data["datetime"].min()
    train_end = training_data["datetime"].max()
    test_day = trading_date

    result = detect_regimes_train_test_rolling(
        final_df,
        train_start,
        train_end,
        test_day,
        lookback=config.regime_detection.lookback_minutes,
    )

    if result.empty:
        logger.warning("No regime results for %s. Nothing to save.", ticker)
        return result

    regime_dir = config.data.output_dir / "regime_detection" / str(config.data.run_date)
    regime_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(regime_dir / f"{ticker}_regimes_{trading_date.date()}.csv")
    return result
