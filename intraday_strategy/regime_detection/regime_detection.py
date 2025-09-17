from functools import lru_cache
import os
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import numpy as np
from common import get_logger

logger = get_logger()

def load_hmm_data(base_path, trading_date, ticker):

    trading_date = pd.to_datetime(trading_date)
    three_months_back = trading_date - relativedelta(months=3)

    # Collect months in range
    months = pd.date_range(
        start=three_months_back.replace(day=1),
        end=trading_date,
        freq="MS"
    )

    all_data = []
    for month in months:
        year = month.year
        month_name = month.strftime("%B")
        folder_name = f"Cash Data {month_name} {year}"
        nested_path = os.path.join(
            base_path, str(year), folder_name, f"{ticker}.csv"
        )
        nested_path = nested_path.replace("\\", "/")
        logger.info(nested_path)

        if os.path.exists(nested_path):
            df = pd.read_csv(nested_path)

            # Clean column names (remove < >)
            df.columns = [c.strip("<>").lower() for c in df.columns]

            # Combine date + time into DateTime
            df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"])

            all_data.append(df)

    if not all_data:
        raise FileNotFoundError(f"No data found for {ticker} in last 3 months")

    data = pd.concat(all_data).sort_values("datetime")

    # Training = last 3 months up to *day before trading_date*
    training_data = data[data["datetime"] < trading_date]

    # Trading = only the trading_date
    trading_data = data[data["datetime"].dt.date == trading_date.date()]

    return training_data, trading_data


# def load_hmm_data(config):
    
#             loader = StockDataLoader(
#                 base_dir=config.data.data_dir,
#                 start=start_date.strftime("%Y-%m-%d %H:%M:%S"),
#                 end=end_date.strftime("%Y-%m-%d %H:%M:%S"),
#                 tickers=[stock_a, stock_b],
#                 select_columns=["close"],
#                 impute=True
#             )



def detect_regimes_train_test_rolling(data, train_start, train_end, test_day, lookback=60):
    # 1. Training set
    train_data = data[(data.index >= train_start) & (data.index <= train_end)].copy()
    if train_data.empty:
        raise ValueError("Training data is empty. Check train_start/train_end dates.")

    # Training features
    train_data['log_return'] = np.log(train_data['close'] / train_data['close'].shift(1))
    window = 15
    train_data['volatility'] = train_data['log_return'].rolling(window).std() * np.sqrt(window)
    train_data['slope'] = train_data['close'].rolling(window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0]
    )
    train_data.dropna(inplace=True)

    features_train = train_data[['log_return', 'volatility', 'slope']].values
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)

    # Train HMM
    hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=200, random_state=42)
    hmm_model.fit(features_train_scaled)

    # Identify regimes (trending vs sideways)
    train_data['regime'] = hmm_model.predict(features_train_scaled)
    regime_stats = train_data.groupby('regime')[['volatility', 'slope']].mean()
    trending_regime = regime_stats['slope'].abs().idxmax()
    sideways_regime = 1 - trending_regime

    # 2. Test set
    test_data = data[data.index >= test_day].copy()
    if test_data.empty:
        raise ValueError("No data found for test_day. Check your date filter.")

    results = []

    # Rolling prediction for each minute
    for i in range(lookback, len(test_data)):
        window_slice = test_data.iloc[i - lookback:i + 1].copy()  # last 60 minutes + current
        window_slice['log_return'] = np.log(window_slice['close'] / window_slice['close'].shift(1))
        window_slice['volatility'] = window_slice['log_return'].rolling(window).std() * np.sqrt(window)
        window_slice['slope'] = window_slice['close'].rolling(window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        window_slice.dropna(inplace=True)

        if len(window_slice) == 0:
            continue

        features_window = window_slice[['log_return', 'volatility', 'slope']].values
        features_window_scaled = scaler.transform(features_window)

        # Predict regime for the latest point in this window
        probs = hmm_model.predict_proba(features_window_scaled)[-1]
        regime = hmm_model.predict(features_window_scaled)[-1]

        results.append({
            "ticker": test_data['ticker'].iloc[i],
            "timestamp": test_data.index[i],
            "close": test_data['close'].iloc[i],
            "regime": regime,
            "prob_sideways": probs[sideways_regime],
            "prob_trending": probs[trending_regime],
            "regime_label": "Trending" if regime == trending_regime else "Sideways"
        })

    result_df = pd.DataFrame(results).set_index("timestamp")
    return result_df


# Save Every minute regime detected  for the ticker
def save_regime_detected(config, ticker):
    trading_date = config.data.run_date
    base_path = config.data.data_dir

    training_data, trading_data = load_hmm_data(
        base_path=base_path,
        trading_date=trading_date,
        ticker=ticker
    )

    logger.info("Training data shape:", training_data.shape)
    logger.info("Trading data shape:", trading_data.shape)
    logger.info("Trading day time range:", trading_data["datetime"].min(), "to", trading_data["datetime"].max())
    logger.info("Training day time range:", training_data["datetime"].min(), "to", training_data["datetime"].max())

    final_df = pd.concat([training_data, trading_data])
    final_df['datetime'] = pd.to_datetime(final_df['datetime'])
    final_df.sort_values(['ticker', 'datetime'], inplace=True)
    final_df.set_index('datetime', inplace=True)

    # Train on 3 months
    train_start = training_data["datetime"].min()
    train_end = training_data["datetime"].max()
    test_day = trading_date

    result = detect_regimes_train_test_rolling(final_df, train_start, train_end, test_day, lookback=60)
    result.to_csv( config.data.output_dir / "regime_detected" / f"{ticker}_regimes_{trading_date}.csv")

    return result