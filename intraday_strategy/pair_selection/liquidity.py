from datetime import datetime, timedelta
from typing import List, Optional, Dict

import numpy as np
import polars as pl
from common import get_logger
from common import StockDataLoader


class LiquidityPredictor:
    """
    Analyzes and filters stocks based on liquidity.

    Provides basic and advanced filtering methods to rank stocks by liquidity using
    a composite score of volume, trading frequency, Amihud illiquidity, and volume consistency.
    """

    def __init__(
        self,
        base_dir: str,
        run_date: str,
        lookback: int,
        trading_start: str = "09:15:00",
        trading_end: str = "15:30:00",
        tickers: Optional[List[str]] = None,
    ):
        """
        Initializes the LiquidityPredictor.

        Args:
            base_dir: Base directory for stock data.
            run_date: Date for the analysis.
            lookback: Number of days for historical data.
            trading_start: Trading day start time.
            trading_end: Trading day end time.
            tickers: List of tickers to analyze. If None, all are used.
        """
        self.logger = get_logger(__name__)
        self.base_dir = base_dir

        # Convert run_date to datetime
        self.run_date = datetime.strptime(run_date, "%Y-%m-%d")

        # Set start and end dates implicitly
        self.end_day = self.run_date
        self.start_day = self.run_date - timedelta(days=lookback)

        # Market timings
        self.start_time = datetime.strptime(trading_start, "%H:%M:%S").time()
        self.end_time = datetime.strptime(trading_end, "%H:%M:%S").time()

        # Stock tickers (optional)
        self.tickers = tickers

        # Logging for debugging
        self._log_initial_config()

    def _log_initial_config(self):
        """Logs the initial configuration of the analyzer."""
        self.logger.info("=" * 50)
        self.logger.info("        LIQUIDITY PREDICTOR CONFIGURATION")
        self.logger.info("=" * 50)
        self.logger.info(f"Base Directory     : {self.base_dir}")
        self.logger.info(f"Run Date           : {self.run_date.date()}")
        self.logger.info(f"Start Day          : {self.start_day.date()}")
        self.logger.info(f"End Day            : {self.end_day.date()}")
        self.logger.info(f"Trading Window     : {self.start_time} → {self.end_time}")
        self.logger.info(
            f"Tickers            : {self.tickers if self.tickers else 'Using default universe'}"
        )
        self.logger.info("=" * 50)

    def filter_basic(self, volume_threshold: int = 1000) -> List[str]:
        """
        Filters tickers by a minimum average trading volume.

        Args:
            volume_threshold: Minimum average daily trading volume.

        Returns:
            A list of tickers passing the filter.
        """
        self.logger.info(
            f"Running basic filter with volume threshold: {volume_threshold}"
        )
        loader = StockDataLoader(
            tickers=self.tickers,
            start=f"{self.start_day.strftime('%Y-%m-%d')} {self.start_time}",
            end=f"{self.end_day.strftime('%Y-%m-%d')} {self.end_time}",
            select_columns=["volume"],
            base_dir=self.base_dir,
        )
        data_dict = loader.get_data_for_tickers()
        filtered = []

        for ticker, df in data_dict.items():
            if not df.is_empty() and df["volume"].fill_null(0).mean() >= volume_threshold:
                filtered.append(ticker)

        self.logger.info(f"Found {len(filtered)} tickers passing basic filter.")
        return filtered

    def filter_advanced(
        self,
        tickers: List[str],
        custom_weights: Optional[Dict[str, float]] = None,
        min_score: float = 30.0,
    ) -> List[str]:
        """
        Filters and ranks tickers by a composite liquidity score.

        Args:
            tickers: List of tickers to filter and rank.
            custom_weights: Custom weights for liquidity indicators.
            min_score: Minimum liquidity score for a stock to be included.

        Returns:
            A list of tickers passing the filter, ranked by liquidity.
        """
        self.logger.info(f"Running advanced filter with min score: {min_score}")
        if custom_weights is None:
            custom_weights = {
                "average_volume": 0.4,
                "trading_frequency": 0.3,
                "amihud_liquidity": 0.2,
                "volume_consistency": 0.1,
            }

        loader = StockDataLoader(
            tickers=tickers,
            start=f"{self.start_day.strftime('%Y-%m-%d')} {self.start_time}",
            end=f"{self.end_day.strftime('%Y-%m-%d')} {self.end_time}",
            select_columns=["volume", "close", "open"],
            base_dir=self.base_dir,
        )
        data_dict = loader.get_data_for_tickers()

        scores = []
        for ticker, df in data_dict.items():
            score = self._compute_weighted_score(df, custom_weights)
            if score is not None and score >= min_score:
                scores.append((ticker, score))

        # Sort by descending score
        scores.sort(key=lambda x: x[1], reverse=True)

        result_tickers = [t[0] for t in scores]
        self.logger.info(
            f"Found {len(result_tickers)} tickers passing advanced filter."
        )
        return result_tickers

    def _compute_weighted_score(
        self, df: pl.DataFrame, weights: Dict[str, float]
    ) -> Optional[float]:
        """
        Computes the weighted liquidity score for a stock.

        Args:
            df: Historical data for the stock.
            weights: Weights for the liquidity indicators.

        Returns:
            The weighted liquidity score.
        """

        avg_volume = self._compute_average_volume_score(df)
        trading_freq = self._compute_trading_frequency_score(df)
        amihud = self._compute_amihud_liquidity_score(df)
        volume_consistency = self._compute_volume_consistency_score(df)

        if any(
            v is None for v in [avg_volume, trading_freq, amihud, volume_consistency]
        ):
            return None

        return (
            avg_volume * weights["average_volume"]
            + trading_freq * weights["trading_frequency"]
            + amihud * weights["amihud_liquidity"]
            + volume_consistency * weights["volume_consistency"]
        )

    def _compute_average_volume_score(self, df: pl.DataFrame) -> Optional[float]:
        """Computes the average volume score (log-normalized)."""
        if df.is_empty():
            return None
        mean = df["volume"].mean()
        if mean is None or mean <= 0:
            return 0.0
        return min(100, max(0, (np.log10(mean) - 2) / 5 * 100))

    def _compute_trading_frequency_score(self, df: pl.DataFrame) -> Optional[float]:
        """Computes the trading frequency score (percentage of active trading periods)."""
        if df.is_empty():
            return None
        total = df.height
        if total == 0:
            return 0.0
        active = df.filter(pl.col("volume") > 0).height
        return active / total * 100

    def _compute_amihud_liquidity_score(self, df: pl.DataFrame) -> Optional[float]:
        """
        Computes the Amihud illiquidity score.

        Amihud illiquidity = average( |return| / volume ).
        Higher values mean less liquid.
        We invert & scale to create a liquidity score between 0 and 100.
        """
        if df.is_empty():
            return None

        # Compute absolute returns
        df = df.with_columns(
            ((pl.col("close") / pl.col("close").shift(1) - 1).abs()).alias("return")
        )

        # Filter valid volume
        df = df.filter(pl.col("volume") > 0)

        if df.is_empty():
            return None

        # Amihud ratio
        df = df.with_columns((pl.col("return") / pl.col("volume")).alias("amihud"))

        illiq = df["amihud"].mean()
        if illiq is None or np.isnan(illiq):
            return None

        # Score inversion and scaling (clip between 0 and 100)
        score = 100 - (np.log10(illiq + 1e-12) * 20)  # avoid log(0)
        return float(np.clip(score, 0, 100))

    def _compute_volume_consistency_score(self, df: pl.DataFrame) -> Optional[float]:
        """Computes the volume consistency score (based on the coefficient of variation)."""
        if df.is_empty() or "volume" not in df.columns:
            return None
        mean = df["volume"].mean()
        std = df["volume"].std()

        if mean is None or std is None or mean == 0:
            return 0.0

        cv = std / mean
        return 100 - cv * 30 if cv <= 1 else 70 * np.exp(-(cv - 1) * 0.5)
