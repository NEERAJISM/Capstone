from datetime import datetime, timedelta
from typing import List, Optional, Dict

import numpy as np
import polars as pl
from common import get_logger
from common import StockDataLoader

class LiquidityPredictor:
    """
    Analyzes the liquidity of a list of stocks.

    This class provides methods for filtering and ranking stocks based on their liquidity.
    It uses a two-stage filtering process:

    1.  **Basic Filtering**: A preliminary filtering of stocks based on their average trading
        volume. This step removes stocks that are clearly illiquid.

    2.  **Advanced Filtering**: A more sophisticated filtering and ranking of stocks based on a
        composite liquidity score. This score is a weighted average of several liquidity
        indicators, including:
        -   Average trading volume
        -   Trading frequency
        -   Amihud illiquidity measure
        -   Volume consistency

    The class is designed to be used as a preliminary step in a quantitative trading
    strategy, to ensure that the selected stocks are sufficiently liquid to be traded
    without significant market impact.
    """
    def __init__(
        self,
        base_dir: str,
        run_date: str,
        lookback: int,
        trading_start: str = "09:15:00",
        trading_end: str = "15:30:00",
        tickers: Optional[List[str]] = None
    ):
        """
        Initializes the LiquidityPredictor.

        Args:
            base_dir: The base directory where the stock data is located.
            run_date: The date for which the analysis is to be run.
            lookback: The number of days to look back for historical data.
            trading_start: The start time of the trading day.
            trading_end: The end time of the trading day.
            tickers: A list of tickers to analyze. If None, all available tickers will be used.
        """
        self.logger = get_logger()
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
        self.logger.info(f"Tickers            : {self.tickers if self.tickers else 'Using default universe'}")
        self.logger.info("=" * 50)

    def filter_basic(self, volume_threshold: int = 1000) -> List[str]:
        """
        Filters the tickers based on a minimum average trading volume.

        This method removes stocks that are clearly illiquid.

        Args:
            volume_threshold: The minimum average daily trading volume for a stock to be included.

        Returns:
            A list of tickers that pass the volume filter.
        """
        self.logger.info(f"Running basic filter with volume threshold: {volume_threshold}")
        loader = StockDataLoader(
            tickers=self.tickers,
            start=f"{self.start_day.strftime('%Y-%m-%d')} {self.start_time}",
            end=f"{self.end_day.strftime('%Y-%m-%d')} {self.end_time}",
            select_columns=["volume"],
            base_dir=self.base_dir
        )
        data_dict = loader.get_data_for_tickers()
        filtered = []

        for ticker, df in data_dict.items():
            if not df.is_empty() and df["volume"].mean() >= volume_threshold:
                filtered.append(ticker)
        
        self.logger.info(f"Found {len(filtered)} tickers passing basic filter.")
        return filtered

    def filter_advanced(
        self,
        tickers: List[str],
        custom_weights: Optional[Dict[str, float]] = None,
        min_score: float = 30.0
    ) -> List[str]:
        """
        Filters and ranks the tickers based on a composite liquidity score.

        This method uses a weighted average of several liquidity indicators to compute a
        composite liquidity score for each stock. The stocks are then ranked based on this
        score, and only those that meet a minimum score are returned.

        Args:
            tickers: The list of tickers to be filtered and ranked.
            custom_weights: A dictionary of custom weights for the liquidity indicators. If None,
                default weights will be used.
            min_score: The minimum liquidity score for a stock to be included.

        Returns:
            A list of tickers that pass the advanced liquidity filter, ranked from most to
            least liquid.
        """
        self.logger.info(f"Running advanced filter with min score: {min_score}")
        if custom_weights is None:
            custom_weights = {
                "average_volume": 0.4,
                "trading_frequency": 0.3,
                "amihud_liquidity": 0.2,
                "volume_consistency": 0.1
            }

        loader = StockDataLoader(
            tickers=tickers,
            start=f"{self.start_day.strftime('%Y-%m-%d')} {self.start_time}",
            end=f"{self.end_day.strftime('%Y-%m-%d')} {self.end_time}",
            select_columns=["volume", "close", "open"],
            base_dir=self.base_dir
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
        self.logger.info(f"Found {len(result_tickers)} tickers passing advanced filter.")
        return result_tickers

    def _compute_weighted_score(self, df: pl.DataFrame, weights: Dict[str, float]) -> Optional[float]:
        """
        Computes the weighted liquidity score for a single stock.

        Args:
            df: The historical data for the stock.
            weights: A dictionary of weights for the liquidity indicators.

        Returns:
            The weighted liquidity score.
        """

        avg_volume = self._compute_average_volume_score(df)
        trading_freq = self._compute_trading_frequency_score(df)
        amihud = self._compute_amihud_liquidity_score(df)
        volume_consistency = self._compute_volume_consistency_score(df)

        if any(v is None for v in [avg_volume, trading_freq, amihud, volume_consistency]):
            return None

        return (
            avg_volume * weights["average_volume"] +
            trading_freq * weights["trading_frequency"] +
            amihud * weights["amihud_liquidity"] +
            volume_consistency * weights["volume_consistency"]
        )

    def _compute_average_volume_score(self, df: pl.DataFrame) -> Optional[float]:
        """Computes the average volume score (log-normalized)."""
        if df.is_empty() or "volume" not in df.columns:
            return None
        mean = df["volume"].mean()
        if mean is None or mean <= 0:
            return 0.0
        return min(100, max(0, (np.log10(mean) - 2) / 5 * 100))

    def _compute_trading_frequency_score(self, df: pl.DataFrame) -> Optional[float]:
        """Computes the trading frequency score (percentage of active trading periods)."""
        if df.is_empty() or "volume" not in df.columns:
            return None
        total = df.height
        if total == 0:
            return 0.0
        active = df.filter(pl.col("volume") > 0).height
        return active / total * 100

    def _compute_amihud_liquidity_score(self, df: pl.DataFrame) -> Optional[float]:
        """
        Computes the Amihud illiquidity score.

        The Amihud illiquidity measure is defined as the average ratio of the absolute
        daily return to the daily trading volume.
        """
        if df.is_empty() or "close" not in df.columns or "volume" not in df.columns:
            return None
            
        df = df.with_columns((pl.col("close").pct_change().abs()).alias("return"))    
        df = df.filter(pl.col("volume") > 0)

        if df.is_empty():
            return None

        df = df.with_columns((pl.col("return") / pl.col("volume")).alias("amihud"))
        illiq = df["amihud"].mean()
        
        if illiq is None:
            return None
            
        # Score inversion and scaling
        return max(0, 100 - (np.log10(illiq * 1e6) * 20))


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