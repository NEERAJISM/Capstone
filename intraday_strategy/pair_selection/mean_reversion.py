from itertools import combinations
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import numpy as np
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from joblib import Parallel, delayed

from common import StockDataLoader
from common import get_logger


class MeanReversionAnalyzer:
    """
    Analyzes stocks to find mean-reverting pairs for statistical arbitrage.

    The process involves:
    1. Filtering stocks by volatility.
    2. Computing features (mean reversion, volatility, autocorrelation).
    3. Clustering stocks based on features.
    4. Finding cointegrated pairs within clusters.
    5. Analyzing mean-reversion score of the spread.
    """

    def __init__(
        self,
        base_dir: str,
        run_date: str,
        lookback: int,
        trading_start: str = "09:15:00",
        trading_end: str = "15:30:00",
        tickers: Optional[List[str]] = None,
        min_mean_reversion: float = 0.05,
    ):
        """
        Initializes the MeanReversionAnalyzer.

        Args:
            base_dir: Base directory for stock data.
            run_date: Date for the analysis.
            lookback: Number of days for historical data.
            trading_start: Trading day start time.
            trading_end: Trading day end time.
            tickers: List of tickers to analyze. If None, all are used.
            min_mean_reversion: Minimum mean-reversion score for a pair.
        """
        self.base_dir = base_dir
        self.run_date = datetime.strptime(run_date, "%Y-%m-%d")
        self.end_day = self.run_date
        self.start_day = self.run_date - timedelta(days=lookback)
        self.start_time = datetime.strptime(trading_start, "%H:%M:%S").time()
        self.end_time = datetime.strptime(trading_end, "%H:%M:%S").time()
        self.tickers = tickers
        self.logger = get_logger(__name__)

        self.min_mean_reversion = min_mean_reversion
        self._log_initial_config()

    def _log_initial_config(self):
        """Logs the initial configuration of the analyzer."""
        self.logger.info("=" * 50)
        self.logger.info("        PAIR TRADING ANALYZER CONFIGURATION")
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

    def filter_basic(self, volatility_threshold: float = 0.005) -> List[str]:
        """
        Filters tickers by a minimum volatility threshold.

        Args:
            volatility_threshold: Minimum average daily volatility.

        Returns:
            A list of tickers passing the filter.
        """
        loader = StockDataLoader(
            tickers=self.tickers,
            start=f"{self.start_day.strftime('%Y-%m-%d')} {self.start_time}",
            end=f"{self.end_day.strftime('%Y-%m-%d')} {self.end_time}",
            select_columns=["close"],
            base_dir=self.base_dir,
        )
        data_dict = loader.get_data_for_tickers()
        self.data_dict = data_dict  # cache data to avoid reloading
        filtered = []

        for ticker, df in data_dict.items():
            if df.is_empty():
                continue
            returns = df["close"].pct_change().abs()
            avg_volatility = returns.mean()
            if avg_volatility is not None and avg_volatility >= volatility_threshold:
                filtered.append(ticker)
        self.filtered_tickers = filtered
        self.logger.info(f"Tickers after basic filtering: {filtered}")
        return filtered

    def compute_features(self) -> Dict[str, np.ndarray]:
        """
        Computes features for each stock (mean reversion, volatility, autocorrelation).

        Returns:
            A dictionary of tickers and their feature vectors.
        """
        features = {}
        for ticker in self.filtered_tickers:
            df = self.data_dict[ticker]
            if df.is_empty():
                continue
            prices = df["close"].to_numpy()
            returns = np.diff(prices) / prices[:-1]

            # Feature vector: mean-reversion, volatility, autocorrelation
            rolling_mean = np.convolve(prices, np.ones(5) / 5, mode="valid")
            deviations = prices[4:] - rolling_mean
            mean_rev_score = np.mean(np.abs(deviations))
            vol_score = np.mean(np.abs(returns))
            autocorr_score = np.corrcoef(returns[:-1], returns[1:])[0, 1]

            features[ticker] = np.array([mean_rev_score, vol_score, autocorr_score])
        self.features = features
        return features

    def cluster_stocks(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Clusters stocks based on their features.

        Args:
            n_clusters: The number of clusters to create.

        Returns:
            A dictionary of cluster IDs and the tickers in each cluster.
        """
        if not hasattr(self, "features") or not self.features:
            self.logger.warning("No features computed. Skipping clustering.")
            self.clusters = {}
            return {}

        tickers = list(self.features.keys())
        X = np.array(list(self.features.values()))

        # Handle case with fewer samples than clusters
        if len(tickers) < n_clusters:
            self.logger.warning(
                f"Number of tickers ({len(tickers)}) is less than n_clusters ({n_clusters}). Setting n_clusters to {len(tickers)}."
            )
            n_clusters = len(tickers)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        clusters = {i: [] for i in range(n_clusters)}
        for ticker, label in zip(tickers, labels):
            clusters[label].append(ticker)
        self.clusters = clusters
        return clusters

    def _check_pair(self, t1: str, t2: str) -> Optional[List[str]]:
        """
        Checks if a stock pair is cointegrated with a sufficient mean-reversion score.

        Args:
            t1: First ticker.
            t2: Second ticker.

        Returns:
            A dictionary with pair properties if it's a good candidate, else None.
        """
        df1 = self.data_dict.get(t1)
        df2 = self.data_dict.get(t2)

        if df1 is None or df1.is_empty() or df2 is None or df2.is_empty():
            return None

        prices1 = df1["close"].to_numpy()
        prices2 = df2["close"].to_numpy()

        # Ensure same length
        min_len = min(len(prices1), len(prices2))
        prices1 = prices1[-min_len:]
        prices2 = prices2[-min_len:]

        # Beta hedge ratio
        X = add_constant(prices2)
        model = OLS(prices1, X).fit()
        beta = model.params[1]

        # Spread
        spread = prices1 - beta * prices2

        # Cointegration test
        score, pvalue, _ = coint(prices1, prices2)
        if pvalue >= 0.05:
            return None

        # Stationarity test
        adf_result = adfuller(spread)
        if adf_result[1] >= 0.05:
            return None

        # Minimum mean-reversion filter
        mean_rev_score = np.mean(np.abs(spread - np.mean(spread)))
        if mean_rev_score < self.min_mean_reversion:
            return None

        return {
            "tickers": [t1, t2],
            "beta": float(beta),
            "mean_reversion_score": float(mean_rev_score),
        }

    def find_pairs_in_cluster(
        self, cluster_tickers: List[str], n_jobs: int = -1
    ) -> List[List[str]]:
        """
        Finds all cointegrated pairs within a stock cluster.

        Args:
            cluster_tickers: A list of tickers in the cluster.
            n_jobs: Number of parallel jobs to run (-1 for all CPUs).

        Returns:
            A list of all cointegrated pairs in the cluster.
        """

        # Generate all unique ticker pairs
        ticker_pairs = list(combinations(cluster_tickers, 2))
        self.logger.info(
            f"Checking {len(ticker_pairs)} pairs in cluster with {len(cluster_tickers)} tickers."
        )

        # Use joblib to parallelize the _check_pair function
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._check_pair)(t1, t2) for t1, t2 in ticker_pairs
        )

        # Filter out None results
        pairs = [res for res in results if res is not None]

        self.logger.info(f"Found {len(pairs)} cointegrated pairs.")
        return pairs

    def analyze(
        self, volatility_threshold: float = 0.001, n_clusters: int = 5, n_jobs: int = -1
    ) -> Dict[str, Dict]:
        """
        Runs the full analysis pipeline to find cointegrated pairs.

        Args:
            volatility_threshold: Minimum average daily volatility for a stock.
            n_clusters: Number of clusters to create.
            n_jobs: The number of parallel jobs to run. -1 means using all available CPUs.

        Returns:
            A dictionary with cluster IDs, tickers, and cointegrated pairs.
        """
        self.filter_basic(volatility_threshold)
        self.compute_features()
        self.cluster_stocks(n_clusters=n_clusters)

        def process_cluster(cluster_id, tickers):
            if len(tickers) < 2:
                return None
            pairs = self.find_pairs_in_cluster(
                tickers, n_jobs=1
            )  # Run inner loop sequentially
            return f"cluster_{cluster_id}", {"tickers": tickers, "pairs": pairs}

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_cluster)(cluster_id, tickers)
            for cluster_id, tickers in self.clusters.items()
        )
        results = list(filter(lambda x: x is not None, results))

        output = {cluster_id: data for cluster_id, data in results if data is not None}
        return output
