import os
import sys
import re
import glob
import json
import pickle
import random
import calendar
from pathlib import Path
from itertools import chain, combinations, islice
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Optional, Callable, Dict, Union, Tuple

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed

from ...common.data_loader import StockDataLoader
from ...common.utils import get_logger

logger = get_logger(__name__)

class PairTradingAnalyzer:
    """
    Analyzes a list of stocks to find pairs that exhibit mean-reverting behavior.

    This class implements a multi-step process to identify pairs of stocks that are suitable for
    statistical arbitrage strategies based on mean reversion. The process includes:

    1.  **Basic Filtering**: A preliminary filtering of stocks based on their price volatility.
        This step removes stocks that do not exhibit sufficient price movement to be interesting
        for a trading strategy.

    2.  **Feature Computation**: For the remaining stocks, the class computes a set of features
        that characterize their price behavior. These features include measures of mean
        reversion, volatility, and autocorrelation.

    3.  **Clustering**: The stocks are then clustered into groups based on their computed
        features. This is done to group together stocks with similar characteristics, which
        can improve the efficiency of the pair-finding process.

    4.  **Pair Finding**: Within each cluster, the class searches for pairs of stocks that are
        cointegrated. Cointegration is a statistical property that suggests a long-run
        equilibrium relationship between the two stocks. The class uses the Johansen test
        to test for cointegration.

    5.  **Mean Reversion Analysis**: For each cointegrated pair, the class computes the
        mean-reversion score of the spread between the two stocks. A high mean-reversion
        score indicates that the spread tends to revert to its historical mean, which is the
        key property exploited in a mean-reversion trading strategy.

    The class is designed to be used in conjunction with the `LiquidityAnalyzer` class, which
    provides a list of liquid stocks to be analyzed.
    """
    def __init__(
        self,
        base_dir: str,
        run_date: str,
        lookback: int,
        trading_start: str = "09:15:00",
        trading_end: str = "15:30:00",
        tickers: Optional[List[str]] = None,
        min_mean_reversion: float = 0.05   
    ):
        """
        Initializes the PairTradingAnalyzer.

        Args:
            base_dir: The base directory where the stock data is located.
            run_date: The date for which the analysis is to be run.
            lookback: The number of days to look back for historical data.
            trading_start: The start time of the trading day.
            trading_end: The end time of the trading day.
            tickers: A list of tickers to analyze. If None, all available tickers will be used.
            min_mean_reversion: The minimum mean-reversion score for a pair to be considered.
        """
        self.base_dir = base_dir
        self.run_date = datetime.strptime(run_date, "%Y-%m-%d")
        self.end_day = self.run_date
        self.start_day = self.run_date - timedelta(days=lookback)
        self.start_time = datetime.strptime(trading_start, "%H:%M:%S").time()
        self.end_time = datetime.strptime(trading_end, "%H:%M:%S").time()
        self.tickers = tickers
        self.min_mean_reversion = min_mean_reversion
        self._log_initial_config()


    def _log_initial_config(self):
        """Logs the initial configuration of the analyzer."""
        logger.info("=" * 50)
        logger.info("        PAIR TRADING ANALYZER CONFIGURATION")
        logger.info("=" * 50)
        logger.info(f"Base Directory     : {self.base_dir}")
        logger.info(f"Run Date           : {self.run_date.date()}")
        logger.info(f"Start Day          : {self.start_day.date()}")
        logger.info(f"End Day            : {self.end_day.date()}")
        logger.info(f"Trading Window     : {self.start_time} → {self.end_time}")
        logger.info(f"Tickers            : {self.tickers if self.tickers else 'Using default universe'}")
        logger.info("=" * 50)

    def filter_basic(self, volatility_threshold: float = 0.005) -> List[str]:
        """
        Filters the tickers based on a minimum volatility threshold.

        This method removes stocks that do not exhibit sufficient price movement to be interesting
        for a trading strategy.

        Args:
            volatility_threshold: The minimum average daily volatility for a stock to be included.

        Returns:
            A list of tickers that pass the volatility filter.
        """
        loader = StockDataLoader(
            tickers=self.tickers,
            start=f"{self.start_day.strftime('%Y-%m-%d')} {self.start_time}",
            end=f"{self.end_day.strftime('%Y-%m-%d')} {self.end_time}",
            select_columns=["close"],
            base_dir=self.base_dir
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
        logger.info(f"Tickers after basic filtering: {filtered}")
        return filtered

    def compute_features(self) -> Dict[str, np.ndarray]:
        """
        Computes a set of features for each stock that characterize its price behavior.

        The features include measures of mean reversion, volatility, and autocorrelation.

        Returns:
            A dictionary where the keys are the tickers and the values are the feature vectors.
        """
        features = {}
        for ticker in self.filtered_tickers:
            df = self.data_dict[ticker]
            if df.is_empty():
                continue
            prices = df["close"].to_numpy()
            returns = np.diff(prices) / prices[:-1]

            # Feature vector: mean-reversion, volatility, autocorrelation
            rolling_mean = np.convolve(prices, np.ones(5)/5, mode='valid')
            deviations = prices[4:] - rolling_mean
            mean_rev_score = np.mean(np.abs(deviations))
            vol_score = np.mean(np.abs(returns))
            autocorr_score = np.corrcoef(returns[:-1], returns[1:])[0, 1]

            features[ticker] = np.array([mean_rev_score, vol_score, autocorr_score])
        self.features = features
        return features

    def cluster_stocks(self, n_clusters: int = 5) -> Dict[int, List[str]]:
        """
        Clusters the stocks into groups based on their computed features.

        This is done to group together stocks with similar characteristics, which can improve the
        efficiency of the pair-finding process.

        Args:
            n_clusters: The number of clusters to create.

        Returns:
            A dictionary where the keys are the cluster IDs and the values are the lists of tickers
            in each cluster.
        """
        if not hasattr(self, 'features') or not self.features:
            logger.warning("No features computed. Skipping clustering.")
            self.clusters = {}
            return {}
            
        tickers = list(self.features.keys())
        X = np.array(list(self.features.values()))
        
        # Handle case with fewer samples than clusters
        if len(tickers) < n_clusters:
            logger.warning(f"Number of tickers ({len(tickers)}) is less than n_clusters ({n_clusters}). Setting n_clusters to {len(tickers)}.")
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
        Checks if a pair of stocks is cointegrated and has a sufficient mean-reversion score.

        Args:
            t1: The first ticker in the pair.
            t2: The second ticker in the pair.

        Returns:
            A dictionary containing the pair's properties if it is a good candidate, otherwise None.
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

        return  {
            "tickers": [t1, t2],
            "beta": float(beta),
            "mean_reversion_score": float(mean_rev_score)
        }

    def find_pairs_in_cluster(self, cluster_tickers: List[str], n_jobs: int = -1) -> List[List[str]]:
        """
        Finds all cointegrated pairs within a cluster of stocks.

        Args:
            cluster_tickers: A list of tickers in the cluster.
            n_jobs: The number of parallel jobs to run. -1 means using all available CPUs.

        Returns:
            A list of all cointegrated pairs found in the cluster.
        """

        # Generate all unique ticker pairs
        ticker_pairs = list(combinations(cluster_tickers, 2))
        logger.info(f"Checking {len(ticker_pairs)} pairs in cluster with {len(cluster_tickers)} tickers.")

        # Use joblib to parallelize the _check_pair function
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._check_pair)(t1, t2) for t1, t2 in ticker_pairs
        )

        # Filter out None results
        pairs = [res for res in results if res is not None]

        logger.info(f"Found {len(pairs)} cointegrated pairs.")
        return pairs


    def analyze(self, volatility_threshold: float = 0.001, n_clusters: int = 5) -> Dict[str, Dict]:
        """
        Runs the full analysis pipeline to find cointegrated pairs.

        The pipeline includes basic filtering, feature computation, clustering, and pair finding.

        Args:
            volatility_threshold: The minimum average daily volatility for a stock to be included.
            n_clusters: The number of clusters to create.

        Returns:
            A dictionary containing the results of the analysis. The keys are the cluster IDs, and
            the values are dictionaries containing the tickers in the cluster and the cointegrated
            pairs found in the cluster.
        """
        self.filter_basic(volatility_threshold)
        self.compute_features()
        self.cluster_stocks(n_clusters=n_clusters)

        output = {}
        for cluster_id, tickers in self.clusters.items():
            if len(tickers) < 2:
                continue
            pairs = self.find_pairs_in_cluster(tickers, n_jobs=1)  # sequential
            output[f"cluster_{cluster_id}"] = {"tickers": tickers, "pairs": pairs}

        return output
