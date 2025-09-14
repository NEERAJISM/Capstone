import json
from pathlib import Path
from typing import List
from .liquidity import LiquidityPredictor
from .mean_reversion import PairTradingAnalyzer
from common import get_logger

logger = get_logger()


def generate_pair_json(
    base_dir: str = "../../downloaded_files",
    run_date: str = "2021-04-01",
    lookback: int = 60,
    tickers_universe: List[str] = [],
    volume_threshold: int = 1000,
    min_mean_reversion: float = 0.04,
    volatility_threshold: float = 0.001,
    n_clusters_pairs: int = 5,
    output_dir: str = ".",
):
    """
    Identifies and analyzes potential pairs for statistical arbitrage strategies.

    1. Filters for liquid stocks.
    2. Analyzes pairs for cointegration and mean-reversion.
    3. Saves results to a JSON file.
    """

    # Liquidity Filtering
    predictor = LiquidityPredictor(
        base_dir=base_dir,
        lookback=lookback,
        run_date=run_date,
        tickers=tickers_universe,
    )
    basic_tickers = predictor.filter_basic(volume_threshold=volume_threshold)
    logger.info("Tickers passing basic volume filter: %s", basic_tickers)

    liquid_tickers = predictor.filter_advanced(
        tickers=basic_tickers,
        custom_weights={
            "average_volume": 0.4,
            "trading_frequency": 0.3,
            "amihud_liquidity": 0.2,
            "volume_consistency": 0.1,
        },
        min_score=30.0,
    )
    logger.info("Ranked liquid tickers: %s", liquid_tickers)

    # Pair Analysis
    analyzer = PairTradingAnalyzer(
        base_dir=base_dir,
        run_date=run_date,
        lookback=lookback,
        tickers=liquid_tickers,
        min_mean_reversion=min_mean_reversion,
    )
    result = analyzer.analyze(
        volatility_threshold=volatility_threshold, n_clusters=n_clusters_pairs
    )

    for cluster_name, cluster_info in result.items():
        logger.info("%s:", cluster_name)
        logger.info("Tickers: %s", cluster_info["tickers"])
        logger.info("Pairs: %s", cluster_info["pairs"])
        logger.info("-" * 40)

    filename = (
        f"pair_trading_result_{run_date}_lookback-{lookback}"
        f"_vol-{volume_threshold}"
        f"_minmr-{min_mean_reversion}"
        f"_volth-{volatility_threshold}"
        f"_clusters-{n_clusters_pairs}.json"
    )
    output_path = Path(output_dir) / filename
    with open(output_path, "w") as f:
        json.dump(result, f, indent=4)

    logger.info("Saved result to %s", output_path)
    return str(output_path.resolve())
