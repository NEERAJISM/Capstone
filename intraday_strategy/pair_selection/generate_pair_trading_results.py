import json
from pathlib import Path
from typing import List
from .liquidity import LiquidityPredictor
from .mean_reversion import MeanReversionAnalyzer
from common import get_logger
from config import config
from common import Plots

logger = get_logger(__name__)



def get_pair_json_path_cached():
    output_dir = Path(config.data.output_dir) / "pair_selection" / config.data.run_date

    output_dir.mkdir(parents=True, exist_ok=True)


    prefix = (
        f"pair_trading_result_{config.data.run_date}"
        f"_lookback-{config.pair_selection.lookback_days}"
        f"_vol-{config.pair_selection.volume_threshold}"
        f"_minmr-{config.pair_selection.min_mean_reversion}"
        f"_volth-{config.pair_selection.volatility_threshold}"
    )

    matches = list(output_dir.glob(f"{prefix}_clusters-*.json"))
    if matches:
        logger.info(f"Found existing pair JSON: {matches[0]}")
        return matches[0]

    # Otherwise, generate it
    logger.info("No existing pair JSON found. Generating new one...")
    return generate_pair_json(
        base_dir=config.data.data_dir,
        run_date=config.data.run_date,
        lookback=config.pair_selection.lookback_days,
        tickers_universe=config.data.tickers_universe,
        volume_threshold=config.pair_selection.volume_threshold,
        min_mean_reversion=config.pair_selection.min_mean_reversion,
        volatility_threshold=config.pair_selection.volatility_threshold,
        n_clusters_pairs=config.pair_selection.n_clusters_pairs,
        output_dir=config.data.output_dir / "pair_selection",
    )

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
    analyzer = MeanReversionAnalyzer(
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
    output_dir = Path(config.data.output_dir) / "pair_selection" / config.data.run_date

    with open(output_dir / filename, "w") as f:
        json.dump(result, f, indent=4)

    logger.info("Saved result to %s", output_dir / filename)
    Plots.plot_clusters(output_dir / filename)
    return str((output_dir / filename).resolve())
