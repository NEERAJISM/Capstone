from typing import List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import os
from datetime import time
import pandas as pd


@dataclass
class DataConfig:
    """Configuration for data-related parameters."""

    data_dir: Path
    output_dir: Path
    run_date: str
    tickers_universe: List[str]

    def __post_init__(self):
        """Ensure paths exist and are properly formatted."""
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_dir / "pair_selection", exist_ok=True)
        os.makedirs(self.output_dir / "backtest", exist_ok=True)
        os.makedirs(self.output_dir / "regime_detection", exist_ok=True)


@dataclass
class PairSelectionConfig:
    """Configuration for pair selection parameters."""

    volume_threshold: int
    min_mean_reversion: float
    volatility_threshold: float
    n_clusters_pairs: int
    lookback_days: int


@dataclass
class RegimeDetectionConfig:
    """Configuration for pair selection parameters."""
    lookback_minutes: int
    lookback_months:int


@dataclass
class StrategyConfig:
    """Configuration for trading strategy parameters."""

    # Capital and risk
    capital: float = 100000.0
    per_trade_risk: float = 0.02

    # Transaction costs
    stt_pct: float = 0.00025
    slippage_pct: float = 0.001
    brokerage_pct: float = 0.0003

    # Mean reversion parameters
    rolling_window: int = 60
    z_entry: float = 2.5
    z_exit: float = 0.2
    min_hold_bars: int = 5
    cooldown_bars: int = 15
    vol_filter: float = 0.0005

    # Trading hours
    start_time: time = field(default_factory=lambda: pd.to_datetime("10:15:00").time())
    end_time: time = field(default_factory=lambda: pd.to_datetime("14:30:00").time())


@dataclass
class BacktestConfig:
    """Main configuration class that combines all configs."""

    data: DataConfig
    pair_selection: PairSelectionConfig
    regime_detection: RegimeDetectionConfig
    strategy: StrategyConfig

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a dictionary."""
        return {
            "data": self.data.__dict__,
            "pair_selection": self.pair_selection.__dict__,
            "regime_detection": self.regime_detection.__dict__,
            "strategy": self.strategy.__dict__,
        }


# ================== CONFIG ================== #
# Initialize individual configs
data_config = DataConfig(
    data_dir="data",
    output_dir="results",
    run_date="2022-01-31",
    tickers_universe=[],
)

pair_selection_config = PairSelectionConfig(
    volume_threshold=1000,
    min_mean_reversion=0.01,
    volatility_threshold=0.0005,
    n_clusters_pairs=5,
    lookback_days=30,
)

regime_detection_config = RegimeDetectionConfig(
    lookback_months=3,
    lookback_minutes=60,

)

strategy_config = StrategyConfig(
    capital=100000.0,
    per_trade_risk=0.02,
    stt_pct=0.00025,
    slippage_pct=0.001,
    brokerage_pct=0.0003,
    rolling_window=60,
    z_entry=2.5,
    z_exit=0.2,
    min_hold_bars=5,
    cooldown_bars=15,
    vol_filter=0.0005,
)

# Main config that combines all configs
config = BacktestConfig(
    data=data_config, pair_selection=pair_selection_config, regime_detection=regime_detection_config,strategy=strategy_config
)
# =========================================== #
