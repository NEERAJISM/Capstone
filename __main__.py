import os
from pathlib import Path
from capstone.intraday_strategy import generate_pair_json
from capstone.common import StockDataLoader

os.chdir("capstone")
loader = StockDataLoader(
    base_dir="data",
    start="2021-08-01 09:15:00",
    end = "2021-10-31 15:30:00",
    tickers = [],
    impute= True
 
)
print(loader.get_data_for_tickers())
   
path = generate_pair_json(base_dir =Path("data"), run_date="2022-01-01", lookback=30, tickers_universe=[], volume_threshold=1000, min_mean_reversion=0.01, volatility_threshold=0.0005, n_clusters_pairs=5, output_dir="./results/pair_selection")     
print(path)