from pathlib import Path
from pair_selection.generate_pair_trading_results import generate_pair_json


# loader = StockDataLoader(
#     base_dir="downloaded_files",
#     start="2021-08-01 09:15:00",
#     end = "2021-10-31 15:30:00",
#     tickers = [],
#     impute= True
 
# )
   
print(generate_pair_json(base_dir =Path("./downloaded_files"), run_date="2021-10-29", lookback=60, tickers_universe=[], volume_threshold=1000, min_mean_reversion=0.04, volatility_threshold=0.001, n_clusters_pairs=5, output_dir="./data"))     