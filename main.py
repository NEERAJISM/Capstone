import datetime

from intraday_strategy import generate_pair_json
import json
import pandas as pd
from backtest import run_pair
import os
from common import StockDataLoader
from config import config

def main():
    # Generate pair selection results
    pair_json_path = generate_pair_json(
        base_dir=config.data.data_dir,
        run_date=config.data.run_date,
        lookback=config.data.lookback_days,
        tickers_universe=config.data.tickers_universe,
        volume_threshold=config.pair_selection.volume_threshold,
        min_mean_reversion=config.pair_selection.min_mean_reversion,
        volatility_threshold=config.pair_selection.volatility_threshold,
        n_clusters_pairs=config.pair_selection.n_clusters_pairs,
        output_dir=config.data.output_dir / "pair_selection",
    )

    with open(pair_json_path, "r") as f:
        clusters = json.load(f)

    all_results = []
    for cid, cdata in clusters.items():
        pairs = cdata.get("pairs", [])
        print(pairs)
        if not pairs:
            continue
        print(f"\n=== Running {cid} ===")
        for a_pair in pairs:
            stock_a, stock_b = a_pair["tickers"]
            end_date = datetime.datetime.strptime(config.data.run_date, "%Y-%m-%d")
            start_date = end_date - datetime.timedelta(days=config.data.lookback_days) 
            
            loader = StockDataLoader(
                base_dir=config.data.data_dir,
                start=start_date.strftime("%Y-%m-%d %H:%M:%S"),
                end=end_date.strftime("%Y-%m-%d %H:%M:%S"),
                tickers=[stock_a, stock_b],
                impute=True
            )
            
            result = run_pair(stock_a, stock_b, loader, config)
            if result:
                all_results.append(result)

    # Save leaderboard
    if all_results:
        leaderboard = pd.DataFrame(all_results)
        leaderboard.sort_values("net_pnl", ascending=False, inplace=True)
        
        # Save results to output directory
        results_dir = config.data.output_dir / "backtest_results"
        os.makedirs(results_dir, exist_ok=True)
        
        leaderboard_path = results_dir / "pairs_leaderboard.csv"
        leaderboard.to_csv(leaderboard_path, index=False)
        
        print(f"\n=== Leaderboard saved: {leaderboard_path} ===")
        print(leaderboard.head(10))
        
        # Save detailed results
        summary = {
            'run_timestamp': datetime.datetime.now().isoformat(),
            'config': {
                'run_date': config.data.run_date,
                'lookback_days': config.data.lookback_days,
                'n_pairs_tested': len(all_results),
                'n_clusters': len(clusters)
            },
            'metrics': {
                'total_trades': leaderboard['trades'].sum(),
                'total_pnl': leaderboard['net_pnl'].sum(),
                'avg_win_rate': leaderboard['win_rate'].mean(),
                'avg_daily_sharpe': leaderboard['daily_sharpe'].mean()
            }
        }
        
        with open(results_dir / 'backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
    else:
        print("No valid results generated from any pairs.")


if __name__ == "__main__":
    main()
