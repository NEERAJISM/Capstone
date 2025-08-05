import pandas as pd

from mean_reversion.kalman_filter import run_kalman_filter

PAIRS_FILE = "pairs.csv"


def main():
    pairs_df = pd.read_csv(PAIRS_FILE)
    for _, row in pairs_df.iterrows():
        stock1 = row['stock1']
        stock2 = row['stock2']
        print(f"Processing pair: {stock1} vs {stock2}")
        run_kalman_filter(stock1, stock2)


if __name__ == "__main__":
    main()
