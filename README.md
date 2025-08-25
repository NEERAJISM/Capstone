# WQU Capstone

## Regime-based Pair-Trading Model for Intraday Mean Reversion in Indian Stock Markets

### Setup

1. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

2. Activate the virtual environment:
    ```bash
    source ./venv/Scripts/activate  # On Windows using git bash
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Project Structure (with data folder and pairs json)

````
backtest/
│   backtest.py                 # Main script to run backtests on all pairs
│   output/                     # Generated strategy plots, PnL CSVs
common/
│   utils.py                    # Utility functions (load data, preprocessing)
│   plots.py                    # Visualization functions
intraday_strategy/
│   mean_reversion_intraday_strategy.py  # Strategy logic with Kalman filter
│   kalman_filter/
│       kalman_filter.py        # Kalman filter hedge ratio estimation
data/
│   2021/Cash Data April 2021/  # Intraday stock CSV files (1-min OHLCV)
│   2022/Cash Data April 2022/  # Intraday stock CSV files (1-min OHLCV)
│   pair_trading_result.json    # JSON file with clusters and stock pairs
````

5. Running the Backtest

   ````bash
   python backtest/backtest.py
   ````

Team Members:
---------------

1. Neeraj Patidar – 8patidarneeraj@gmail.com
2. Vishesh Mangla - manglavishesh64@gmail.com
3. Manish Kumar Chaudhary - mkumarchaudhary06@gmail.com

Folder:
---------------
<img width="409" height="531" alt="image" src="https://github.com/user-attachments/assets/ed795e98-d5f1-42f9-a1bc-37d32e52e54f" />
