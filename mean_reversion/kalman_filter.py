import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

DATA_FOLDER = "../2021/Cash Data April 2021/"
OUTPUT_FOLDER = "output_plots"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def load_stock(filepath):
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['<date>'] + ' ' + df['<time>'], format='%m/%d/%Y %H:%M:%S')
    df = df[['datetime', '<close>']].copy()
    df.rename(columns={'<close>': 'Close'}, inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.set_index('datetime', inplace=True)
    return df

def run_kalman_filter(stock1, stock2):
    try:
        df1 = load_stock(os.path.join(DATA_FOLDER, f"{stock1}.csv"))
        df2 = load_stock(os.path.join(DATA_FOLDER, f"{stock2}.csv"))
    except Exception as e:
        print(f"Error loading {stock1} or {stock2}: {e}")
        return

    df = pd.merge(df1, df2, left_index=True, right_index=True, suffixes=(f'_{stock1}', f'_{stock2}'))
    df.dropna(inplace=True)
    if df.empty:
        print(f"No overlapping data between {stock1} and {stock2}, skipping.")
        return

    Y = df[f'Close_{stock1}'].values
    X = df[f'Close_{stock2}'].values

    obs_mat = np.stack([X, np.ones(len(X))], axis=1)[:, np.newaxis, :]

    delta = 1e-4
    trans_cov = delta / (1 - delta) * np.eye(2)

    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_covariance=trans_cov,
        observation_covariance=1.0,
    )

    kf.observation_matrices = obs_mat
    state_means, _ = kf.filter(Y)

    hedge_ratio = state_means[:, 0]
    intercept = state_means[:, 1]
    spread = Y - (hedge_ratio * X + intercept)

    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std

    # Plotting
    plt.figure(figsize=(14, 8))
    plt.suptitle(f'Kalman Filter Mean Reversion\n{stock1} vs {stock2}')

    plt.subplot(3, 1, 1)
    plt.plot(df.index, hedge_ratio, label='Hedge Ratio')
    plt.ylabel('Hedge Ratio')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(df.index, spread, color='orange', label='Spread')
    plt.axhline(spread_mean, color='black', linestyle='--', label='Mean')
    plt.axhline(spread_mean + spread_std, color='green', linestyle='--', label='+1 STD')
    plt.axhline(spread_mean - spread_std, color='red', linestyle='--', label='-1 STD')
    plt.ylabel('Spread')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df.index, z_score, label='Spread Z-Score')
    plt.axhline(0, color='black', linestyle='--')
    plt.axhline(1, color='green', linestyle='--', label='Entry Threshold (+1)')
    plt.axhline(-1, color='red', linestyle='--', label='Entry Threshold (-1)')
    plt.ylabel('Z-Score')
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    filename = f"{stock1}_{stock2}_kalman_plot.png"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot: {filepath}")
