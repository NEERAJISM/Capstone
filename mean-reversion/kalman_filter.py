import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# ----------- Load & preprocess data ------------
def load_stock(filepath):
    df = pd.read_csv(filepath)
    # Parse datetime (US format MM/DD/YYYY)
    df['datetime'] = pd.to_datetime(df['<date>'] + ' ' + df['<time>'], format='%m/%d/%Y %H:%M:%S')
    df = df[['datetime', '<close>']].copy()
    df = df.rename(columns={'<close>': 'Close'})
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    return df.set_index('datetime')

# ----------- Load files ------------
icici = load_stock("../2021/Cash Data April 2021/ICICIBANK.csv")
hdfc = load_stock("../2021/Cash Data April 2021/HDFCBANK.csv")

# ----------- Merge and align on datetime ------------
df = pd.merge(icici, hdfc, left_index=True, right_index=True, suffixes=('_ICICI', '_HDFC'))
df.dropna(inplace=True)

# ----------- Prepare observation vectors ------------
Y = df['Close_ICICI'].values  # dependent (ICICI)
X = df['Close_HDFC'].values   # independent (HDFC)

# Prepare observation matrix with correct shape
obs_mat = np.stack([X, np.ones(len(X))], axis=1)  # (n_timesteps, 2)
obs_mat = obs_mat[:, np.newaxis, :]               # (n_timesteps, 1, 2)

# Kalman Filter setup
delta = 1e-4  # process noise
trans_cov = delta / (1 - delta) * np.eye(2)  # transition covariance

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

# Compute spread
spread = Y - (hedge_ratio * X + intercept)

# ----------- Plot results ------------
plt.figure(figsize=(14, 6))

# Hedge Ratio
plt.subplot(2, 1, 1)
plt.plot(df.index, hedge_ratio, label='Hedge Ratio (Kalman)')
plt.title('Time-Varying Hedge Ratio (ICICI vs HDFC Bank)')
plt.legend()

# Spread
plt.subplot(2, 1, 2)
plt.plot(df.index, spread, label='Spread = ICICI - β × HDFC', color='orange')
plt.axhline(spread.mean(), color='black', linestyle='--', label='Mean')
plt.axhline(spread.mean() + spread.std(), color='green', linestyle='--', label='+1 STD')
plt.axhline(spread.mean() - spread.std(), color='red', linestyle='--', label='-1 STD')
plt.title('Kalman Filter Estimated Spread')
plt.legend()

plt.tight_layout()
plt.show()
