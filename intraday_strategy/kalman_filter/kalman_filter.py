from pykalman import KalmanFilter
import numpy as np

# Kalman params
DELTA = 5e-4
OBS_COV = 1.0

class Kalman:

    @staticmethod
    def kalman_hedge(x, y, delta=DELTA, obs_cov=OBS_COV):
        """Time-varying hedge ratio and intercept using a 2D state Kalman filter."""
        n = len(x)
        obs_mats = np.stack([x, np.ones(n)], axis=1)[:, np.newaxis, :]  # (n,1,2)
        trans_cov = delta / (1.0 - delta) * np.eye(2)

        kf = KalmanFilter(
            transition_matrices=np.eye(2),
            transition_covariance=trans_cov,
            observation_covariance=obs_cov,
            initial_state_mean=np.zeros(2),
            initial_state_covariance=np.ones((2, 2)),
        )
        kf.observation_matrices = obs_mats
        state_means, _ = kf.filter(y)
        beta = state_means[:, 0]
        intercept = state_means[:, 1]
        return beta, intercept
