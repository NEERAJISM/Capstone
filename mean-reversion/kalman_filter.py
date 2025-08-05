import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter


class MeanReversionKalman:
    def __init__(self):
        # Initialize Kalman filter
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # State transition matrix [mean, velocity]
        self.kf.F = np.array([[1., 1.],
                              [0., 1.]])

        # Observation matrix - we observe the mean directly
        self.kf.H = np.array([[1., 0.]])

        # Process noise - how much mean can change
        self.kf.Q = np.array([[0.001, 0.],
                              [0., 0.0001]])

        # Observation noise - measurement error
        self.kf.R = np.array([[0.01]])

        # Initial state [mean, velocity]
        self.kf.x = np.array([[0.], [0.]])

        # Initial uncertainty
        self.kf.P = np.array([[1., 0.],
                              [0., 0.1]])

    def update_mean(self, spread_value):
        """Update mean estimate with new spread observation"""
        self.kf.predict()
        self.kf.update(spread_value)

        return self.kf.x[0, 0]  # Return current mean estimate

    def get_confidence(self):
        """Get confidence in current mean estimate"""
        return np.sqrt(self.kf.P[0, 0])

    def generate_signal(self, current_spread):
        """Generate trading signal"""
        estimated_mean = self.kf.x[0, 0]
        confidence = self.get_confidence()

        # Z-score calculation
        z_score = (current_spread - estimated_mean) / confidence

        if z_score > 2.0:
            return "SELL_SPREAD"  # Short A, Long B
        elif z_score < -2.0:
            return "BUY_SPREAD"  # Long A, Short B
        else:
            return "HOLD"


# Usage Example
def backtest_example():
    # Sample spread data (replace with real NSE pair data)
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')

    # Simulate mean-reverting spread with changing mean
    true_mean = np.cumsum(np.random.normal(0, 0.01, 1000))  # Evolving mean
    spread_data = true_mean + np.random.normal(0, 0.1, 1000)  # Add noise

    # Initialize Kalman filter
    kalman_mr = MeanReversionKalman()

    # Track results
    signals = []
    kalman_means = []

    for i, spread in enumerate(spread_data):
        # Update mean estimate
        estimated_mean = kalman_mr.update_mean(spread)
        kalman_means.append(estimated_mean)

        # Generate signal
        signal = kalman_mr.generate_signal(spread)
        signals.append(signal)

        if i % 100 == 0:
            print(f"Step {i}: Spread={spread:.3f}, "
                  f"Estimated Mean={estimated_mean:.3f}, "
                  f"Signal={signal}")

    # Results analysis
    results_df = pd.DataFrame({
        'timestamp': dates,
        'spread': spread_data,
        'true_mean': true_mean,
        'kalman_mean': kalman_means,
        'signal': signals
    })

    return results_df


# Run backtest
if __name__ == "__main__":
    results = backtest_example()
    print("\nBacktest completed!")
    print(f"Total signals generated: {len(results)}")
    print(f"Buy signals: {sum(results['signal'] == 'BUY_SPREAD')}")
    print(f"Sell signals: {sum(results['signal'] == 'SELL_SPREAD')}")