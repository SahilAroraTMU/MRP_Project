import pandas as pd

try:
    from pykalman import KalmanFilter
except Exception:  # pragma: no cover - optional dependency fallback
    KalmanFilter = None

def apply_kalman_filter(df):
    df = df.copy()
    close_prices = pd.to_numeric(df['Close'], errors='coerce')

    if KalmanFilter is None:
        # Fall back to an exponential moving average when pykalman is not
        # installed. This preserves a smoothed price series for the downstream
        # features and keeps the pipeline runnable in minimal environments.
        df['Kalman_Close'] = close_prices.ewm(span=10, adjust=False).mean()
    else:
        kf = KalmanFilter(
            initial_state_mean=close_prices.iloc[0],
            n_dim_obs=1
        )

        state_means, _ = kf.filter(close_prices.to_numpy())
        df['Kalman_Close'] = state_means.flatten()

    df['Kalman_Return'] = (
        df['Kalman_Close'].pct_change()
    )

    df['Kalman_Volatility'] = (
        df['Kalman_Return'].rolling(7).std()
    )

    return df
