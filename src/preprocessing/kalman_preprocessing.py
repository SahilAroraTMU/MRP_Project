
from pykalman import KalmanFilter

def apply_kalman_filter(df):

    close_prices = df['Close'].values

    kf = KalmanFilter(
        initial_state_mean=close_prices[0],
        n_dim_obs=1
    )

    state_means, _ = kf.filter(close_prices)

    df['Kalman_Close'] = state_means.flatten()

    df['Kalman_Return'] = (
        df['Kalman_Close'].pct_change()
    )

    df['Kalman_Volatility'] = (
        df['Kalman_Return'].rolling(7).std()
    )

    return df
