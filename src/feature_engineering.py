import numpy as np

def create_features(df):

    # Daily Returns
    df['Return'] = (
        df['Close'].pct_change()
    )

    # Rolling Volatility
    df['Volatility_7D'] = (
        df['Return']
        .rolling(window=7)
        .std()
    )

    # Amihud Illiquidity Ratio
    df['Illiquidity'] = (
        abs(df['Return']) / df['Volume']
    )

    # Turnover Ratio
    df['Turnover_Ratio'] = (
        df['Volume'] /
        df['Volume'].rolling(30).mean()
    )

    # Lagged Sentiment Features
    df['Sentiment_Lag_1'] = (
        df['Avg_Sentiment'].shift(1)
    )

    df['Sentiment_Lag_2'] = (
        df['Avg_Sentiment'].shift(2)
    )

    # Market Regime Classification
    median_volatility = (
        df['Volatility_7D'].median()
    )

    df['Market_Regime'] = np.where(
        df['Volatility_7D'] > median_volatility,
        'High Volatility',
        'Low Volatility'
    )

    # Remove Missing Values
    df = df.dropna()

    return df
