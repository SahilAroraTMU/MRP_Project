import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------
# Close Price Plot
# ---------------------------------------------------

def plot_close_price(df):

    plt.figure(figsize=(10,5))

    plt.plot(
        df['Date'],
        df['Close']
    )

    plt.title('TSLA Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')

    plt.savefig(
        'outputs/figures/close_price.png'
    )

    plt.close()


# ---------------------------------------------------
# Volume Plot
# ---------------------------------------------------

def plot_volume(df):

    plt.figure(figsize=(10,5))

    plt.plot(
        df['Date'],
        df['Volume']
    )

    plt.title('Trading Volume')
    plt.xlabel('Date')
    plt.ylabel('Volume')

    plt.savefig(
        'outputs/figures/volume.png'
    )

    plt.close()


# ---------------------------------------------------
# Sentiment Distribution
# ---------------------------------------------------

def plot_sentiment_distribution(df):

    plt.figure(figsize=(8,5))

    plt.hist(
        df['Avg_Sentiment'],
        bins=25
    )

    plt.title('Sentiment Distribution')

    plt.savefig(
        'outputs/figures/sentiment_distribution.png'
    )

    plt.close()


# ---------------------------------------------------
# Outlier Detection
# ---------------------------------------------------

def boxplot_outlier_detection(df):

    plt.figure(figsize=(8,5))

    sns.boxplot(
        x=df['Volume']
    )

    plt.title(
        'Trading Volume Outlier Detection'
    )

    plt.savefig(
        'outputs/figures/outlier_detection.png'
    )

    plt.close()


# ---------------------------------------------------
# Correlation Heatmap
# ---------------------------------------------------

def correlation_heatmap(df):

    cols = [
        'Close',
        'Volume',
        'Return',
        'Volatility_7D',
        'Illiquidity',
        'Turnover_Ratio',
        'Avg_Sentiment',
        'Comment_Count'
    ]

    corr = df[cols].corr()

    plt.figure(figsize=(10,6))

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm'
    )

    plt.title('Correlation Heatmap')

    plt.savefig(
        'outputs/figures/correlation_heatmap.png'
    )

    corr.to_csv(
        'outputs/results/correlation_matrix.csv'
    )

    plt.close()
