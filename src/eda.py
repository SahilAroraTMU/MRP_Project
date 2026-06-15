import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

EDA_OUTPUT_DIR = Path('outputs/eda')
FIGURE_OUTPUT_DIR = Path('outputs/figures')

def _save_current_plot(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


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

    _save_current_plot(FIGURE_OUTPUT_DIR / 'close_price.png')


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

    _save_current_plot(FIGURE_OUTPUT_DIR / 'volume.png')


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

    _save_current_plot(FIGURE_OUTPUT_DIR / 'sentiment_distribution.png')


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

    _save_current_plot(FIGURE_OUTPUT_DIR / 'outlier_detection.png')


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

    _save_current_plot(FIGURE_OUTPUT_DIR / 'correlation_heatmap.png')

    corr.to_csv(
        'outputs/results/correlation_matrix.csv'
    )

def duplicate_repost_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['repost_count'], bins=30, kde=False)
    plt.title('Duplicate/Repost Distribution')
    plt.xlabel('Repost Count')
    plt.ylabel('Frequency')
    _save_current_plot(EDA_OUTPUT_DIR / 'duplicate_repost_distribution.png')

def unique_author_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['unique_author_count'], bins=30, kde=False)
    plt.title('Unique Author Distribution')
    plt.xlabel('Unique Author Count')
    plt.ylabel('Frequency')
    _save_current_plot(EDA_OUTPUT_DIR / 'unique_author_distribution.png')

def information_diffusion_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['information_diffusion_score'], bins=30, kde=False)
    plt.title('Information Diffusion Distribution')
    plt.xlabel('Information Diffusion Score')
    plt.ylabel('Frequency')
    _save_current_plot(EDA_OUTPUT_DIR / 'information_diffusion_distribution.png')

def engagement_score_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df['engagement_score'], bins=30, kde=True)
    plt.title('Engagement Score Distribution')
    plt.xlabel('Engagement Score')
    plt.ylabel('Frequency')
    _save_current_plot(EDA_OUTPUT_DIR / 'engagement_score_distribution.png')

def engagement_score_vs_trading_volume(df):
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='engagement_score', y='Volume')
    plt.title('Engagement Score vs Trading Volume')
    plt.xlabel('Engagement Score')
    plt.ylabel('Trading Volume')
    _save_current_plot(EDA_OUTPUT_DIR / 'engagement_score_vs_trading_volume.png')

def engagement_score_vs_illiquidity(df):
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x='engagement_score', y='Illiquidity')
    plt.title('Engagement Score vs Illiquidity')
    plt.xlabel('Engagement Score')
    plt.ylabel('Illiquidity')
    _save_current_plot(EDA_OUTPUT_DIR / 'engagement_score_vs_illiquidity.png')
