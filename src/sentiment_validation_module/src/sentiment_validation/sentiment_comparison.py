
import os
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = 'outputs/sentiment_validation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sentiment_distribution_comparison(df):
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df[
            [
                'VADER_Sentiment',
                'TextBlob_Sentiment',
                'FinBERT_Sentiment'
            ]
        ]
    )

    plt.title('Sentiment Score Comparison')

    plt.savefig(
        f'{OUTPUT_DIR}/sentiment_boxplot.png'
    )

    plt.close()

def sentiment_histograms(df):
    plt.figure(figsize=(12, 6))

    plt.hist(
        df['VADER_Sentiment'],
        alpha=0.5,
        bins=30,
        label='VADER'
    )

    plt.hist(
        df['TextBlob_Sentiment'],
        alpha=0.5,
        bins=30,
        label='TextBlob'
    )

    plt.hist(
        df['FinBERT_Sentiment'],
        alpha=0.5,
        bins=30,
        label='FinBERT'
    )

    plt.legend()

    plt.title(
        'Sentiment Distribution Comparison'
    )

    plt.savefig(
        f'{OUTPUT_DIR}/sentiment_histograms.png'
    )

    plt.close()

def sentiment_correlation(df):
    plt.figure(figsize=(8, 6))

    corr = df[
        [
            'VADER_Sentiment',
            'TextBlob_Sentiment',
            'FinBERT_Sentiment'
        ]
    ].corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm'
    )

    plt.title(
        'Sentiment Correlation Heatmap'
    )

    plt.savefig(
        f'{OUTPUT_DIR}/sentiment_correlation_heatmap.png'
    )

    plt.close()

def sentiment_scatterplot(df):
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x='VADER_Sentiment',
        y='FinBERT_Sentiment',
        data=df
    )

    plt.title('VADER vs FinBERT')

    plt.savefig(
        f'{OUTPUT_DIR}/sentiment_scatterplot.png'
    )

    plt.close()
