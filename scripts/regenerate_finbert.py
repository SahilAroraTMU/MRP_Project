from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import create_features
from src.sentiment_analysis import finbert_sentiment


def load_reddit_text(path, text_column):
    df = pd.read_csv(
        PROJECT_ROOT / path,
        usecols=['created_utc', text_column],
    )
    df = df.dropna()
    df['Date'] = pd.to_datetime(
        df['created_utc'],
        unit='s'
    ).dt.date
    df['text'] = df[text_column].astype(str)
    return df[['Date', 'text']]


def build_daily_text():
    comments = load_reddit_text(
        'data/raw/reddit_comments.csv',
        'body'
    )
    posts = load_reddit_text(
        'data/raw/reddit_posts.csv',
        'title'
    )
    reddit = pd.concat(
        [comments, posts],
        ignore_index=True
    )
    return (
        reddit
        .groupby('Date')['text']
        .apply(lambda values: ' '.join(values))
        .reset_index()
    )


def main():
    merged_path = PROJECT_ROOT / 'data/processed/merged_dataset.csv'
    final_path = PROJECT_ROOT / 'outputs/results/final_merged_dataset.csv'

    merged = pd.read_csv(merged_path)
    merged['Date'] = pd.to_datetime(merged['Date']).dt.date

    daily_text = build_daily_text()
    target_dates = set(merged['Date'])
    daily_text = daily_text[daily_text['Date'].isin(target_dates)].copy()

    daily_text['FinBERT_Sentiment'] = daily_text['text'].apply(
        finbert_sentiment
    )

    merged = merged.drop(
        columns=['FinBERT_Sentiment'],
        errors='ignore'
    ).merge(
        daily_text[['Date', 'FinBERT_Sentiment']],
        on='Date',
        how='left'
    )
    merged['FinBERT_Sentiment'] = (
        merged['FinBERT_Sentiment']
        .fillna(0)
    )

    merged.to_csv(
        merged_path,
        index=False
    )

    final = create_features(merged)
    final.to_csv(
        final_path,
        index=False
    )

    print(merged['FinBERT_Sentiment'].describe())
    print(
        'non_zero',
        int(merged['FinBERT_Sentiment'].ne(0).sum()),
        'of',
        len(merged)
    )


if __name__ == '__main__':
    main()
