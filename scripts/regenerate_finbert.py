from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import create_features
from src.preprocessing.engagement_preprocessing import calculate_engagement_score
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
    preprocessed_path = PROJECT_ROOT / 'data/processed/preprocessed_data_3.csv'
    reddit_workbook_path = PROJECT_ROOT / 'data/processed/preprocessed_data_1.xlsx'
    final_path = PROJECT_ROOT / 'outputs/results/final_merged_dataset.csv'

    source_path = (
        preprocessed_path
        if preprocessed_path.exists()
        else merged_path
    )

    merged = pd.read_csv(source_path)
    merged['Date'] = pd.to_datetime(merged['Date']).dt.date

    daily_text = build_daily_text()
    target_dates = set(merged['Date'])
    daily_text = daily_text[daily_text['Date'].isin(target_dates)].copy()

    daily_text['FinBERT_Sentiment'] = daily_text['text'].apply(
        finbert_sentiment
    )

    if reddit_workbook_path.exists():
        sheets = pd.read_excel(
            reddit_workbook_path,
            sheet_name=None
        )
        posts = sheets.get('cleaned_posts')

        if posts is not None:
            posts['Date'] = pd.to_datetime(posts['Date']).dt.date
            posts = posts.drop(
                columns=[
                    'FinBERT_Sentiment',
                    'sentiment_magnitude',
                    'engagement_score',
                    'engagement_factor',
                    'reddit_score'
                ],
                errors='ignore'
            ).merge(
                daily_text[['Date', 'FinBERT_Sentiment']],
                on='Date',
                how='left'
            )
            posts['FinBERT_Sentiment'] = (
                posts['FinBERT_Sentiment']
                .fillna(0)
            )
            posts = calculate_engagement_score(posts)
            sheets['cleaned_posts'] = posts

            with pd.ExcelWriter(reddit_workbook_path) as writer:
                for sheet_name, sheet_df in sheets.items():
                    sheet_df.to_excel(
                        writer,
                        sheet_name=sheet_name,
                        index=False
                    )

            post_daily = (
                posts
                .groupby('Date')
                .agg({
                    'engagement_score': 'mean',
                    'tesla_relevance': 'mean',
                    'comment_count': 'mean',
                    'repost_count': 'mean',
                    'unique_author_count': 'mean',
                    'discussion_intensity': 'mean',
                    'information_diffusion_score': 'mean',
                    'sentiment_magnitude': 'mean'
                })
                .reset_index()
            )

    merged = merged.drop(
        columns=[
            'FinBERT_Sentiment',
            'sentiment_magnitude'
        ],
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
    merged['sentiment_magnitude'] = (
        merged['FinBERT_Sentiment']
        .abs()
    )

    if 'post_daily' in locals():
        post_feature_columns = [
            'engagement_score',
            'tesla_relevance',
            'comment_count',
            'repost_count',
            'unique_author_count',
            'discussion_intensity',
            'information_diffusion_score',
            'sentiment_magnitude'
        ]
        merged = merged.drop(
            columns=post_feature_columns,
            errors='ignore'
        ).merge(
            post_daily,
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

    merged.to_csv(
        preprocessed_path,
        index=False
    )

    try:
        final = create_features(merged)
        final.to_csv(
            final_path,
            index=False
        )
    except KeyError as exc:
        print(
            "Skipped final feature-engineered model dataset refresh; "
            f"missing column: {exc}"
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
