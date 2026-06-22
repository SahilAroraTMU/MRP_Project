from pathlib import Path
import os
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

cache_dir = PROJECT_ROOT / '.cache'
cache_dir.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(cache_dir / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(cache_dir))
os.environ.setdefault('MPLBACKEND', 'Agg')

from src.fetch_data import fetch_yahoo_finance_data
from src.preprocess_data import combine_reddit_data, merge_datasets, preprocess_financial_data
from src.sentiment_analysis import aggregate_daily_text, apply_daily_finbert, apply_fast_sentiment, vader_sentiment
from src.preprocessing.engagement_preprocessing import calculate_engagement_score
from src.preprocessing.final_merge_preprocessing import merge_reddit_financial
from src.preprocessing.kalman_preprocessing import apply_kalman_filter
from src.preprocessing.relationship_preprocessing import (
    build_post_comment_relationship,
    calculate_comment_counts,
    detect_orphan_comments,
)
from src.preprocessing.reddit_preprocessing import (
    load_reddit_data,
    preprocess_comment_reposts,
    preprocess_post_reposts,
    preprocess_timestamps,
)
from src.preprocessing.topic_preprocessing import detect_topic, tesla_relevance
from src.feature_engineering import create_features
from src.train_models import train_linear_regression, train_random_forest, train_xgboost
from src.evaluate_models import evaluate_model


def load_external_daily_finbert():
    candidates = [
        PROJECT_ROOT / 'outputs' / 'eda_storytelling' / 'daily_finbert_sentiment.csv',
        PROJECT_ROOT / 'outputs' / 'eda_storytelling' / 'daily_aggregated_finbert.csv',
    ]
    finbert_path = next((path for path in candidates if path.exists()), candidates[0])
    if not finbert_path.exists():
        return None

    df = pd.read_csv(finbert_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

    if {'Avg_Comment_Sentiment', 'Comment_Count', 'Avg_Post_Sentiment', 'Post_Count'}.issubset(df.columns):
        df['Comment_Count'] = pd.to_numeric(df['Comment_Count'], errors='coerce').fillna(0)
        df['Post_Count'] = pd.to_numeric(df['Post_Count'], errors='coerce').fillna(0)
        df['Avg_Comment_Sentiment'] = pd.to_numeric(df['Avg_Comment_Sentiment'], errors='coerce').fillna(0)
        df['Avg_Post_Sentiment'] = pd.to_numeric(df['Avg_Post_Sentiment'], errors='coerce').fillna(0)
        grouped = (
            df.assign(
                total_score=(
                    df['Avg_Comment_Sentiment'] * df['Comment_Count'] +
                    df['Avg_Post_Sentiment'] * df['Post_Count']
                ),
                total_count=df['Comment_Count'] + df['Post_Count'],
            )
            .groupby('Date', as_index=False)[['total_score', 'total_count']]
            .sum()
        )
        grouped['FinBERT_Sentiment'] = grouped['total_score'] / grouped['total_count'].replace(0, pd.NA)
        return grouped[['Date', 'FinBERT_Sentiment']]

    if 'FinBERT_Sentiment' not in df.columns and 'Avg_FinBERT_Sentiment' in df.columns:
        df = df.rename(columns={'Avg_FinBERT_Sentiment': 'FinBERT_Sentiment'})

    return (
        df[['Date', 'FinBERT_Sentiment']]
        .dropna(subset=['Date'])
        .groupby('Date', as_index=False)['FinBERT_Sentiment']
        .mean()
    )


def build_reddit_pipeline():
    print('Loading raw Reddit data...')
    posts_df, comments_df = load_reddit_data(
        PROJECT_ROOT / 'data' / 'raw' / 'reddit_posts.csv',
        PROJECT_ROOT / 'data' / 'raw' / 'reddit_comments.csv',
    )

    posts_df = preprocess_timestamps(posts_df)
    comments_df = preprocess_timestamps(comments_df)
    posts_df = preprocess_post_reposts(posts_df)
    comments_df = preprocess_comment_reposts(comments_df)

    merged_relation = build_post_comment_relationship(posts_df, comments_df)
    posts_df = calculate_comment_counts(posts_df, merged_relation)
    merged_relation = detect_orphan_comments(merged_relation)

    posts_df['tesla_relevance'] = (
        (
            posts_df['title'].fillna('').astype(str) + ' ' +
            posts_df.get('selftext', pd.Series('', index=posts_df.index)).fillna('').astype(str)
        ).apply(tesla_relevance)
    )
    posts_df['context_topic'] = posts_df['title'].apply(detect_topic)

    posts_df['VADER_Sentiment'] = (
        (
            posts_df['title'].fillna('').astype(str) + ' ' +
            posts_df.get('selftext', pd.Series('', index=posts_df.index)).fillna('').astype(str)
        ).apply(vader_sentiment)
    )

    posts_df = calculate_engagement_score(posts_df)

    reddit_df = combine_reddit_data(comments_df.copy(), posts_df.copy())
    reddit_df = apply_fast_sentiment(reddit_df)

    daily_finbert = aggregate_daily_text(reddit_df)
    external_daily_finbert = load_external_daily_finbert()
    if external_daily_finbert is not None:
        daily_finbert = daily_finbert.merge(external_daily_finbert, on='Date', how='left')
    else:
        daily_finbert = apply_daily_finbert(daily_finbert)

    reddit_df = reddit_df.merge(daily_finbert[['Date', 'FinBERT_Sentiment']], on='Date', how='left')

    reddit_daily = (
        reddit_df
        .groupby('Date')
        .agg({
            'VADER_Sentiment': 'mean',
            'TextBlob_Sentiment': 'mean',
            'score': 'mean',
            'text': 'count',
        })
        .reset_index()
        .rename(columns={
            'VADER_Sentiment': 'Avg_Sentiment',
            'score': 'Avg_Reddit_Score',
            'text': 'Comment_Count',
        })
    )

    reddit_daily = reddit_daily.merge(daily_finbert[['Date', 'FinBERT_Sentiment']], on='Date', how='left')

    advanced_daily = (
        posts_df
        .groupby('Date')
        .agg({
            'engagement_score': 'mean',
            'tesla_relevance': 'mean',
            'comment_count': 'mean',
            'repost_count': 'mean',
            'unique_author_count': 'mean',
            'discussion_intensity': 'mean',
            'information_diffusion_score': 'mean',
            'sentiment_magnitude': 'mean',
            'VADER_Sentiment': 'mean',
        })
        .reset_index()
        .rename(columns={'VADER_Sentiment': 'Post_VADER_Sentiment'})
    )

    reddit_daily = reddit_daily.merge(advanced_daily, on='Date', how='left')

    return posts_df, comments_df, merged_relation, reddit_daily, daily_finbert


def train_and_save_models(merged_df):
    merged_df = create_features(merged_df)
    merged_df.to_csv(PROJECT_ROOT / 'outputs' / 'results' / 'final_merged_dataset.csv', index=False)
    merged_df.to_csv(PROJECT_ROOT / 'data' / 'processed' / 'preprocessed_data_3.csv', index=False)

    feature_columns = [
        'Close',
        'Volume',
        'Return',
        'Volatility_7D',
        'Turnover_Ratio',
        'FinBERT_Sentiment',
        'Sentiment_Lag_1',
        'Sentiment_Lag_2',
        'information_diffusion_score',
        'tesla_relevance',
    ]
    X = merged_df[feature_columns]
    y = merged_df['Illiquidity']

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
        random_state=42,
    )

    target_scale = 1e10
    y_train_scaled = y_train * target_scale

    lr_model = train_linear_regression(X_train, y_train_scaled)
    rf_model = train_random_forest(X_train, y_train_scaled)
    xgb_model = train_xgboost(X_train, y_train_scaled)

    lr_results = evaluate_model(lr_model, X_test, y_test, prediction_scale=1 / target_scale)
    rf_results = evaluate_model(rf_model, X_test, y_test, prediction_scale=1 / target_scale)
    xgb_results = evaluate_model(xgb_model, X_test, y_test, prediction_scale=1 / target_scale)

    print('\nLinear Regression')
    print(lr_results)
    print('\nRandom Forest')
    print(rf_results)
    print('\nXGBoost')
    print(xgb_results)

    results_df = pd.DataFrame([
        {'Model': 'Linear Regression', **lr_results},
        {'Model': 'Random Forest', **rf_results},
        {'Model': 'XGBoost', **xgb_results},
    ])
    results_df.to_csv(PROJECT_ROOT / 'outputs' / 'results' / 'model_results.csv', index=False)

    predictions_df = pd.DataFrame({
        'Actual': y_test,
        'Linear_Regression_Predictions': lr_model.predict(X_test) / target_scale,
        'Random_Forest_Predictions': rf_model.predict(X_test) / target_scale,
        'XGBoost_Predictions': xgb_model.predict(X_test) / target_scale,
    })
    predictions_df.to_csv(PROJECT_ROOT / 'outputs' / 'results' / 'model_predictions.csv', index=False)

    print('Saved model_results.csv and model_predictions.csv')


def main():
    if os.environ.get('MRP_SKIP_FULL_REBUILD') == '1':
        merged_path = PROJECT_ROOT / 'data' / 'processed' / 'preprocessed_data_3.csv'
        if not merged_path.exists():
            raise FileNotFoundError(
                'MRP_SKIP_FULL_REBUILD=1 was set, but data/processed/preprocessed_data_3.csv does not exist.'
            )

        merged_df = pd.read_csv(merged_path)
        train_and_save_models(merged_df)
        return

    fetch_yahoo_finance_data()

    posts_df, comments_df, merged_relation, reddit_daily, daily_finbert = build_reddit_pipeline()

    with pd.ExcelWriter(PROJECT_ROOT / 'data' / 'processed' / 'preprocessed_data_1.xlsx') as writer:
        posts_df.to_excel(writer, sheet_name='cleaned_posts', index=False)
        comments_df.to_excel(writer, sheet_name='cleaned_comments', index=False)
        merged_relation.to_excel(writer, sheet_name='post_comment_links', index=False)

    financial_df = preprocess_financial_data()
    financial_df = apply_kalman_filter(financial_df)
    financial_df.to_csv(PROJECT_ROOT / 'data' / 'processed' / 'preprocessed_data_2.csv', index=False)

    merged_df = merge_reddit_financial(reddit_daily, financial_df)
    merged_df.to_csv(PROJECT_ROOT / 'data' / 'processed' / 'preprocessed_data_3.csv', index=False)

    train_and_save_models(merged_df)


if __name__ == '__main__':
    main()
