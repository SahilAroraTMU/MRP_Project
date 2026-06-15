
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess_data import combine_reddit_data
from src.sentiment_analysis import (
    aggregate_daily_text,
    apply_daily_finbert,
    apply_fast_sentiment,
    vader_sentiment
)

from reddit_preprocessing import *
from relationship_preprocessing import *
from engagement_preprocessing import *
from topic_preprocessing import *
from financial_preprocessing import *
from kalman_preprocessing import *
from final_merge_preprocessing import *

posts_df, comments_df = load_reddit_data(
    'data/raw/reddit_posts.csv',
    'data/raw/reddit_comments.csv'
)

posts_df = preprocess_timestamps(posts_df)
comments_df = preprocess_timestamps(comments_df)

posts_df = preprocess_post_reposts(posts_df)
comments_df = preprocess_comment_reposts(comments_df)

merged_relation = build_post_comment_relationship(
    posts_df,
    comments_df
)

posts_df = calculate_comment_counts(
    posts_df,
    merged_relation
)

merged_relation = detect_orphan_comments(
    merged_relation
)

posts_df['tesla_relevance'] = (
    (
        posts_df['title'].fillna('').astype(str) + ' ' +
        posts_df.get(
            'selftext',
            pd.Series('', index=posts_df.index)
        ).fillna('').astype(str)
    )
    .apply(tesla_relevance)
)

posts_df['context_topic'] = (
    posts_df['title'].apply(detect_topic)
)

posts_df['VADER_Sentiment'] = (
    (
        posts_df['title'].fillna('').astype(str) + ' ' +
        posts_df.get(
            'selftext',
            pd.Series('', index=posts_df.index)
        ).fillna('').astype(str)
    )
    .apply(vader_sentiment)
)

posts_df = calculate_engagement_score(posts_df)

reddit_sentiment_df = combine_reddit_data(
    comments_df.copy(),
    posts_df.copy()
)
reddit_sentiment_df = apply_fast_sentiment(reddit_sentiment_df)

daily_finbert = aggregate_daily_text(reddit_sentiment_df)
daily_finbert = apply_daily_finbert(daily_finbert)

sentiment_daily = (
    reddit_sentiment_df
    .groupby('Date')
    .agg({
        'VADER_Sentiment': 'mean',
        'TextBlob_Sentiment': 'mean',
        'score': 'mean',
        'text': 'count'
    })
    .reset_index()
    .rename(
        columns={
            'VADER_Sentiment': 'Avg_Sentiment',
            'score': 'Avg_Reddit_Score',
            'text': 'Comment_Count'
        }
    )
)

sentiment_daily = sentiment_daily.merge(
    daily_finbert[
        ['Date', 'FinBERT_Sentiment']
    ],
    on='Date',
    how='left'
)

with pd.ExcelWriter('data/processed/preprocessed_data_1.xlsx') as writer:
    posts_df.to_excel(writer, sheet_name='cleaned_posts', index=False)
    comments_df.to_excel(writer, sheet_name='cleaned_comments', index=False)
    merged_relation.to_excel(writer, sheet_name='post_comment_links', index=False)

financial_df = preprocess_financial_data(
    'data/raw/tsla_yahoo_finance.csv'
)

financial_df = apply_kalman_filter(financial_df)

financial_df.to_csv(
    'data/processed/preprocessed_data_2.csv',
    index=False
)

advanced_daily = (
    posts_df.groupby('Date')
    .agg({
        'engagement_score': 'mean',
        'tesla_relevance': 'mean',
        'comment_count': 'mean',
        'repost_count': 'mean',
        'unique_author_count': 'mean',
        'discussion_intensity': 'mean',
        'information_diffusion_score': 'mean',
        'sentiment_magnitude': 'mean',
        'VADER_Sentiment': 'mean'
    })
    .reset_index()
)

advanced_daily = advanced_daily.rename(
    columns={
        'VADER_Sentiment': 'Post_VADER_Sentiment'
    }
)

reddit_daily = sentiment_daily.merge(
    advanced_daily,
    on='Date',
    how='left'
)

final_df = merge_reddit_financial(
    reddit_daily,
    financial_df
)

final_df.to_csv(
    'data/processed/preprocessed_data_3.csv',
    index=False
)

print('All preprocessing completed successfully.')
