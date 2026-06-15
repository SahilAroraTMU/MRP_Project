
import re

import pandas as pd

def load_reddit_data(posts_path, comments_path):
    posts_df = pd.read_csv(posts_path)
    comments_df = pd.read_csv(comments_path)
    return posts_df, comments_df

def preprocess_timestamps(df):
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    df['Date'] = df['created_utc'].dt.date
    return df

def _normalize_text(value):
    value = '' if pd.isna(value) else str(value).lower()
    value = re.sub(r'http\S+|www\.\S+', ' ', value)
    value = re.sub(r'[^a-z0-9\s]', ' ', value)
    return re.sub(r'\s+', ' ', value).strip()

def _engagement_rank(df):
    score = pd.to_numeric(
        df['score'] if 'score' in df.columns else pd.Series(0, index=df.index),
        errors='coerce'
    ).fillna(0)

    comments = pd.to_numeric(
        (
            df['comment_count']
            if 'comment_count' in df.columns
            else pd.Series(0, index=df.index)
        ),
        errors='coerce'
    ).fillna(0)

    return score + comments

def _author_or_occurrence(df):
    if 'author' in df.columns:
        return df['author'].astype(str).fillna('[missing_author]')

    if 'id' in df.columns:
        return df['id'].astype(str)

    return pd.Series(
        df.index.astype(str),
        index=df.index
    )

def preprocess_post_reposts(posts_df):
    posts_df = posts_df.copy()
    posts_df = posts_df.sort_values('created_utc').reset_index(drop=True)

    posts_df['normalized_post_title'] = (
        posts_df['title']
        if 'title' in posts_df.columns
        else pd.Series('', index=posts_df.index)
    ).fillna('').astype(str).apply(_normalize_text)
    posts_df['normalized_post_text'] = posts_df['normalized_post_title']

    posts_df['author_key'] = _author_or_occurrence(posts_df)
    posts_df['_engagement_rank'] = _engagement_rank(posts_df)

    grouped = posts_df.groupby('normalized_post_title', dropna=False)
    original_time = grouped['created_utc'].transform('min')
    occurrence_count = grouped['normalized_post_title'].transform('size')
    unique_authors = grouped['author_key'].transform('nunique')

    posts_df['original_post_time'] = original_time
    posts_df['original_post_flag'] = (
        posts_df['created_utc'].eq(posts_df['original_post_time'])
    ).astype(int)
    posts_df['repost_flag'] = (
        posts_df['original_post_flag'].eq(0)
    ).astype(int)

    # Repost identification is title based across all available authors:
    # the first chronological title occurrence is kept as the original, while
    # later same-title rows are removed and summarized as repost_count.
    posts_df['repost_count'] = (occurrence_count - 1).astype(int)
    posts_df['unique_author_count'] = unique_authors.astype(int)

    posts_df = (
        posts_df
        .sort_values(['normalized_post_title', 'created_utc'])
        .drop_duplicates(
            subset=['normalized_post_title'],
            keep='first'
        )
        .sort_values('created_utc')
        .reset_index(drop=True)
    )

    posts_df['original_post_flag'] = 1
    posts_df['repost_flag'] = 0

    return posts_df.drop(columns=['_engagement_rank'])

def preprocess_comment_reposts(comments_df):
    comments_df = comments_df.copy()
    comments_df = comments_df.sort_values('created_utc').reset_index(drop=True)

    comments_df['normalized_comment_text'] = (
        comments_df.get('body', '')
        .fillna('')
        .astype(str)
        .apply(_normalize_text)
    )
    comments_df['author_key'] = _author_or_occurrence(comments_df)
    comments_df['_engagement_rank'] = _engagement_rank(comments_df)

    # Comment duplicate aggregation mirrors post handling: same author and
    # same normalized text is collapsed to the highest-engagement occurrence.
    comments_df = (
        comments_df
        .sort_values(
            ['normalized_comment_text', 'author_key', '_engagement_rank', 'created_utc'],
            ascending=[True, True, False, True]
        )
        .drop_duplicates(
            subset=['normalized_comment_text', 'author_key'],
            keep='first'
        )
        .sort_values('created_utc')
        .reset_index(drop=True)
    )

    grouped = comments_df.groupby('normalized_comment_text', dropna=False)
    occurrence_count = grouped['normalized_comment_text'].transform('size')
    first_time = grouped['created_utc'].transform('min')

    comments_df['original_comment_flag'] = (
        comments_df['created_utc'].eq(first_time)
    ).astype(int)
    comments_df['comment_repost_flag'] = (
        comments_df['original_comment_flag'].eq(0)
    ).astype(int)
    comments_df['comment_repost_count'] = (occurrence_count - 1).astype(int)

    return comments_df.drop(columns=['_engagement_rank'])

def remove_post_duplicates(posts_df):
    return preprocess_post_reposts(posts_df)

def remove_comment_duplicates(comments_df):
    return preprocess_comment_reposts(comments_df)
