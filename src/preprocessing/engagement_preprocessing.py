
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ENGAGEMENT_INPUT_COLUMNS = [
    'repost_count',
    'comment_count',
    'score',
    'tesla_relevance',
]

def add_information_diffusion_score(df):
    df = df.copy()
    reposts = pd.to_numeric(
        df.get('repost_count', 0),
        errors='coerce'
    ).fillna(0)
    authors = pd.to_numeric(
        df.get('unique_author_count', 0),
        errors='coerce'
    ).fillna(0)

    # Information diffusion captures both repeat circulation and breadth of
    # participation, following investor-attention work where wider discussion
    # spread is treated as a stronger market information signal.
    df['information_diffusion_score'] = reposts * authors
    return df

def add_sentiment_magnitude(df):
    df = df.copy()

    sentiment_column = (
        'FinBERT_Sentiment'
        if 'FinBERT_Sentiment' in df.columns
        else 'VADER_Sentiment'
    )

    if sentiment_column not in df.columns:
        df[sentiment_column] = 0

    df['sentiment_source'] = sentiment_column
    df['sentiment_magnitude'] = (
        pd.to_numeric(
            df[sentiment_column],
            errors='coerce'
        )
        .fillna(0)
        .abs()
    )

    return df

def calculate_engagement_score(df):
    df = add_information_diffusion_score(df)
    df = add_sentiment_magnitude(df)

    for column in ENGAGEMENT_INPUT_COLUMNS:
        if column not in df.columns:
            df[column] = 0

        df[column] = pd.to_numeric(
            df[column],
            errors='coerce'
        ).fillna(0)

    normalized = pd.DataFrame(
        MinMaxScaler().fit_transform(df[ENGAGEMENT_INPUT_COLUMNS]),
        columns=[f'{column}_normalized' for column in ENGAGEMENT_INPUT_COLUMNS],
        index=df.index
    )

    df = pd.concat(
        [
            df.drop(columns=normalized.columns, errors='ignore'),
            normalized
        ],
        axis=1
    )

    # Engagement score construction replaces the former unscaled log formula
    # with a weighted MinMax-normalized index. The weights privilege discussion
    # depth, Reddit voting, Tesla specificity, information diffusion, and the
    # absolute strength of the selected sentiment model, with VADER preferred
    # for this preprocessing path and FinBERT retained as a compatibility
    # fallback when older processed files are supplied.
    df['engagement_score'] = (
        0.25 * df['repost_count_normalized'] +
        0.25 * df['comment_count_normalized'] +
        0.25 * df['score_normalized'] +
        0.25 * df['tesla_relevance_normalized']
    )

    df['reddit_score'] = df['score']
    return df

def calculate_engagement_factor(df):
    df = calculate_engagement_score(df)
    df['engagement_factor'] = df['engagement_score']
    return df
