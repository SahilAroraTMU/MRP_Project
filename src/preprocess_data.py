import pandas as pd

# ---------------------------------------------------
# Financial Dataset Preprocessing
# ---------------------------------------------------

def preprocess_financial_data():

    raw_path = 'data/raw/tsla_yahoo_finance.csv'

    df = pd.read_csv(raw_path)

    if df.empty or 'Date' not in df.columns or df['Date'].isna().all():
        df = pd.read_csv(
            'data/processed/merged_dataset.csv',
            usecols=[
                'Date',
                'Close',
                'High',
                'Low',
                'Open',
                'Volume'
            ]
        )

    # Step 1 - Date Conversion
    df['Date'] = pd.to_datetime(
        df['Date']
    )

    # Step 2 - Numeric Conversion
    numeric_cols = [
        'Close',
        'High',
        'Low',
        'Open',
        'Volume'
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    # Step 3 - Remove Missing Values
    df = df.dropna()

    # Step 4 - Sort Chronologically
    df = df.sort_values('Date')

    return df


# ---------------------------------------------------
# Reddit Dataset Preprocessing
# ---------------------------------------------------

def preprocess_reddit_data(file_path):

    df = pd.read_csv(file_path)

    # Step 1 - Timestamp Conversion
    df['created_utc'] = pd.to_datetime(
        df['created_utc'],
        unit='s'
    )

    # Step 2 - Create Date Column
    df['Date'] = (
        df['created_utc']
        .dt.date
    )

    # Step 3 - Remove Missing Values
    df = df.dropna()

    return df


# ---------------------------------------------------
# Combine Reddit Posts + Comments
# ---------------------------------------------------

def combine_reddit_data(
    comments_df,
    posts_df
):

    comments_df['text'] = (
        comments_df['body']
        .astype(str)
    )

    posts_df['text'] = (
        posts_df['title']
        .astype(str)
    )

    combined_df = pd.concat([
        comments_df[['Date', 'text', 'score']],
        posts_df[['Date', 'text', 'score']]
    ])

    return combined_df


# ---------------------------------------------------
# Aggregate Daily Reddit Sentiment
# ---------------------------------------------------

def aggregate_reddit_sentiment(df):

    reddit_daily = (
        df.groupby('Date')
        .agg({
            'VADER_Sentiment': 'mean',
            'TextBlob_Sentiment': 'mean',
            'FinBERT_Sentiment': 'mean',
            'score': 'mean',
            'text': 'count'
        })
        .reset_index()
    )

    reddit_daily.rename(
        columns={
            'VADER_Sentiment': 'Avg_Sentiment',
            'score': 'Avg_Reddit_Score',
            'text': 'Comment_Count'
        },
        inplace=True
    )

    return reddit_daily


# ---------------------------------------------------
# Merge Reddit + TSLA Data
# ---------------------------------------------------

def merge_datasets(
    financial_df,
    reddit_daily
):

    financial_df['Date'] = (
        financial_df['Date']
        .dt.date
    )

    merged_df = pd.merge(
        financial_df,
        reddit_daily,
        on='Date',
        how='inner'
    )

    merged_df.to_csv(
        'data/processed/merged_dataset.csv',
        index=False
    )

    return merged_df
