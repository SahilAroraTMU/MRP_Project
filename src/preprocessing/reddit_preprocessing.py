
import pandas as pd

def load_reddit_data(posts_path, comments_path):
    posts_df = pd.read_csv(posts_path)
    comments_df = pd.read_csv(comments_path)
    return posts_df, comments_df

def preprocess_timestamps(df):
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    df['Date'] = df['created_utc'].dt.date
    return df

def _available_columns(df, columns):
    return [column for column in columns if column in df.columns]

def remove_post_duplicates(posts_df):
    subset = _available_columns(
        posts_df,
        ['id', 'author', 'title', 'created_utc']
    )

    if not subset:
        return posts_df.drop_duplicates()

    return posts_df.drop_duplicates(subset=subset)

def remove_comment_duplicates(comments_df):
    subset = _available_columns(
        comments_df,
        ['id', 'author', 'body', 'created_utc']
    )

    if not subset:
        return comments_df.drop_duplicates()

    return comments_df.drop_duplicates(subset=subset)
