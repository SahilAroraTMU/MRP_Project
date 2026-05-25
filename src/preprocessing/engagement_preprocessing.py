
import numpy as np

def calculate_engagement_factor(df):
    df['engagement_factor'] = (
        1 + np.log1p(df['comment_count']) + (df['score'] / 100)
    )
    return df
