
import pandas as pd

def merge_reddit_financial(reddit_df, financial_df):

    financial_df['Date'] = financial_df['Date'].dt.date

    return pd.merge(
        financial_df,
        reddit_df,
        on='Date',
        how='inner'
    )
