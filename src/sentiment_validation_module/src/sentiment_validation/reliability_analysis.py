
import os
import pandas as pd

from src.sentiment_analysis import finbert_sentiment

OUTPUT_DIR = 'outputs/sentiment_validation'
os.makedirs(OUTPUT_DIR, exist_ok=True)
VALIDATION_SAMPLE_SIZE = int(os.environ.get('MRP_VALIDATION_SAMPLE_SIZE', '100'))

def sentiment_label(score):

    if score >= 0.75:
        return 'Extremely Positive'

    elif score >= 0.25:
        return 'Positive'

    elif score <= -0.75:
        return 'Extremely Negative'

    elif score <= -0.25:
        return 'Negative'

    else:
        return 'Neutral'

def create_reliability_dataset(df):
    sample_size = min(
        VALIDATION_SAMPLE_SIZE,
        len(df)
    )

    sample_df = df.sample(
        n=sample_size,
        random_state=42
    ).copy()

    sample_df['FinBERT_Sentiment'] = (
        sample_df['text']
        .astype(str)
        .apply(finbert_sentiment)
    )

    sample_df['VADER_Label'] = (
        sample_df['VADER_Sentiment']
        .apply(sentiment_label)
    )

    sample_df['TextBlob_Label'] = (
        sample_df['TextBlob_Sentiment']
        .apply(sentiment_label)
    )

    sample_df['FinBERT_Label'] = (
        sample_df['FinBERT_Sentiment']
        .apply(sentiment_label)
    )

    sample_df.to_csv(
        f'{OUTPUT_DIR}/reliability_analysis.csv',
        index=False
    )

    print('Saved reliability analysis.')
