
import os
import pandas as pd

from sklearn.metrics import (
    cohen_kappa_score
)

from src.sentiment_analysis import finbert_sentiment

OUTPUT_DIR = 'outputs/sentiment_validation'
os.makedirs(OUTPUT_DIR, exist_ok=True)
VALIDATION_SAMPLE_SIZE = int(os.environ.get('MRP_VALIDATION_SAMPLE_SIZE', '100'))

def classify_sentiment(x):

    if x > 0.05:
        return 'positive'

    elif x < -0.05:
        return 'negative'

    else:
        return 'neutral'

def calculate_kappa_scores(df):
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

    vader = (
        sample_df['VADER_Sentiment']
        .apply(classify_sentiment)
    )

    textblob = (
        sample_df['TextBlob_Sentiment']
        .apply(classify_sentiment)
    )

    finbert = (
        sample_df['FinBERT_Sentiment']
        .apply(classify_sentiment)
    )

    results = pd.DataFrame([

        {
            'Comparison':
                'VADER vs TextBlob',

            'Sample_Size':
                sample_size,

            'Kappa':
                cohen_kappa_score(
                    vader,
                    textblob
                )
        },

        {
            'Comparison':
                'VADER vs FinBERT',

            'Sample_Size':
                sample_size,

            'Kappa':
                cohen_kappa_score(
                    vader,
                    finbert
                )
        },

        {
            'Comparison':
                'TextBlob vs FinBERT',

            'Sample_Size':
                sample_size,

            'Kappa':
                cohen_kappa_score(
                    textblob,
                    finbert
                )
        }

    ])

    results.to_csv(
        f'{OUTPUT_DIR}/kappa_scores.csv',
        index=False
    )

    print('Saved kappa scores.')
