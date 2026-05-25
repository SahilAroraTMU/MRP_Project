from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment_validation_module.src.sentiment_validation.kappa_analysis import (
    calculate_kappa_scores,
)
from src.sentiment_validation_module.src.sentiment_validation.reliability_analysis import (
    create_reliability_dataset,
)


def main():
    validation_path = (
        PROJECT_ROOT
        / 'outputs'
        / 'sentiment_validation'
        / 'sarcasm_analysis.csv'
    )

    df = pd.read_csv(
        validation_path,
        usecols=[
            'Date',
            'text',
            'score',
            'VADER_Sentiment',
            'TextBlob_Sentiment',
            'FinBERT_Sentiment',
        ],
    )

    create_reliability_dataset(df)
    calculate_kappa_scores(df)

    reliability = pd.read_csv(
        PROJECT_ROOT
        / 'outputs'
        / 'sentiment_validation'
        / 'reliability_analysis.csv'
    )
    print(reliability['FinBERT_Sentiment'].describe())
    print(
        'non_zero',
        int(reliability['FinBERT_Sentiment'].ne(0).sum()),
        'of',
        len(reliability),
    )


if __name__ == '__main__':
    main()
