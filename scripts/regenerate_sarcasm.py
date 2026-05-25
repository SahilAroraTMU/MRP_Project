from pathlib import Path
import os
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment_validation_module.src.sentiment_validation.sarcasm_detection import (
    get_sarcasm_pipeline,
    SARCASM_BATCH_SIZE,
)
from src.sentiment_analysis import finbert_sentiment


def main():
    path = (
        PROJECT_ROOT
        / 'outputs'
        / 'sentiment_validation'
        / 'sarcasm_analysis.csv'
    )
    output_path = Path(
        os.environ.get(
            'MRP_SARCASM_OUTPUT',
            str(path),
        )
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    temp_path = output_path.with_suffix('.tmp.csv')
    chunk_size = int(os.environ.get('MRP_SARCASM_CHUNK_SIZE', '5000'))
    sample_size = os.environ.get('MRP_SARCASM_SAMPLE_SIZE')
    pipeline_model = get_sarcasm_pipeline()

    if pipeline_model is None:
        raise RuntimeError('Sarcasm model is unavailable.')

    rows_written = 0
    first_chunk = True

    reader = pd.read_csv(
        path,
        usecols=[
            'Date',
            'text',
            'score',
            'VADER_Sentiment',
            'TextBlob_Sentiment',
            'FinBERT_Sentiment',
        ],
        chunksize=chunk_size,
    )

    for chunk in reader:
        if sample_size is not None:
            remaining = int(sample_size) - rows_written

            if remaining <= 0:
                break

            chunk = chunk.head(remaining)

        texts = (
            chunk['text']
            .astype(str)
            .str[:512]
            .tolist()
        )
        chunk['FinBERT_Sentiment'] = [
            finbert_sentiment(text)
            for text in texts
        ]
        results = pipeline_model(
            texts,
            batch_size=SARCASM_BATCH_SIZE,
        )
        chunk['Sarcasm_Label'] = [
            result['label']
            for result in results
        ]
        chunk['Sarcasm_Score'] = [
            result['score']
            for result in results
        ]
        chunk.to_csv(
            temp_path,
            index=False,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
        )

        rows_written += len(chunk)
        first_chunk = False
        print(f'Processed {rows_written} rows', flush=True)

    temp_path.replace(output_path)

    updated = pd.read_csv(output_path)
    print(updated['Sarcasm_Label'].value_counts(dropna=False).head(10))
    print(updated['Sarcasm_Score'].describe())
    print(updated['FinBERT_Sentiment'].describe())
    print(
        'sarcasm_non_zero',
        int(updated['Sarcasm_Score'].ne(0).sum()),
        'of',
        len(updated),
    )
    print(
        'finbert_non_zero',
        int(updated['FinBERT_Sentiment'].ne(0).sum()),
        'of',
        len(updated),
    )


if __name__ == '__main__':
    main()
