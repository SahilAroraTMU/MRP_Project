from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

import os

from textblob import TextBlob


# ---------------------------------------------------
# Initialize Models
# ---------------------------------------------------

analyzer = SentimentIntensityAnalyzer()

finbert_pipeline = None
finbert_unavailable = False
FINBERT_CHARS_PER_CHUNK = 1800
FINBERT_MAX_CHUNKS = int(os.environ.get('FINBERT_MAX_CHUNKS', '2'))


def get_finbert_pipeline():
    global finbert_pipeline
    global finbert_unavailable

    if finbert_pipeline is not None:
        return finbert_pipeline

    if finbert_unavailable:
        return None

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline
        )

        local_files_only = os.environ.get('MRP_OFFLINE') == '1'
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only
        )

        finbert_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512
        )
        return finbert_pipeline

    except Exception as exc:
        finbert_unavailable = True
        print(
            "FinBERT unavailable; using neutral FinBERT sentiment. "
            f"Reason: {exc}"
        )
        return None


# ---------------------------------------------------
# VADER Sentiment
# ---------------------------------------------------

def vader_sentiment(text):

    score = analyzer.polarity_scores(
        str(text)
    )

    return score['compound']


# ---------------------------------------------------
# TextBlob Sentiment
# ---------------------------------------------------

def textblob_sentiment(text):

    try:

        analysis = TextBlob(
            str(text)
        )

        return analysis.sentiment.polarity

    except:

        return 0


# ---------------------------------------------------
# FinBERT Sentiment
# ---------------------------------------------------

def finbert_sentiment(text):

    try:
        pipeline_model = get_finbert_pipeline()

        if pipeline_model is None:
            return 0

        text = str(text)
        chunks = [
            text[start:start + FINBERT_CHARS_PER_CHUNK]
            for start in range(
                0,
                min(len(text), FINBERT_CHARS_PER_CHUNK * FINBERT_MAX_CHUNKS),
                FINBERT_CHARS_PER_CHUNK
            )
        ] or ['']

        results = pipeline_model(
            chunks,
            top_k=None
        )
        scores = []

        for result in results:
            probabilities = {
                item['label'].lower(): item['score']
                for item in result
            }
            scores.append(
                probabilities.get('positive', 0)
                - probabilities.get('negative', 0)
            )

        return sum(scores) / len(scores)

    except:

        return 0


# ---------------------------------------------------
# Fast Sentiment Pipeline
# ---------------------------------------------------

def apply_fast_sentiment(df):

    # Limit text size
    df['text'] = (
        df['text']
        .astype(str)
        .str[:512]
    )

    # VADER
    df['VADER_Sentiment'] = (
        df['text']
        .apply(vader_sentiment)
    )

    # TextBlob
    df['TextBlob_Sentiment'] = (
        df['text']
        .apply(textblob_sentiment)
    )

    return df


# ---------------------------------------------------
# Aggregate Daily Text for FinBERT
# ---------------------------------------------------

def aggregate_daily_text(df):

    daily_text = (
        df.groupby('Date')['text']
        .apply(
            lambda x: ' '.join(x)
        )
        .reset_index()
    )

    return daily_text


# ---------------------------------------------------
# Apply FinBERT Daily
# ---------------------------------------------------

def apply_daily_finbert(df):

    # Limit text size
    df['text'] = (
        df['text']
        .astype(str)
    )

    df['FinBERT_Sentiment'] = (
        df['text']
        .apply(finbert_sentiment)
    )

    return df
