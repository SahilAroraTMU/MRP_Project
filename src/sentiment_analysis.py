from vaderSentiment.vaderSentiment import (
    SentimentIntensityAnalyzer
)

from textblob import TextBlob


# ---------------------------------------------------
# Initialize Models
# ---------------------------------------------------

analyzer = SentimentIntensityAnalyzer()

finbert_pipeline = None
finbert_unavailable = False


def get_finbert_pipeline():
    global finbert_pipeline
    global finbert_unavailable

    if finbert_pipeline is not None:
        return finbert_pipeline

    if finbert_unavailable:
        return None

    try:
        from transformers import pipeline

        finbert_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            local_files_only=True
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

        result = pipeline_model(
            str(text)
        )[0]

        label = result['label']
        score = result['score']

        if label == 'positive':
            return score

        elif label == 'negative':
            return -score

        else:
            return 0

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
        .str[:512]
    )

    df['FinBERT_Sentiment'] = (
        df['text']
        .apply(finbert_sentiment)
    )

    return df
