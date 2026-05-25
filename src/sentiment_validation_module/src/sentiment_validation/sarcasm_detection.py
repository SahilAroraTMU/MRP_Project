import os
import pandas as pd


OUTPUT_DIR = (
    'outputs/sentiment_validation'
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ===================================================
# LOAD SARCASM MODEL
# ===================================================

sarcasm_pipeline = None
sarcasm_unavailable = False
SARCASM_BATCH_SIZE = int(os.environ.get('MRP_SARCASM_BATCH_SIZE', '64'))


def get_sarcasm_pipeline():
    global sarcasm_pipeline
    global sarcasm_unavailable

    if sarcasm_pipeline is not None:
        return sarcasm_pipeline

    if sarcasm_unavailable:
        return None

    if (
        os.environ.get('MRP_OFFLINE') == '1'
        or os.environ.get('MRP_SKIP_SARCASM_MODEL') == '1'
    ):
        sarcasm_unavailable = True
        reason = (
            'MRP_OFFLINE=1'
            if os.environ.get('MRP_OFFLINE') == '1'
            else 'MRP_SKIP_SARCASM_MODEL=1'
        )
        print(
            'Sarcasm model unavailable; using UNKNOWN sarcasm labels. '
            f'Reason: {reason}'
        )
        return None

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        model_name = 'cardiffnlp/twitter-roberta-base-irony'
        local_files_only = os.environ.get('MRP_SARCASM_LOCAL_ONLY') == '1'
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )

        sarcasm_pipeline = pipeline(
            'text-classification',
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
        )
        return sarcasm_pipeline

    except Exception as exc:
        sarcasm_unavailable = True
        print(
            'Sarcasm model unavailable; using UNKNOWN sarcasm labels. '
            f'Reason: {exc}'
        )
        return None


# ===================================================
# TRANSFORMER SARCASM DETECTION
# ===================================================

def detect_sarcasm(text):

    try:
        pipeline_model = get_sarcasm_pipeline()

        if pipeline_model is None:
            return {
                'label': 'UNKNOWN',
                'score': 0
            }

        result = pipeline_model(
            str(text)[:512]
        )[0]

        return {
            'label': result['label'],
            'score': result['score']
        }

    except:

        return {
            'label': 'UNKNOWN',
            'score': 0
        }


# ===================================================
# APPLY SARCASM DETECTION
# ===================================================

def generate_sarcasm_analysis(df):
    df = df.copy()
    pipeline_model = get_sarcasm_pipeline()

    if pipeline_model is None:
        df['Sarcasm_Label'] = 'UNKNOWN'
        df['Sarcasm_Score'] = 0
    else:
        texts = (
            df['text']
            .astype(str)
            .str[:512]
            .tolist()
        )

        results = pipeline_model(
            texts,
            batch_size=SARCASM_BATCH_SIZE
        )

        df['Sarcasm_Label'] = [
            result['label']
            for result in results
        ]

        df['Sarcasm_Score'] = [
            result['score']
            for result in results
        ]

    df.to_csv(

        f'{OUTPUT_DIR}/sarcasm_analysis.csv',

        index=False
    )

    print(
        'Saved advanced sarcasm analysis.'
    )
