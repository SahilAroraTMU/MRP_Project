#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
import math
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment_analysis import get_finbert_pipeline, apply_fast_sentiment

OUTPUT = PROJECT_ROOT / 'outputs' / 'eda_storytelling' / 'raw_sentiment_finbert.csv'
CHARS = int(os.environ.get('FINBERT_CHARS_PER_CHUNK', '1800'))
MAX_CHUNKS = int(os.environ.get('FINBERT_MAX_CHUNKS', '2'))
BATCH_SIZE = int(os.environ.get('FINBERT_BATCH_SIZE', '32'))
PRINT_EVERY = int(os.environ.get('FINBERT_PRINT_EVERY', '100'))

os.makedirs(OUTPUT.parent, exist_ok=True)

print('Loading raw posts and comments...')
posts = pd.read_csv(PROJECT_ROOT / 'data' / 'raw' / 'reddit_posts.csv', usecols=['created_utc','title','selftext'], low_memory=False)
comments = pd.read_csv(PROJECT_ROOT / 'data' / 'raw' / 'reddit_comments.csv', usecols=['created_utc','body'], low_memory=False)

for df in (posts, comments):
    if 'created_utc' in df.columns:
        df['Date'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    else:
        df['Date'] = pd.NaT

posts['text'] = posts['title'].fillna('').astype(str) + ' ' + posts.get('selftext', pd.Series('', index=posts.index)).fillna('').astype(str)
comments['text'] = comments['body'].fillna('').astype(str)
posts['source'] = 'post'
comments['source'] = 'comment'
posts['row_id'] = posts.index
comments['row_id'] = comments.index

combined = pd.concat([
    posts[['Date','text','source','row_id']],
    comments[['Date','text','source','row_id']]
], ignore_index=True)
combined = combined.dropna(subset=['text']).copy()

print('Applying fast (VADER/TextBlob) sentiment...')
combined = apply_fast_sentiment(combined)

print('Deduplicating texts...')
unique_texts = combined['text'].drop_duplicates().reset_index(drop=True)
print('Unique texts to score:', len(unique_texts))

pipeline = get_finbert_pipeline()
if pipeline is None:
    from transformers import pipeline as hf_pipeline
    pipeline = hf_pipeline('sentiment-analysis', model='ProsusAI/finbert', tokenizer='ProsusAI/finbert', truncation=True, max_length=512)

# Tune torch threads for CPU
try:
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 4)))
except Exception:
    pass

scores = []
start_time = time.time()
for i in range(0, len(unique_texts), BATCH_SIZE):
    batch_texts = [str(t)[:CHARS * MAX_CHUNKS] for t in unique_texts[i:i+BATCH_SIZE]]
    out = pipeline(batch_texts, top_k=None)
    # normalize outputs
    for entry in out:
        # entry might be a list of label dicts or a single dict
        if isinstance(entry, list):
            probs = {item['label'].lower(): item['score'] for item in entry}
        elif isinstance(entry, dict):
            probs = {entry['label'].lower(): entry['score']}
        else:
            probs = {}
        score = probs.get('positive', 0.0) - probs.get('negative', 0.0)
        scores.append(score)

    if (i // BATCH_SIZE) % PRINT_EVERY == 0:
        elapsed = time.time() - start_time
        processed = min(i + BATCH_SIZE, len(unique_texts))
        per = elapsed / processed if processed else 0
        remaining = (len(unique_texts) - processed) * per
        print(f'Processed {processed}/{len(unique_texts)} unique texts — elapsed {elapsed:.1f}s — est remaining {remaining/60:.1f}m')
        sys.stdout.flush()

# Build DataFrame of scores
scores_df = pd.DataFrame({'text': unique_texts, 'FinBERT_Sentiment': scores})
print('Merging scores back to full dataset...')
combined = combined.merge(scores_df, on='text', how='left')

print('Saving to', OUTPUT)
combined.to_csv(OUTPUT, index=False)
print('Done — saved rows', len(combined))
