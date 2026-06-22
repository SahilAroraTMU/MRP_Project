#!/usr/bin/env python3
"""
Standalone dual-GPU FinBERT sentiment analysis for Kaggle Reddit dataset.
Splits comments 50/50, processes both halves sequentially on GPU 0 and GPU 1.
Auto-detects dataset location and filenames.
"""

import argparse
import pandas as pd
import torch
from pathlib import Path
import sys

# Handle Kaggle notebook kernel args
try:
    import inspect
except:
    pass


def get_project_root() -> Path:
    """Get project root, handling Kaggle environment."""
    # Check if we're in Kaggle environment
    kaggle_working = Path('/kaggle/working')
    if kaggle_working.exists():
        return kaggle_working
    
    if '__file__' in globals():
        return Path(__file__).resolve().parents[2]

    try:
        import inspect
        return Path(inspect.getsourcefile(lambda: 0)).resolve().parents[2]
    except Exception:
        return Path.cwd()


def find_dataset_root(input_root=None) -> Path:
    """Find dataset root in Kaggle environment."""
    if input_root:
        return Path(input_root)
    
    kag_input = Path('/kaggle/input')
    
    # Search recursively for TSLA dataset files in /kaggle/input
    if kag_input.exists():
        for root in kag_input.rglob('.'):
            if not root.is_dir():
                continue
            if (root / 'one-year-of-tsla-on-reddit-comments.csv').exists() and \
               (root / 'one-year-of-tsla-on-reddit-posts.csv').exists():
                return root
    
    # Fallback
    return Path('/kaggle/input')


def find_files(root: Path):
    """Find comments and posts files."""
    comment_file = root / 'one-year-of-tsla-on-reddit-comments.csv'
    post_file = root / 'one-year-of-tsla-on-reddit-posts.csv'
    
    return comment_file, post_file


def finbert_scores(texts, pipeline, batch_size=16):
    """Score texts using FinBERT, chunking long texts."""
    scores = []
    
    for text in texts:
        if pd.isna(text) or text == '':
            scores.append({'positive': 0.0, 'negative': 0.0, 'neutral': 1.0})
            continue
        
        text_str = str(text)
        # Split into chunks of max 1800 chars
        chunks = [text_str[i:i+1800] for i in range(0, len(text_str), 1800)]
        chunk_scores = []
        
        for chunk in chunks:
            try:
                result = pipeline(chunk[:512])  # FinBERT max is 512 tokens
                chunk_scores.append(result[0] if isinstance(result, list) else result)
            except Exception as e:
                chunk_scores.append({'label': 'neutral', 'score': 1.0})
        
        # Average across chunks
        if chunk_scores:
            avg_pos = sum(s.get('score', 0.0) if s.get('label') == 'positive' else 0.0 for s in chunk_scores) / len(chunk_scores)
            avg_neg = sum(s.get('score', 0.0) if s.get('label') == 'negative' else 0.0 for s in chunk_scores) / len(chunk_scores)
            avg_neu = sum(s.get('score', 0.0) if s.get('label') == 'neutral' else 0.0 for s in chunk_scores) / len(chunk_scores)
            scores.append({'positive': avg_pos, 'negative': avg_neg, 'neutral': avg_neu})
        else:
            scores.append({'positive': 0.0, 'negative': 0.0, 'neutral': 1.0})
    
    return scores


def process_shard(device, comments_df, posts_df, batch_size):
    """Process a shard of data on a specific GPU."""
    try:
        torch.cuda.set_device(device)
        from transformers import pipeline
        
        print(f'[GPU {device}] Loading FinBERT model...')
        pipe = pipeline('sentiment-analysis', model='ProsusAI/finbert', device=device)
        
        # Determine text column names
        comment_text_col = 'text' if 'text' in comments_df.columns else comments_df.columns[0]
        post_text_col = 'text' if 'text' in posts_df.columns else posts_df.columns[0]
        
        # Score comments
        print(f'[GPU {device}] Scoring {len(comments_df)} comments (column: {comment_text_col})...')
        comment_scores = finbert_scores(comments_df[comment_text_col].tolist(), pipe, batch_size)
        comments_df_out = comments_df.copy()
        for key in ['positive', 'negative', 'neutral']:
            comments_df_out[key] = [s[key] for s in comment_scores]
        
        # Score posts
        print(f'[GPU {device}] Scoring {len(posts_df)} posts (column: {post_text_col})...')
        post_scores = finbert_scores(posts_df[post_text_col].tolist(), pipe, batch_size)
        posts_df_out = posts_df.copy()
        for key in ['positive', 'negative', 'neutral']:
            posts_df_out[key] = [s[key] for s in post_scores]
        
        # Aggregate daily (look for date column)
        date_col = None
        for col in ['date', 'Date', 'created_at', 'timestamp']:
            if col in comments_df_out.columns:
                date_col = col
                break
        
        daily = pd.DataFrame()
        if date_col:
            daily = comments_df_out.groupby(date_col)[['positive', 'negative', 'neutral']].mean().reset_index()
            daily.columns = ['Date', 'positive', 'negative', 'neutral']
        
        print(f'[GPU {device}] Shard complete')
        return {
            'comments': comments_df_out,
            'posts': posts_df_out,
            'daily': daily
        }
    
    except Exception as e:
        print(f'[GPU {device}] Error: {e}')
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description='Dual-GPU FinBERT sentiment analysis')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--input-root', default=None)
    parser.add_argument('--output-dir', default='outputs/eda_storytelling')
    args, unknown = parser.parse_known_args()
    
    if unknown:
        print('Ignoring extra arguments:', unknown)
    
    # Detect paths
    dataset_root = find_dataset_root(args.input_root)
    comments_path, posts_path = find_files(dataset_root)
    project_root = get_project_root()
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'Dataset root: {dataset_root}')
    print(f'Comments: {comments_path}')
    print(f'Posts: {posts_path}')
    print(f'Output: {output_dir}')
    
    # Load data
    print('Loading comments...')
    comments_full = pd.read_csv(comments_path)
    
    print('Loading posts...')
    posts_full = pd.read_csv(posts_path)
    
    # Split comments 50/50
    print(f'Splitting {len(comments_full)} comments...')
    mid = len(comments_full) // 2
    comments_shard0 = comments_full.iloc[:mid].reset_index(drop=True)
    comments_shard1 = comments_full.iloc[mid:].reset_index(drop=True)
    
    posts_shard0 = posts_full.iloc[:len(posts_full)//2].reset_index(drop=True)
    posts_shard1 = posts_full.iloc[len(posts_full)//2:].reset_index(drop=True)
    
    # Process shards sequentially on different GPUs
    print('Processing GPU 0...')
    result0 = process_shard(device=0, comments_df=comments_shard0, posts_df=posts_shard0, batch_size=args.batch_size)
    
    print('Processing GPU 1...')
    result1 = process_shard(device=1, comments_df=comments_shard1, posts_df=posts_shard1, batch_size=args.batch_size)
    
    # Merge outputs
    print('Merging results...')
    
    # Comments
    comments_merged = pd.concat([result0['comments'], result1['comments']], ignore_index=True)
    comments_out_name = comments_path.name.replace('.csv', '_finbert.csv')
    comments_merged.to_csv(output_dir / comments_out_name, index=False)
    print(f'Saved: {comments_out_name}')
    
    # Posts
    posts_merged = pd.concat([result0['posts'], result1['posts']], ignore_index=True)
    posts_out_name = posts_path.name.replace('.csv', '_finbert.csv')
    posts_merged.to_csv(output_dir / posts_out_name, index=False)
    print(f'Saved: {posts_out_name}')
    
    # Daily aggregation
    if not result0['daily'].empty and not result1['daily'].empty:
        daily_merged = pd.concat([result0['daily'], result1['daily']], ignore_index=True)
        daily_agg = daily_merged.groupby('Date')[['positive', 'negative', 'neutral']].mean().reset_index()
        daily_agg.to_csv(output_dir / 'daily_aggregated_finbert.csv', index=False)
        print('Saved: daily_aggregated_finbert.csv')
    
    print(f'\nDone! Results in {output_dir}')


if __name__ == '__main__':
    main()
