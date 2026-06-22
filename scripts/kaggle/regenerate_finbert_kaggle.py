import argparse
import os
from pathlib import Path
import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def get_project_root() -> Path:
    if '__file__' in globals():
        return Path(__file__).resolve().parents[2]

    try:
        import inspect
        return Path(inspect.getsourcefile(lambda: 0)).resolve().parents[2]
    except Exception:
        return Path.cwd()


def find_kaggle_dataset_root() -> Path | None:
    kaggle_input = Path('/kaggle/input')
    if not kaggle_input.exists():
        return None

    # Search recursively for dataset files
    for root in kaggle_input.rglob('.'):
        if not root.is_dir():
            continue
        # Check for either naming convention
        has_reddit_comments = (root / 'reddit_comments.csv').exists()
        has_reddit_posts = (root / 'reddit_posts.csv').exists()
        has_tsla_comments = (root / 'one-year-of-tsla-on-reddit-comments.csv').exists()
        has_tsla_posts = (root / 'one-year-of-tsla-on-reddit-posts.csv').exists()

        if (has_reddit_comments and has_reddit_posts) or (has_tsla_comments and has_tsla_posts):
            return root

    return None


def load_reddit_text(path: Path, text_column: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=['created_utc', text_column])
    df = df.dropna(subset=[text_column])
    df['Date'] = pd.to_datetime(df['created_utc'], unit='s').dt.date
    df['text'] = df[text_column].astype(str)
    return df[['Date', 'text']]


def build_finbert_pipeline(device: int = 0, local_files_only: bool = False):
    model_name = 'ProsusAI/finbert'
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=local_files_only
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        local_files_only=local_files_only
    )
    return pipeline(
        'sentiment-analysis',
        model=model,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
        device=device
    )


def split_text_chunks(text: str, chars_per_chunk: int = 1800, max_chunks: int = 2):
    text = str(text)
    if len(text) == 0:
        return ['']
    chunks = [
        text[i:i + chars_per_chunk]
        for i in range(
            0,
            min(len(text), chars_per_chunk * max_chunks),
            chars_per_chunk
        )
    ]
    return chunks or ['']


def score_from_result(result):
    probabilities = {item['label'].lower(): item['score'] for item in result}
    return probabilities.get('positive', 0) - probabilities.get('negative', 0)


def finbert_scores(texts, finbert_pipeline, batch_size=16):
    scores = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_chunks = [split_text_chunks(text) for text in batch_texts]
        flat_texts = [chunk for chunks in batch_chunks for chunk in chunks]

        results = finbert_pipeline(flat_texts, top_k=None)
        chunk_scores = [score_from_result(result) for result in results]

        idx = 0
        for chunks in batch_chunks:
            text_scores = chunk_scores[idx: idx + len(chunks)]
            idx += len(chunks)
            scores.append(sum(text_scores) / len(text_scores))

    return scores


def compute_and_save(
    comments_path: Path,
    posts_path: Path,
    output_dir: Path,
    device: int,
    batch_size: int,
    local_only: bool
):
    output_dir.mkdir(parents=True, exist_ok=True)

    comments = load_reddit_text(comments_path, 'body')
    posts = load_reddit_text(posts_path, 'title')

    print('Loading FinBERT pipeline on device', device)
    finbert_pipe = build_finbert_pipeline(device=device, local_files_only=local_only)

    print('Scoring comments...')
    comments['FinBERT_Sentiment'] = finbert_scores(
        comments['text'].tolist(),
        finbert_pipe,
        batch_size=batch_size
    )
    comments.to_csv(output_dir / 'one-year-of-tsla-on-reddit-comments_finbert.csv', index=False)

    print('Scoring posts...')
    posts['FinBERT_Sentiment'] = finbert_scores(
        posts['text'].tolist(),
        finbert_pipe,
        batch_size=batch_size
    )
    posts.to_csv(output_dir / 'one-year-of-tsla-on-reddit-posts_finbert.csv', index=False)

    print('Aggregating daily text and scoring daily FinBERT...')
    daily_text = pd.concat([comments[['Date', 'text']], posts[['Date', 'text']]], ignore_index=True)
    daily_text = (
        daily_text.groupby('Date')['text']
        .apply(lambda values: ' '.join(values))
        .reset_index()
    )
    daily_text['FinBERT_Sentiment'] = finbert_scores(
        daily_text['text'].tolist(),
        finbert_pipe,
        batch_size=batch_size
    )
    daily_text.to_csv(output_dir / 'daily_aggregated_finbert.csv', index=False)

    print('Saved outputs to', output_dir)
    print('Comments rows:', len(comments))
    print('Posts rows:', len(posts))
    print('Daily rows:', len(daily_text))


def main():
    parser = argparse.ArgumentParser(
        description='Kaggle-safe FinBERT regeneration for raw Reddit post/comment sentiment'
    )
    parser.add_argument('--device', type=int, default=0, help='GPU device index for transformers pipeline')
    parser.add_argument('--batch-size', type=int, default=16, help='Number of examples to score per pipeline batch')
    parser.add_argument('--local-only', action='store_true', help='Load FinBERT from local cache only')
    parser.add_argument('--output-subdir', default='outputs/eda_storytelling', help='Output directory relative to project root')
    parser.add_argument('--input-root', default=None, help='Optional input root for one-year-of-tsla-on-reddit-comments.csv and one-year-of-tsla-on-reddit-posts.csv')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignoring extra notebook/kernel arguments:', unknown)


    project_root = get_project_root()
    dataset_root = None

    if args.input_root:
        dataset_root = Path(args.input_root)
    else:
        dataset_root = find_kaggle_dataset_root()

    # Accept multiple filename conventions used across datasets
    def _find_files(root: Path):
        comment_names = [
            'reddit_comments.csv',
            'one-year-of-tsla-on-reddit-comments.csv',
            'one-year-of-tsla-on-reddit-comments_1.csv'
        ]
        post_names = [
            'reddit_posts.csv',
            'one-year-of-tsla-on-reddit-posts.csv',
            'one-year-of-tsla-on-reddit-posts_1.csv'
        ]

        for name in comment_names:
            candidate = root / name
            if candidate.exists():
                comment_path = candidate
                break
        else:
            comment_path = None

        for name in post_names:
            candidate = root / name
            if candidate.exists():
                post_path = candidate
                break
        else:
            post_path = None

        return comment_path, post_path

    if dataset_root is not None:
        print('Using dataset root:', dataset_root)
        comments_path, posts_path = _find_files(dataset_root)
    else:
        comments_path, posts_path = _find_files(project_root / 'data' / 'raw')

    # Final fallback to original names under project root if detection failed
    if comments_path is None:
        comments_path = project_root / 'data' / 'raw' / 'reddit_comments.csv'
    if posts_path is None:
        posts_path = project_root / 'data' / 'raw' / 'reddit_posts.csv'

    output_dir = project_root / args.output_subdir

    compute_and_save(
        comments_path,
        posts_path,
        output_dir,
        device=args.device,
        batch_size=args.batch_size,
        local_only=args.local_only
    )


if __name__ == '__main__':
    main()
