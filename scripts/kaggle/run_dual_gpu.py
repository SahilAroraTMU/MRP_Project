import argparse
import subprocess
import shutil
import sys
from pathlib import Path
import time


def get_project_root() -> Path:
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


def split_comments_block(comments_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # count lines (including header)
    with comments_path.open('r', encoding='utf-8') as f:
        total = sum(1 for _ in f)
    if total <= 1:
        raise RuntimeError('No comment rows found')
    header_lines = 1
    data_lines = total - header_lines
    first_count = data_lines // 2

    # Use same filename for output shards
    shard0 = out_dir / comments_path.name
    shard1 = out_dir / (comments_path.stem + '_2' + comments_path.suffix)

    with comments_path.open('r', encoding='utf-8') as fin, \
         shard0.open('w', encoding='utf-8') as f0, \
         shard1.open('w', encoding='utf-8') as f1:
        header = fin.readline()
        f0.write(header)
        f1.write(header)

        for i, line in enumerate(fin):
            if i < first_count:
                f0.write(line)
            else:
                f1.write(line)

    return shard0, shard1


def prepare_shard_dirs(comments_shard0: Path, comments_shard1: Path, posts_path: Path, base_temp: Path):
    shard_dirs = []
    for idx, comments_file in enumerate([comments_shard0, comments_shard1]):
        d = base_temp / f'shard{idx}'
        d.mkdir(parents=True, exist_ok=True)
        # copy actual comments file with original name
        shutil.copy(comments_file, d / comments_file.name)
        # copy posts with original name
        shutil.copy(posts_path, d / posts_path.name)
        shard_dirs.append(d)
    return shard_dirs


def run_shard_process(script_path: Path, shard_dir: Path, device: int, batch_size: int, output_subdir: Path, local_only: bool):
    # Ensure script path exists and is absolute
    script_path = script_path.resolve()
    if not script_path.exists():
        raise FileNotFoundError(f'Script not found: {script_path}')
    
    cmd = [
        sys.executable,
        str(script_path),
        '--device', str(device),
        '--batch-size', str(batch_size),
        '--input-root', str(shard_dir),
        '--output-subdir', str(output_subdir)
    ]
    if local_only:
        cmd.append('--local-only')

    print(f'Running subprocess: {" ".join(cmd)}')
    return subprocess.Popen(cmd)


def merge_outputs(project_root: Path, out_dir: Path, out_subdir: str):
    import pandas as pd
    import glob

    # Find output files dynamically (they may be named based on input filename)
    shard0_dir = project_root / out_subdir / 'shard0'
    shard1_dir = project_root / out_subdir / 'shard1'

    # Find comments finbert files
    comments_files = list(shard0_dir.glob('*_finbert.csv'))
    if comments_files:
        # Filter for comments (not posts or daily)
        comments_files = [f for f in comments_files if 'daily' not in f.name and 'posts' not in f.name]
    
    if comments_files:
        comments_base_name = comments_files[0].name  # Get actual filename
        s0_comments = shard0_dir / comments_base_name
        s1_comments = shard1_dir / comments_base_name
        
        if s0_comments.exists() and s1_comments.exists():
            df0 = pd.read_csv(s0_comments)
            df1 = pd.read_csv(s1_comments)
            combined = pd.concat([df0, df1], ignore_index=True)
            combined.to_csv(out_dir / comments_base_name, index=False)
            print(f'Merged comments: {comments_base_name}')

    # Find posts finbert files
    posts_files = list(shard0_dir.glob('*posts*_finbert.csv'))
    if posts_files:
        posts_base_name = posts_files[0].name
        s0_posts = shard0_dir / posts_base_name
        s1_posts = shard1_dir / posts_base_name
        
        if s0_posts.exists() and s1_posts.exists():
            df0 = pd.read_csv(s0_posts)
            df1 = pd.read_csv(s1_posts)
            combined = pd.concat([df0, df1], ignore_index=True)
            combined.to_csv(out_dir / posts_base_name, index=False)
            print(f'Merged posts: {posts_base_name}')

    # Aggregate daily
    daily_files = list(shard0_dir.glob('daily*.csv'))
    if daily_files:
        daily_base_name = daily_files[0].name
        d0 = shard0_dir / daily_base_name
        d1 = shard1_dir / daily_base_name
        
        if d0.exists() and d1.exists():
            df0 = pd.read_csv(d0)
            df1 = pd.read_csv(d1)
            combined = pd.concat([df0, df1], ignore_index=True)
            # aggregate by Date by taking mean
            combined = combined.groupby('Date').mean().reset_index()
            combined.to_csv(out_dir / daily_base_name, index=False)
            print(f'Aggregated daily: {daily_base_name}')


def main():
    parser = argparse.ArgumentParser(description='Run regenerate_finbert_kaggle.py on two GPUs by sharding comments')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--input-root', default=None, help='Optional dataset root containing reddit_comments.csv and reddit_posts.csv')
    parser.add_argument('--output-subdir', default='outputs/eda_storytelling', help='Output subdir under project root')
    parser.add_argument('--local-only', action='store_true')
    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignoring extra notebook/kernel arguments:', unknown)

    project_root = get_project_root()
    script_path = project_root / 'scripts' / 'kaggle' / 'regenerate_finbert_kaggle.py'

    if args.input_root:
        dataset_root = Path(args.input_root)
    else:
        kag_input = Path('/kaggle/input')
        dataset_root = None
        # Search recursively for dataset files in /kaggle/input
        if kag_input.exists():
            for root in kag_input.rglob('.'):
                if not root.is_dir():
                    continue
                has_reddit_comments = (root / 'reddit_comments.csv').exists()
                has_reddit_posts = (root / 'reddit_posts.csv').exists()
                has_tsla_comments = (root / 'one-year-of-tsla-on-reddit-comments.csv').exists()
                has_tsla_posts = (root / 'one-year-of-tsla-on-reddit-posts.csv').exists()

                if (has_reddit_comments and has_reddit_posts) or (has_tsla_comments and has_tsla_posts):
                    dataset_root = root
                    break
        
        if dataset_root is None:
            dataset_root = project_root / 'data' / 'raw'

    # find exact file names (support multiple naming conventions)
    def _find_files(root: Path):
        comment_names = ['reddit_comments.csv', 'one-year-of-tsla-on-reddit-comments.csv']
        post_names = ['reddit_posts.csv', 'one-year-of-tsla-on-reddit-posts.csv']
        comment_file = None
        post_file = None
        for name in comment_names:
            p = root / name
            if p.exists():
                comment_file = p
                break
        for name in post_names:
            p = root / name
            if p.exists():
                post_file = p
                break
        return comment_file, post_file

    comments_path, posts_path = _find_files(dataset_root)
    if comments_path is None:
        comments_path = dataset_root / 'reddit_comments.csv'
    if posts_path is None:
        posts_path = dataset_root / 'reddit_posts.csv'

    temp = project_root / 'tmp' / f'finbert_shards_{int(time.time())}'
    temp.mkdir(parents=True, exist_ok=True)

    print('Counting and splitting comments...')
    shard0_file, shard1_file = split_comments_block(comments_path, temp)

    print('Preparing shard directories...')
    shard_dirs = prepare_shard_dirs(shard0_file, shard1_file, posts_path, temp)

    out_base = project_root / args.output_subdir
    out_base.mkdir(parents=True, exist_ok=True)

    shard0_out = out_base / 'shard0'
    shard1_out = out_base / 'shard1'
    shard0_out.mkdir(parents=True, exist_ok=True)
    shard1_out.mkdir(parents=True, exist_ok=True)

    print('Starting shard processes...')
    p0 = run_shard_process(script_path, shard_dirs[0], device=0, batch_size=args.batch_size, output_subdir=str(args.output_subdir + '/shard0'), local_only=args.local_only)
    p1 = run_shard_process(script_path, shard_dirs[1], device=1, batch_size=args.batch_size, output_subdir=str(args.output_subdir + '/shard1'), local_only=args.local_only)

    print('Waiting for processes to finish...')
    ret0 = p0.wait()
    ret1 = p1.wait()

    print('Processes finished', ret0, ret1)

    print('Merging outputs...')
    merge_outputs(project_root, out_base, args.output_subdir)

    print('Cleaning temporary files...')
    try:
        shutil.rmtree(temp)
    except Exception:
        pass

    print('Done. Combined outputs in', out_base)


if __name__ == '__main__':
    main()
