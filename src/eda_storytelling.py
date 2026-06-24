import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sentiment_analysis import apply_fast_sentiment

OUTPUT_DIR = Path('outputs/eda_storytelling')
RAW_POSTS_PATH = Path('data/raw/reddit_posts.csv')
PREPROCESSED_POSTS_PATH = Path('data/processed/preprocessed_data_1.xlsx')
PREPROCESSED_DAILY_PATH = Path('data/processed/preprocessed_data_3.csv')
FINAL_MERGED_PATH = Path('outputs/results/final_merged_dataset.csv')
DAILY_FINBERT_SENTIMENT_CANDIDATES = [
    OUTPUT_DIR / 'daily_finbert_sentiment.csv',
    OUTPUT_DIR / 'daily_aggregated_finbert.csv',
]


def _resolve_daily_finbert_path() -> Path:
    for path in DAILY_FINBERT_SENTIMENT_CANDIDATES:
        if path.exists():
            return path
    return DAILY_FINBERT_SENTIMENT_CANDIDATES[0]


def _normalize_text(value):
    value = '' if pd.isna(value) else str(value).lower()
    value = np.str_(value)
    value = value.replace('http', ' http')
    value = ''.join(ch if ch.isalnum() or ch.isspace() else ' ' for ch in value)
    return ' '.join(value.split())


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _load_raw_posts() -> pd.DataFrame:
    df = pd.read_csv(RAW_POSTS_PATH)
    if 'created_utc' in df.columns:
        df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    df['normalized_post_title'] = df['title'].fillna('').astype(str).apply(_normalize_text)
    return df


def _load_deduped_posts() -> pd.DataFrame:
    df = pd.read_excel(PREPROCESSED_POSTS_PATH, engine='openpyxl')
    return df


def _load_preprocessed_daily() -> pd.DataFrame:
    df = pd.read_csv(PREPROCESSED_DAILY_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df.sort_values('Date')


def _load_daily_finbert_sentiment() -> pd.DataFrame:
    df = pd.read_csv(_resolve_daily_finbert_path())
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date'] = df['Date'].dt.normalize()
    if {'Avg_Comment_Sentiment', 'Comment_Count', 'Avg_Post_Sentiment', 'Post_Count'}.issubset(df.columns):
        df['Comment_Count'] = pd.to_numeric(df['Comment_Count'], errors='coerce').fillna(0)
        df['Post_Count'] = pd.to_numeric(df['Post_Count'], errors='coerce').fillna(0)
        df['Avg_Comment_Sentiment'] = pd.to_numeric(df['Avg_Comment_Sentiment'], errors='coerce').fillna(0)
        df['Avg_Post_Sentiment'] = pd.to_numeric(df['Avg_Post_Sentiment'], errors='coerce').fillna(0)
        grouped = (
            df.assign(
                total_score=(
                    df['Avg_Comment_Sentiment'] * df['Comment_Count'] +
                    df['Avg_Post_Sentiment'] * df['Post_Count']
                ),
                total_count=df['Comment_Count'] + df['Post_Count'],
            )
            .groupby('Date', as_index=False)[['total_score', 'total_count']]
            .sum()
        )
        grouped['FinBERT_Sentiment'] = grouped['total_score'] / grouped['total_count'].replace(0, np.nan)
        return grouped.sort_values('Date')

    df = df.rename(columns={'Avg_FinBERT_Sentiment': 'FinBERT_Sentiment'})
    return (
        df[['Date', 'FinBERT_Sentiment']]
        .dropna(subset=['Date'])
        .groupby('Date', as_index=False)['FinBERT_Sentiment']
        .mean()
        .sort_values('Date')
    )


def _load_final_merged() -> pd.DataFrame:
    df = pd.read_csv(FINAL_MERGED_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Date'] = df['Date'].dt.normalize()
    return df.sort_values('Date')


def _load_raw_combined_sentiment() -> pd.DataFrame:
    posts = pd.read_csv(
        RAW_POSTS_PATH,
        usecols=['created_utc', 'title', 'selftext'],
        low_memory=False
    )
    comments = pd.read_csv(
        'data/raw/reddit_comments.csv',
        usecols=['created_utc', 'body'],
        low_memory=False
    )

    for df in [posts, comments]:
        if 'created_utc' in df.columns:
            df['Date'] = pd.to_datetime(
                df['created_utc'],
                unit='s',
                errors='coerce'
            )
        else:
            df['Date'] = pd.NaT

    posts['text'] = (
        posts['title'].fillna('').astype(str) + ' ' +
        posts.get('selftext', pd.Series('', index=posts.index)).fillna('').astype(str)
    )
    comments['text'] = comments['body'].fillna('').astype(str)

    posts['source'] = 'post'
    posts['row_id'] = posts.index
    comments['source'] = 'comment'
    comments['row_id'] = comments.index
    combined = pd.concat(
        [
            posts[['Date', 'text', 'source', 'row_id']],
            comments[['Date', 'text', 'source', 'row_id']]
        ],
        ignore_index=True
    )

    combined = combined.dropna(subset=['text']).copy()
    combined = apply_fast_sentiment(combined)
    combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce')
    return combined.sort_values('Date')


def _attach_daily_finbert(df: pd.DataFrame, daily_finbert_df: pd.DataFrame) -> pd.DataFrame:
    merged = df.copy()
    merged['Date'] = pd.to_datetime(merged['Date'], errors='coerce')
    merged['Daily_Date'] = merged['Date'].dt.normalize()
    finbert_daily = daily_finbert_df[['Date', 'FinBERT_Sentiment']].rename(columns={'Date': 'Daily_Date'})
    merged = merged.merge(finbert_daily, on='Daily_Date', how='left')
    return merged


def _prepare_daily_sentiment_panel(raw_df: pd.DataFrame, daily_finbert_df: pd.DataFrame) -> pd.DataFrame:
    sentiment_cols = {
        'VADER': 'VADER_Sentiment',
        'TextBlob': 'TextBlob_Sentiment',
    }
    daily_raw = (
        raw_df.assign(Date=pd.to_datetime(raw_df['Date'], errors='coerce').dt.normalize())
        [['Date'] + list(sentiment_cols.values())]
        .dropna(subset=['Date'])
        .groupby('Date', as_index=False)
        .mean(numeric_only=True)
        .sort_values('Date')
    )

    finbert_daily = daily_finbert_df[['Date', 'FinBERT_Sentiment']].copy()
    finbert_daily['Date'] = pd.to_datetime(finbert_daily['Date'], errors='coerce').dt.normalize()
    finbert_daily = finbert_daily.dropna(subset=['Date']).sort_values('Date')

    panel_df = daily_raw.merge(finbert_daily, on='Date', how='outer').sort_values('Date')
    return panel_df


def _plot_sentiment_market_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    sentiment_col: str,
    sentiment_label: str,
    market_col: str,
    market_label: str,
    sentiment_color: str,
    market_color: str,
    title: str,
    metric_name: str,
) -> None:
    panel = df[['Date', market_col, sentiment_col]].copy()
    panel[market_col] = pd.to_numeric(panel[market_col], errors='coerce')
    panel[sentiment_col] = pd.to_numeric(panel[sentiment_col], errors='coerce')
    panel = panel.dropna(subset=['Date', market_col, sentiment_col]).sort_values('Date')

    ax2 = ax.twinx()
    if panel.empty:
        ax.text(0.5, 0.5, 'No overlapping data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel(market_label, color=market_color)
        ax2.set_ylabel(sentiment_label, color=sentiment_color)
        return

    market_threshold = panel[market_col].quantile(0.9)
    sentiment_threshold = panel[sentiment_col].quantile(0.9)

    market_line, = ax.plot(panel['Date'], panel[market_col], color=market_color, linewidth=1.25, label=market_label)
    market_spikes = ax.scatter(
        panel.loc[panel[market_col] >= market_threshold, 'Date'],
        panel.loc[panel[market_col] >= market_threshold, market_col],
        s=26,
        marker='x',
        color='#f59e0b',
        linewidth=1.0,
        label=f'{metric_name} Spike',
        zorder=4,
    )
    ax.axhline(market_threshold, color='#f59e0b', linestyle='--', linewidth=0.9, alpha=0.65, label='_nolegend_')
    ax.set_xlabel('Date')
    ax.set_ylabel(market_label, color=market_color)
    ax.tick_params(axis='y', labelcolor=market_color)
    ax.grid(alpha=0.25)

    sentiment_line, = ax2.plot(panel['Date'], panel[sentiment_col], color=sentiment_color, linewidth=1.25, label=sentiment_label)
    sentiment_spikes = ax2.scatter(
        panel.loc[panel[sentiment_col] >= sentiment_threshold, 'Date'],
        panel.loc[panel[sentiment_col] >= sentiment_threshold, sentiment_col],
        s=18,
        color='black',
        alpha=0.85,
        label='Sentiment Spike',
        zorder=5,
    )
    ax2.axhline(sentiment_threshold, color='black', linestyle=':', linewidth=0.9, alpha=0.5, label='_nolegend_')
    ax2.set_ylabel(sentiment_label, color=sentiment_color)
    ax2.tick_params(axis='y', labelcolor=sentiment_color)
    ax.set_title(title)
    ax.legend(
        [market_line, market_spikes, sentiment_line, sentiment_spikes],
        [market_label, f'{metric_name} Spike', sentiment_label, 'Sentiment Spike'],
        loc='upper left',
        fontsize=8,
    )


def plot_duplicate_repost_impact(raw_posts: pd.DataFrame, deduped_posts: pd.DataFrame) -> None:
    raw_count = len(raw_posts)
    original_count = len(deduped_posts)
    repost_count = int(pd.to_numeric(deduped_posts['repost_count'], errors='coerce').fillna(0).sum())

    same_author_duplicates = int(
        (pd.to_numeric(deduped_posts['repost_count'], errors='coerce').fillna(0) + 1 -
         pd.to_numeric(deduped_posts['unique_author_count'], errors='coerce').fillna(0))
        .clip(lower=0)
        .sum()
    )

    categories = ['Original posts', 'Reposts', 'Same-author duplicates']
    counts = [original_count, repost_count, same_author_duplicates]
    percentages = [count / raw_count * 100 if raw_count else 0 for count in counts]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, counts, color=['#2563eb', '#f59e0b', '#dc2626'], edgecolor='black')

    ax.set_title('Impact of Duplicate and Repost Processing')
    ax.set_xlabel('Post category')
    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.25)

    for bar, count, pct in zip(bars, counts, percentages):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(count, 1) * 0.02,
            f'{count:,}\n({pct:.1f}%)',
            ha='center',
            va='bottom',
            fontsize=10,
        )

    caption = (
        'Duplicate detection preserves original posts while summarizing repeats. ' 
        'Reposts are counted from repeated titles and same-author duplicates capture repeated title-author pairs.'
    )
    fig.text(0.5, -0.08, caption, ha='center', fontsize=9)
    _save_figure(fig, OUTPUT_DIR / 'duplicate_repost_impact.png')

    summary = pd.DataFrame({
        'Category': categories,
        'Count': counts,
        'Percentage': [round(p, 2) for p in percentages],
    })
    _save_csv(summary, OUTPUT_DIR / 'duplicate_repost_summary.csv')


def plot_top20_contributors(deduped_posts: pd.DataFrame) -> None:
    top_posts = deduped_posts.copy()
    top_posts['engagement_score'] = pd.to_numeric(top_posts['engagement_score'], errors='coerce').fillna(0.0)
    top_posts['title'] = top_posts.get('title', pd.Series('', index=top_posts.index)).fillna('').astype(str).str.strip()
    top_posts['author_key'] = top_posts.get('author_key', pd.Series('', index=top_posts.index)).fillna('').astype(str).str.strip()
    top_posts = top_posts[top_posts['engagement_score'].notna()].copy()
    top_posts = top_posts.sort_values('engagement_score', ascending=False).head(20).reset_index(drop=True)
    top_posts['Display_Title'] = top_posts['title'].replace('', pd.NA)
    top_posts['Display_Title'] = top_posts['Display_Title'].fillna(top_posts['author_key'])
    top_posts['Display_Title'] = top_posts['Display_Title'].fillna('Untitled post').astype(str)
    top_posts['Display_Title'] = top_posts['Display_Title'].apply(
        lambda value: value if len(value) <= 72 else f'{value[:69].rstrip()}...'
    )
    top_posts['Display_Label'] = [
        f'{rank}. {title}' for rank, title in zip(range(1, len(top_posts) + 1), top_posts['Display_Title'])
    ]
    total_engagement = pd.to_numeric(deduped_posts.get('engagement_score', pd.Series(dtype=float)), errors='coerce').fillna(0.0).sum()

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.subplots_adjust(left=0.38, right=0.95)
    bars = ax.barh(top_posts['Display_Label'][::-1], top_posts['engagement_score'][::-1], color='#2563eb', edgecolor='black')
    ax.set_title('Top 20 Posts by Engagement Score')
    ax.set_xlabel('Engagement score')
    ax.set_ylabel('Post')
    ax.grid(axis='x', alpha=0.25)
    ax.set_xlim(0, top_posts['engagement_score'].max() * 1.12 if not top_posts.empty else 1)

    for bar, score in zip(bars, top_posts['engagement_score'][::-1]):
        ax.text(
            bar.get_width() + max(top_posts['engagement_score'].max() * 0.008, 0.01),
            bar.get_y() + bar.get_height() / 2,
            f'{score:.3f}',
            va='center',
            fontsize=9,
            color='#111827'
        )

    caption = (
        'The chart now ranks the 20 highest-engagement posts directly, using the post title as the label instead of rolling up to authors.'
    )
    fig.text(0.5, -0.08, caption, ha='center', fontsize=9)
    _save_figure(fig, OUTPUT_DIR / 'top20_contributors.png')

    summary = top_posts[['Display_Title', 'author_key', 'engagement_score']].copy()
    summary.insert(0, 'Rank', range(1, len(summary) + 1))
    summary = summary.rename(columns={
        'Display_Title': 'Post_Title',
        'author_key': 'Author',
        'engagement_score': 'Engagement_Score',
    })
    summary['Engagement_Share_pct'] = [round(val / total_engagement * 100, 2) if total_engagement else 0.0 for val in summary['Engagement_Score']]
    summary['Cumulative_Engagement_Share_pct'] = [
        round(summary['Engagement_Score'].iloc[:i + 1].sum() / total_engagement * 100, 2)
        if total_engagement else 0.0
        for i in range(len(summary))
    ]
    _save_csv(summary, OUTPUT_DIR / 'top20_contributors.csv')


def plot_sentiment_model_comparison(raw_df: pd.DataFrame, daily_finbert_df: pd.DataFrame) -> None:
    sentiment_cols = {
        'VADER': 'VADER_Sentiment',
        'TextBlob': 'TextBlob_Sentiment',
    }
    stats = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    bins = np.linspace(-1, 1, 35)

    for ax, (label, col) in zip(axes[:2], sentiment_cols.items()):
        values = pd.to_numeric(raw_df[col], errors='coerce').dropna()
        ax.hist(values, bins=bins, color='#2563eb', edgecolor='black', alpha=0.75)
        ax.set_title(f'{label} Distribution')
        ax.set_xlabel('Sentiment score')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.25)

        stats.append({
            'Model': label,
            'Mean': round(values.mean(), 4),
            'Median': round(values.median(), 4),
            'StdDev': round(values.std(ddof=0), 4),
            'Negative_Share_pct': round((values < 0).mean() * 100, 2),
        })

    finbert_values = pd.to_numeric(daily_finbert_df.get('FinBERT_Sentiment', pd.Series()), errors='coerce').dropna()
    if finbert_values.empty:
        finbert_bins = bins
        finbert_xlim = (-1, 1)
    else:
        finbert_min = float(finbert_values.min())
        finbert_max = float(finbert_values.max())
        finbert_span = max(finbert_max - finbert_min, 0.02)
        finbert_padding = finbert_span * 0.15
        finbert_low = finbert_min - finbert_padding
        finbert_high = finbert_max + finbert_padding
        finbert_bins = np.linspace(finbert_low, finbert_high, 25)
        finbert_xlim = (finbert_low, finbert_high)

    axes[2].hist(finbert_values, bins=finbert_bins, color='#dc2626', edgecolor='black', alpha=0.8)
    axes[2].set_title('FinBERT Distribution')
    axes[2].set_xlabel('Sentiment score')
    axes[2].set_ylabel('Frequency')
    axes[2].set_xlim(*finbert_xlim)
    axes[2].grid(alpha=0.25)

    stats.append({
        'Model': 'FinBERT',
        'Mean': round(finbert_values.mean(), 4),
        'Median': round(finbert_values.median(), 4),
        'StdDev': round(finbert_values.std(ddof=0), 4),
        'Negative_Share_pct': round((finbert_values < 0).mean() * 100, 2),
    })

    caption = (
        'VADER and TextBlob are from raw post/comment data; FinBERT is taken from the dedicated daily FinBERT aggregate.'
    )
    fig.text(0.5, -0.05, caption, ha='center', fontsize=9)
    _save_figure(fig, OUTPUT_DIR / 'sentiment_model_comparison.png')

    stats_df = pd.DataFrame(stats)
    _save_csv(stats_df, OUTPUT_DIR / 'sentiment_statistics.csv')


def plot_finbert_distribution(daily_finbert_df: pd.DataFrame) -> None:
    """Plot FinBERT distribution from the dedicated daily FinBERT aggregate."""
    vals = pd.to_numeric(daily_finbert_df.get('FinBERT_Sentiment', pd.Series()), errors='coerce').dropna()

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-1, 1, 35)

    if vals.empty:
        ax.text(0.5, 0.5, 'No FinBERT values available', ha='center', va='center')
        ax.set_title('FinBERT Distribution (daily aggregate)')
        ax.set_xlabel('Sentiment score')
        ax.set_ylabel('Frequency')
    else:
        ax.hist(vals, bins=bins, color='#dc2626', edgecolor='black', alpha=0.75)
        ax.set_title('FinBERT Distribution (daily aggregate)')
        ax.set_xlabel('Sentiment score')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.25)

    caption = 'FinBERT distribution taken from the dedicated daily FinBERT aggregate.'
    fig.text(0.5, -0.05, caption, ha='center', fontsize=9)
    _save_figure(fig, OUTPUT_DIR / 'finbert_daily_distribution.png')

    stats = {
        'Count': int(vals.count()),
        'Mean': float(vals.mean()) if not vals.empty else None,
        'StdDev': float(vals.std(ddof=0)) if not vals.empty else None,
        'Neg_Share_pct': round((vals < 0).mean() * 100, 2) if not vals.empty else None,
    }
    _save_csv(pd.DataFrame([stats]), OUTPUT_DIR / 'finbert_daily_statistics.csv')


def plot_raw_sentiment_trends(raw_df: pd.DataFrame, daily_finbert_df: pd.DataFrame) -> None:
    raw_with_daily_finbert = _attach_daily_finbert(raw_df, daily_finbert_df)
    sentiment_cols = {
        'VADER': 'VADER_Sentiment',
        'TextBlob': 'TextBlob_Sentiment',
    }
    daily_raw = (
        raw_df.assign(Date=pd.to_datetime(raw_df['Date'], errors='coerce').dt.normalize())
        [['Date'] + list(sentiment_cols.values())]
        .dropna(subset=['Date'])
        .groupby('Date', as_index=False)
        .mean()
        .sort_values('Date')
    )
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    colors = ['#2563eb', '#16a34a', '#dc2626']

    for ax, (label, col), color in zip(axes[:2], sentiment_cols.items(), colors[:2]):
        df = daily_raw[['Date', col]].copy().dropna().sort_values('Date')
        ax.plot(df['Date'], df[col], color=color, linewidth=1.8)
        ax.scatter(df['Date'], df[col], s=12, alpha=0.45, color=color)
        ax.set_title(f'Daily {label} Sentiment Over Time')
        ax.set_ylabel('Sentiment score')
        ax.grid(alpha=0.25)

    finbert_daily = daily_finbert_df[['Date', 'FinBERT_Sentiment']].copy().dropna().sort_values('Date')
    axes[2].plot(finbert_daily['Date'], finbert_daily['FinBERT_Sentiment'], color=colors[2], linewidth=1.8)
    axes[2].scatter(finbert_daily['Date'], finbert_daily['FinBERT_Sentiment'], s=10, alpha=0.45, color=colors[2])
    axes[2].set_title('Daily FinBERT Sentiment Over Time')
    axes[2].set_ylabel('Sentiment score')
    axes[2].grid(alpha=0.25)

    axes[-1].set_xlabel('Date')
    caption = (
        'VADER, TextBlob, and FinBERT are shown as daily sentiment series so the three models can be compared on the same time scale.'
    )
    fig.text(0.5, -0.03, caption, ha='center', fontsize=9)
    _save_figure(fig, OUTPUT_DIR / 'raw_sentiment_trends.png')

    export_df = daily_raw.merge(
        finbert_daily.rename(columns={'FinBERT_Sentiment': 'FinBERT_Sentiment'}),
        on='Date',
        how='left'
    )
    _save_csv(export_df, OUTPUT_DIR / 'raw_sentiment_values.csv')


def plot_daily_sentiment_trends(raw_df: pd.DataFrame, daily_finbert_df: pd.DataFrame) -> None:
    sentiment_cols = {
        'VADER': 'VADER_Sentiment',
        'TextBlob': 'TextBlob_Sentiment',
    }
    grouped = (
        raw_df.assign(Date=raw_df['Date'].dt.normalize())[['Date'] + list(sentiment_cols.values())]
        .dropna(subset=['Date'])
        .groupby('Date')
        .mean()
        .reset_index()
    )

    finbert_daily = daily_finbert_df.rename(columns={'FinBERT_Sentiment': 'FinBERT'})
    grouped = grouped.merge(finbert_daily, on='Date', how='left')

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2563eb', '#16a34a', '#dc2626']
    for color, (label, col) in zip(colors, list(sentiment_cols.items()) + [('FinBERT', 'FinBERT')]):
        ax.plot(grouped['Date'], grouped[col], label=label, color=color, linewidth=1.8)

    ax.set_title('Daily Sentiment Trends Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Average sentiment score')
    ax.grid(alpha=0.25)
    ax.legend()

    caption = (
        'Daily averages are computed from the raw sentiment source for VADER and TextBlob, while FinBERT is read from the dedicated daily FinBERT aggregate.'
    )
    fig.text(0.5, -0.08, caption, ha='center', fontsize=9)
    _save_figure(fig, OUTPUT_DIR / 'daily_sentiment_trends.png')

    export_df = grouped.rename(columns={
        'VADER_Sentiment': 'VADER',
        'TextBlob_Sentiment': 'TextBlob',
    })
    export_df = export_df[['Date', 'VADER', 'TextBlob', 'FinBERT']]
    _save_csv(export_df, OUTPUT_DIR / 'daily_sentiment.csv')


def plot_sentiment_vs_volume(raw_df: pd.DataFrame, daily_finbert_df: pd.DataFrame, final_df: pd.DataFrame) -> None:
    panel_df = _prepare_daily_sentiment_panel(raw_df=raw_df, daily_finbert_df=daily_finbert_df)
    volume_df = final_df[['Date', 'Volume']].copy()
    volume_df['Date'] = pd.to_datetime(volume_df['Date'], errors='coerce').dt.normalize()
    volume_df['Volume'] = pd.to_numeric(volume_df['Volume'], errors='coerce')
    panel_df = panel_df.merge(volume_df, on='Date', how='inner')

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    fig.subplots_adjust(hspace=0.28)
    panel_specs = [
        ('VADER_Sentiment', 'VADER Sentiment', '#1d4ed8'),
        ('TextBlob_Sentiment', 'TextBlob Sentiment', '#15803d'),
        ('FinBERT_Sentiment', 'FinBERT Sentiment', '#dc2626'),
    ]
    for ax, (sentiment_col, sentiment_label, sentiment_color) in zip(axes, panel_specs):
        _plot_sentiment_market_panel(
            ax=ax,
            df=panel_df,
            sentiment_col=sentiment_col,
            sentiment_label=sentiment_label,
            market_col='Volume',
            market_label='Trading Volume',
            sentiment_color=sentiment_color,
            market_color='#2563eb',
            title=f'{sentiment_label} vs Trading Volume',
            metric_name='Volume',
        )

    caption = (
        'Each panel compares trading volume with one sentiment scoring method so the three sentiment series can be read side by side.'
    )
    fig.text(0.5, -0.05, caption, ha='center', fontsize=9)
    _save_figure(fig, OUTPUT_DIR / 'sentiment_vs_volume.png')

    export_df = panel_df[['Date', 'Volume', 'VADER_Sentiment', 'TextBlob_Sentiment', 'FinBERT_Sentiment']].copy()
    export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')
    _save_csv(export_df, OUTPUT_DIR / 'finbert_sentiment_vs_volume_line.csv')


def plot_sentiment_vs_illiquidity(raw_df: pd.DataFrame, daily_finbert_df: pd.DataFrame, final_df: pd.DataFrame) -> None:
    panel_df = _prepare_daily_sentiment_panel(raw_df=raw_df, daily_finbert_df=daily_finbert_df)
    illiquidity_df = final_df[['Date', 'Illiquidity']].copy()
    illiquidity_df['Date'] = pd.to_datetime(illiquidity_df['Date'], errors='coerce').dt.normalize()
    illiquidity_df['Illiquidity'] = pd.to_numeric(illiquidity_df['Illiquidity'], errors='coerce')
    panel_df = panel_df.merge(illiquidity_df, on='Date', how='inner')

    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    fig.subplots_adjust(hspace=0.28)
    panel_specs = [
        ('VADER_Sentiment', 'VADER Sentiment', '#1d4ed8'),
        ('TextBlob_Sentiment', 'TextBlob Sentiment', '#15803d'),
        ('FinBERT_Sentiment', 'FinBERT Sentiment', '#dc2626'),
    ]
    for ax, (sentiment_col, sentiment_label, sentiment_color) in zip(axes, panel_specs):
        _plot_sentiment_market_panel(
            ax=ax,
            df=panel_df,
            sentiment_col=sentiment_col,
            sentiment_label=sentiment_label,
            market_col='Illiquidity',
            market_label='Illiquidity',
            sentiment_color=sentiment_color,
            market_color='#16a34a',
            title=f'{sentiment_label} vs Illiquidity',
            metric_name='Illiquidity',
        )

    caption = (
        'Each panel compares illiquidity with one sentiment scoring method so the three sentiment series can be read side by side.'
    )
    fig.text(0.5, -0.05, caption, ha='center', fontsize=9)
    _save_figure(fig, OUTPUT_DIR / 'sentiment_vs_illiquidity.png')

    export_df = panel_df[['Date', 'Illiquidity', 'VADER_Sentiment', 'TextBlob_Sentiment', 'FinBERT_Sentiment']].copy()
    export_df['Date'] = export_df['Date'].dt.strftime('%Y-%m-%d')
    _save_csv(export_df, OUTPUT_DIR / 'finbert_sentiment_vs_illiquidity_line.csv')


def plot_information_diffusion(daily_df: pd.DataFrame) -> None:
    df = daily_df[['repost_count', 'unique_author_count']].copy()
    df['repost_count'] = pd.to_numeric(df['repost_count'], errors='coerce')
    df['unique_author_count'] = pd.to_numeric(df['unique_author_count'], errors='coerce')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(df['repost_count'].dropna(), bins=20, color='#2563eb', edgecolor='black', alpha=0.75)
    axes[0].set_title('Repost Count Distribution')
    axes[0].set_xlabel('Repost count (daily aggregated)')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(alpha=0.25)

    axes[1].hist(df['unique_author_count'].dropna(), bins=20, color='#16a34a', edgecolor='black', alpha=0.75)
    axes[1].set_title('Unique Author Count Distribution')
    axes[1].set_xlabel('Unique author count (daily aggregated)')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(alpha=0.25)

    caption = (
        'This panel exposes how widely information is shared through reposts and distinct authors on Reddit.'
    )
    fig.text(0.5, -0.08, caption, ha='center', fontsize=9)
    _save_figure(fig, OUTPUT_DIR / 'information_diffusion.png')

    stats = [
        {
            'Metric': 'Repost_Count',
            'Mean': round(df['repost_count'].mean(), 4),
            'Median': round(df['repost_count'].median(), 4),
            'StdDev': round(df['repost_count'].std(ddof=0), 4),
        },
        {
            'Metric': 'Unique_Author_Count',
            'Mean': round(df['unique_author_count'].mean(), 4),
            'Median': round(df['unique_author_count'].median(), 4),
            'StdDev': round(df['unique_author_count'].std(ddof=0), 4),
        },
    ]
    _save_csv(pd.DataFrame(stats), OUTPUT_DIR / 'information_diffusion_statistics.csv')


def plot_lag_analysis(raw_df: pd.DataFrame, final_df: pd.DataFrame, daily_finbert_df: pd.DataFrame) -> None:
    sentiment_sources = [
        ('VADER_Sentiment', 'VADER Sentiment', '#1d4ed8'),
        ('TextBlob_Sentiment', 'TextBlob Sentiment', '#15803d'),
        ('FinBERT_Sentiment', 'FinBERT Sentiment', '#dc2626'),
    ]

    df = final_df[['Date', 'Illiquidity']].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
    df['Illiquidity'] = pd.to_numeric(df['Illiquidity'], errors='coerce')
    df = df.sort_values('Date')

    daily_sentiment = (
        _prepare_daily_sentiment_panel(raw_df=raw_df, daily_finbert_df=daily_finbert_df)
        [['Date'] + [col for col, _, _ in sentiment_sources]]
        .sort_values('Date')
    )
    df = df.merge(daily_sentiment, on='Date', how='inner')

    def pearson(x, y):
        if len(x) < 2:
            return np.nan
        return np.corrcoef(x, y)[0, 1]

    fig, axes = plt.subplots(len(sentiment_sources), 2, figsize=(14, 16))
    fig.subplots_adjust(hspace=0.35, wspace=0.22)

    stats = []
    for row, (sentiment_col, sentiment_label, color) in enumerate(sentiment_sources):
        row_df = df[['Date', 'Illiquidity', sentiment_col]].copy().sort_values('Date')
        row_df[f'{sentiment_col}_Lag_1'] = row_df[sentiment_col].shift(1)
        row_df[f'{sentiment_col}_Lag_2'] = row_df[sentiment_col].shift(2)
        lag1_df = row_df[[f'{sentiment_col}_Lag_1', 'Illiquidity']].dropna()
        lag2_df = row_df[[f'{sentiment_col}_Lag_2', 'Illiquidity']].dropna()

        r1 = pearson(lag1_df[f'{sentiment_col}_Lag_1'], lag1_df['Illiquidity'])
        r2 = pearson(lag2_df[f'{sentiment_col}_Lag_2'], lag2_df['Illiquidity'])
        stats.extend([
            {'Sentiment_Model': sentiment_label, 'Lag': 'Lag 1', 'Pearson_r': round(float(r1), 4), 'N': len(lag1_df)},
            {'Sentiment_Model': sentiment_label, 'Lag': 'Lag 2', 'Pearson_r': round(float(r2), 4), 'N': len(lag2_df)},
        ])

        axes[row, 0].scatter(lag1_df[f'{sentiment_col}_Lag_1'], lag1_df['Illiquidity'], color=color, edgecolor='black', alpha=0.7)
        axes[row, 0].set_title(f'{sentiment_label} Lag 1: Sentiment(t-1) vs Illiquidity(t)')
        axes[row, 0].set_xlabel('Sentiment Lag 1')
        axes[row, 0].set_ylabel('Illiquidity')
        axes[row, 0].grid(alpha=0.25)
        axes[row, 0].text(
            0.05,
            0.92,
            f'Pearson r = {r1:.3f}',
            transform=axes[row, 0].transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

        axes[row, 1].scatter(lag2_df[f'{sentiment_col}_Lag_2'], lag2_df['Illiquidity'], color=color, edgecolor='black', alpha=0.7)
        axes[row, 1].set_title(f'{sentiment_label} Lag 2: Sentiment(t-2) vs Illiquidity(t)')
        axes[row, 1].set_xlabel('Sentiment Lag 2')
        axes[row, 1].set_ylabel('Illiquidity')
        axes[row, 1].grid(alpha=0.25)
        axes[row, 1].text(
            0.05,
            0.92,
            f'Pearson r = {r2:.3f}',
            transform=axes[row, 1].transAxes,
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )

    caption = (
        'Lag analysis investigates whether sentiment effects on liquidity are immediate or delayed by one and two days across VADER, TextBlob, and FinBERT.'
    )
    fig.text(0.5, -0.05, caption, ha='center', fontsize=9)
    _save_figure(fig, OUTPUT_DIR / 'lag_analysis.png')

    _save_csv(pd.DataFrame(stats), OUTPUT_DIR / 'lag_statistics.csv')


def plot_final_feature_heatmap(final_df: pd.DataFrame, daily_df: pd.DataFrame, daily_finbert_df: pd.DataFrame) -> None:
    merged = final_df.drop(
        columns=[
            'FinBERT_Sentiment',
            'repost_count',
            'unique_author_count',
            'information_diffusion_score',
        ],
        errors='ignore'
    ).merge(
        daily_df[['Date', 'repost_count', 'unique_author_count', 'information_diffusion_score']],
        on='Date',
        how='left'
    )
    merged = merged.merge(daily_finbert_df[['Date', 'FinBERT_Sentiment']], on='Date', how='left')

    columns = [
        'Close',
        'Volume',
        'Return',
        'Volatility_7D',
        'Turnover_Ratio',
        'FinBERT_Sentiment',
        'comment_count',
        'repost_count',
        'unique_author_count',
        'information_diffusion_score',
        'Illiquidity',
    ]
    df = merged[columns].copy().apply(pd.to_numeric, errors='coerce')
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    cax = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha='right')
    ax.set_yticklabels(columns)
    ax.set_title('Final Feature Correlation Heatmap for Liquidity Prediction')

    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f'{corr.iat[i, j]:.2f}', ha='center', va='center', color='black', fontsize=8)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    _save_figure(fig, OUTPUT_DIR / 'final_feature_heatmap.png')

    corr.to_csv(OUTPUT_DIR / 'correlation_matrix.csv')


def generate_eda_storytelling() -> None:
    raw_posts = _load_raw_posts()
    deduped_posts = _load_deduped_posts()
    raw_sentiment_df = _load_raw_combined_sentiment()
    daily_df = _load_preprocessed_daily()
    daily_finbert_df = _load_daily_finbert_sentiment()
    final_df = _load_final_merged()
    _save_csv(daily_finbert_df, OUTPUT_DIR / 'daily_finbert_sentiment.csv')

    plot_duplicate_repost_impact(raw_posts, deduped_posts)
    plot_top20_contributors(deduped_posts)
    plot_sentiment_model_comparison(raw_sentiment_df, daily_finbert_df)
    plot_raw_sentiment_trends(raw_sentiment_df, daily_finbert_df)
    plot_daily_sentiment_trends(raw_sentiment_df, daily_finbert_df)
    plot_finbert_distribution(daily_finbert_df)
    plot_sentiment_vs_volume(raw_sentiment_df, daily_finbert_df, final_df)
    plot_sentiment_vs_illiquidity(raw_sentiment_df, daily_finbert_df, final_df)
    plot_information_diffusion(daily_df)
    plot_lag_analysis(raw_sentiment_df, final_df, daily_finbert_df)
    plot_final_feature_heatmap(final_df, daily_df, daily_finbert_df)


if __name__ == '__main__':
    generate_eda_storytelling()
