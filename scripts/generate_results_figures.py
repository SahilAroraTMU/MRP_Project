from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / '.cache'
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(CACHE_DIR / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(CACHE_DIR))

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RESULTS_DIR = PROJECT_ROOT / 'outputs' / 'results'
FIGURE_DIR = RESULTS_DIR / 'figures'


def _save(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_predictions_vs_actual() -> None:
    df = pd.read_csv(RESULTS_DIR / 'sentiment_model_predictions.csv')
    df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')
    prediction_cols = [
        'VADER_Linear_Regression_Predictions',
        'VADER_Random_Forest_Predictions',
        'VADER_XGBoost_Predictions',
        'TextBlob_Linear_Regression_Predictions',
        'TextBlob_Random_Forest_Predictions',
        'TextBlob_XGBoost_Predictions',
        'FinBERT_Linear_Regression_Predictions',
        'FinBERT_Random_Forest_Predictions',
        'FinBERT_XGBoost_Predictions',
    ]
    for col in prediction_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Actual'] + prediction_cols)

    model_specs = [
        ('Linear Regression', 'Linear_Regression_Predictions', '#2563eb'),
        ('Random Forest', 'Random_Forest_Predictions', '#16a34a'),
        ('XGBoost', 'XGBoost_Predictions', '#dc2626'),
    ]
    sentiment_specs = [
        ('VADER', 'VADER_'),
        ('TextBlob', 'TextBlob_'),
        ('FinBERT', 'FinBERT_'),
    ]

    actual = df['Actual']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    min_val = min(actual.min(), *(df[col].min() for col in prediction_cols))
    max_val = max(actual.max(), *(df[col].max() for col in prediction_cols))

    for ax, (sentiment_label, prefix) in zip(axes, sentiment_specs):
        for model_label, model_suffix, color in model_specs:
            col = f'{prefix}{model_suffix}'
            ax.scatter(
                actual,
                df[col],
                s=18,
                alpha=0.45,
                label=model_label,
                color=color,
                edgecolor='none',
            )
        ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', linewidth=1, label='Perfect fit')
        ax.set_title(sentiment_label)
        ax.set_xlabel('Actual Illiquidity')
        ax.grid(alpha=0.25)

    axes[0].set_ylabel('Predicted Illiquidity')
    axes[-1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Predicted vs Actual Illiquidity by Sentiment Score', y=1.02)
    _save(fig, 'predictions_vs_actual.png')


def plot_actual_vs_model_predictions() -> None:
    df = pd.read_csv(RESULTS_DIR / 'sentiment_model_predictions.csv')
    df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')
    prediction_cols = [
        'VADER_Linear_Regression_Predictions',
        'VADER_Random_Forest_Predictions',
        'VADER_XGBoost_Predictions',
        'TextBlob_Linear_Regression_Predictions',
        'TextBlob_Random_Forest_Predictions',
        'TextBlob_XGBoost_Predictions',
        'FinBERT_Linear_Regression_Predictions',
        'FinBERT_Random_Forest_Predictions',
        'FinBERT_XGBoost_Predictions',
    ]
    for col in prediction_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Actual'] + prediction_cols).reset_index(drop=True)
    df['Sample'] = df.index + 1

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    model_specs = [
        (
            'Linear Regression',
            'Linear_Regression_Predictions',
        ),
        (
            'Random Forest',
            'Random_Forest_Predictions',
        ),
        (
            'XGBoost',
            'XGBoost_Predictions',
        ),
    ]
    sentiment_specs = [
        ('VADER', 'VADER_', '#2563eb'),
        ('TextBlob', 'TextBlob_', '#16a34a'),
        ('FinBERT', 'FinBERT_', '#dc2626'),
    ]

    for ax, (model_label, model_suffix) in zip(axes, model_specs):
        ax.plot(df['Sample'], df['Actual'], color='#111827', linewidth=1.6, label='Actual')
        for sentiment_label, prefix, color in sentiment_specs:
            col = f'{prefix}{model_suffix}'
            ax.plot(
                df['Sample'],
                df[col],
                color=color,
                linewidth=1.5,
                label=sentiment_label,
                alpha=0.9,
            )
        ax.set_title(model_label)
        ax.set_ylabel('Illiquidity')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', ncol=4, fontsize=8)

    axes[-1].set_xlabel('Test Sample')
    fig.suptitle('Actual vs Predicted Illiquidity by Sentiment Score', y=0.995)
    _save(fig, 'actual_vs_model_predictions.png')


def plot_prediction_residuals() -> None:
    df = pd.read_csv(RESULTS_DIR / 'sentiment_model_predictions.csv')
    df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')
    prediction_cols = [
        'VADER_Linear_Regression_Predictions',
        'VADER_Random_Forest_Predictions',
        'VADER_XGBoost_Predictions',
        'TextBlob_Linear_Regression_Predictions',
        'TextBlob_Random_Forest_Predictions',
        'TextBlob_XGBoost_Predictions',
        'FinBERT_Linear_Regression_Predictions',
        'FinBERT_Random_Forest_Predictions',
        'FinBERT_XGBoost_Predictions',
    ]
    for col in prediction_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Actual'] + prediction_cols)

    model_specs = [
        ('Linear Regression', 'Linear_Regression_Predictions', '#2563eb'),
        ('Random Forest', 'Random_Forest_Predictions', '#16a34a'),
        ('XGBoost', 'XGBoost_Predictions', '#dc2626'),
    ]
    sentiment_specs = [
        ('VADER', 'VADER_'),
        ('TextBlob', 'TextBlob_'),
        ('FinBERT', 'FinBERT_'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (sentiment_label, prefix) in zip(axes, sentiment_specs):
        for model_label, model_suffix, color in model_specs:
            col = f'{prefix}{model_suffix}'
            residuals = df['Actual'] - df[col]
            ax.scatter(
                df[col],
                residuals,
                s=18,
                alpha=0.45,
                label=model_label,
                color=color,
                edgecolor='none',
            )
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(sentiment_label)
        ax.set_xlabel('Predicted Illiquidity')
        ax.grid(alpha=0.25)

    axes[0].set_ylabel('Residual')
    axes[-1].legend(loc='lower right', fontsize=8)
    fig.suptitle('Prediction Residuals by Sentiment Score', y=1.02)
    _save(fig, 'prediction_residuals.png')


def plot_correlation_heatmap() -> None:
    df = pd.read_csv(RESULTS_DIR / 'final_merged_dataset.csv')
    base_specs = [
        ('Close', 'Close'),
        ('Volume', 'Volume'),
        ('Return', 'Return'),
        ('Volatility_7D', 'Volatility_7D'),
        ('Illiquidity', 'Illiquidity'),
        ('Turnover_Ratio', 'Turnover_Ratio'),
        ('comment_count', 'Comment_Count'),
    ]
    sentiment_specs = [
        ('VADER', 'Avg_Sentiment'),
        ('TextBlob', 'TextBlob_Sentiment'),
        ('FinBERT', 'FinBERT_Sentiment'),
    ]

    corr_rows = []
    for sentiment_label, sentiment_col in sentiment_specs:
        row = {}
        for source_col, display_label in base_specs:
            pair_df = df[[sentiment_col, source_col]].apply(pd.to_numeric, errors='coerce').dropna()
            row[display_label] = pair_df[sentiment_col].corr(pair_df[source_col])
        corr_rows.append(pd.Series(row, name=sentiment_label))

    corr_df = pd.DataFrame(corr_rows)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    sns.heatmap(
        corr_df,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Pearson correlation'},
        ax=ax,
    )
    ax.set_title('Sentiment Correlations with Financial and Behavioral Variables')
    ax.set_xlabel('Variable')
    ax.set_ylabel('Sentiment Score')
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)
    _save(fig, 'correlation_heatmap.png')


def plot_sentiment_prediction_residuals() -> None:
    df = pd.read_csv(RESULTS_DIR / 'sentiment_model_predictions.csv')
    df['Actual'] = pd.to_numeric(df['Actual'], errors='coerce')
    prediction_cols = [
        'VADER_Linear_Regression_Predictions',
        'VADER_Random_Forest_Predictions',
        'VADER_XGBoost_Predictions',
        'TextBlob_Linear_Regression_Predictions',
        'TextBlob_Random_Forest_Predictions',
        'TextBlob_XGBoost_Predictions',
        'FinBERT_Linear_Regression_Predictions',
        'FinBERT_Random_Forest_Predictions',
        'FinBERT_XGBoost_Predictions',
    ]
    for col in prediction_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['Actual'] + prediction_cols).reset_index(drop=True)
    df['Sample'] = df.index + 1

    model_specs = [
        ('Linear Regression', 'Linear_Regression_Predictions'),
        ('Random Forest', 'Random_Forest_Predictions'),
        ('XGBoost', 'XGBoost_Predictions'),
    ]
    sentiment_specs = [
        ('VADER', 'VADER_', '#2563eb'),
        ('TextBlob', 'TextBlob_', '#16a34a'),
        ('FinBERT', 'FinBERT_', '#dc2626'),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True, sharey=True)
    for ax, (model_label, model_suffix) in zip(axes, model_specs):
        for sentiment_label, prefix, color in sentiment_specs:
            col = f'{prefix}{model_suffix}'
            residuals = df['Actual'] - df[col]
            ax.plot(
                df['Sample'],
                residuals,
                linewidth=1.3,
                label=sentiment_label,
                color=color,
                alpha=0.9,
            )
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(model_label)
        ax.set_ylabel('Residual')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right', ncol=3, fontsize=8)

    axes[-1].set_xlabel('Test Sample')
    fig.suptitle('Prediction Residuals by Sentiment Score', y=0.995)
    _save(fig, 'sentiment_prediction_residuals.png')


def plot_sentiment_metric_summary() -> None:
    df = pd.read_csv(RESULTS_DIR / 'sentiment_model_results.csv')
    metric_cols = ['MAE', 'RMSE', 'R2']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=False)

    palettes = {
        'VADER': '#2563eb',
        'TextBlob': '#16a34a',
        'FinBERT': '#dc2626',
    }

    for ax, metric in zip(axes, metric_cols):
        sns.barplot(
            data=df,
            x='Model',
            y=metric,
            hue='Sentiment_Model',
            ax=ax,
            palette=palettes,
        )
        ax.set_title(metric)
        ax.set_xlabel('Machine Learning Model')
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=0.25)
        ax.tick_params(axis='x', rotation=20)
        if metric != 'R2':
            ax.legend_.remove()
        else:
            ax.legend(title='Sentiment Model', loc='upper center', bbox_to_anchor=(1.25, 1.15))

    fig.suptitle('Sentiment-Aware Model Performance Summary', y=1.03)
    fig.tight_layout()
    _save(fig, 'sentiment_model_performance_summary.png')


def main() -> None:
    plot_sentiment_metric_summary()
    plot_predictions_vs_actual()
    plot_actual_vs_model_predictions()
    plot_prediction_residuals()
    plot_correlation_heatmap()
    plot_sentiment_prediction_residuals()
    print(f'Saved figures to {FIGURE_DIR}')


if __name__ == '__main__':
    main()
