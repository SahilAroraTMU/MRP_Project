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


def plot_model_metrics() -> None:
    df = pd.read_csv(RESULTS_DIR / 'model_results.csv')
    metric_cols = ['MAE', 'RMSE', 'R2']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    palette = ['#2563eb', '#16a34a', '#dc2626']

    for ax, metric, color in zip(axes, metric_cols, palette):
        sns.barplot(data=df, x='Model', y=metric, ax=ax, color=color)
        ax.set_title(metric)
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=0.25)
        ax.tick_params(axis='x', rotation=20)

    fig.suptitle('Model Performance Summary', y=1.03)
    fig.tight_layout()
    _save(fig, 'model_performance_summary.png')


def plot_predictions_vs_actual() -> None:
    df = pd.read_csv(RESULTS_DIR / 'model_predictions.csv')
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    fig, ax = plt.subplots(figsize=(8, 8))
    actual = df['Actual']
    predicted_cols = [
        ('Linear_Regression_Predictions', '#2563eb'),
        ('Random_Forest_Predictions', '#16a34a'),
        ('XGBoost_Predictions', '#dc2626'),
    ]
    for col, color in predicted_cols:
        ax.scatter(actual, df[col], s=20, alpha=0.45, label=col.replace('_Predictions', '').replace('_', ' '), color=color, edgecolor='none')

    min_val = min(actual.min(), *(df[col].min() for col, _ in predicted_cols))
    max_val = max(actual.max(), *(df[col].max() for col, _ in predicted_cols))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', linewidth=1, label='Perfect fit')
    ax.set_title('Predicted vs Actual Illiquidity')
    ax.set_xlabel('Actual Illiquidity')
    ax.set_ylabel('Predicted Illiquidity')
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, 'predictions_vs_actual.png')


def plot_actual_vs_model_predictions() -> None:
    df = pd.read_csv(RESULTS_DIR / 'model_predictions.csv')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    df = df.reset_index(drop=True)
    df['Sample'] = df.index + 1

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    series = [
        ('Linear_Regression_Predictions', '#2563eb'),
        ('Random_Forest_Predictions', '#16a34a'),
        ('XGBoost_Predictions', '#dc2626'),
    ]

    for ax, (col, color) in zip(axes, series):
        ax.plot(df['Sample'], df['Actual'], color='#111827', linewidth=1.6, label='Actual')
        ax.plot(df['Sample'], df[col], color=color, linewidth=1.6, label=col.replace('_Predictions', '').replace('_', ' '))
        ax.set_title(col.replace('_Predictions', '').replace('_', ' '))
        ax.set_ylabel('Illiquidity')
        ax.grid(alpha=0.25)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Test Sample')
    fig.suptitle('Actual vs Model Predictions', y=0.995)
    _save(fig, 'actual_vs_model_predictions.png')


def plot_prediction_residuals() -> None:
    df = pd.read_csv(RESULTS_DIR / 'model_predictions.csv')
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, col, color in zip(
        axes,
        ['Linear_Regression_Predictions', 'Random_Forest_Predictions', 'XGBoost_Predictions'],
        ['#2563eb', '#16a34a', '#dc2626']
    ):
        residuals = df['Actual'] - df[col]
        ax.scatter(df[col], residuals, s=18, alpha=0.4, color=color, edgecolor='none')
        ax.axhline(0, color='black', linestyle='--', linewidth=1)
        ax.set_title(col.replace('_Predictions', '').replace('_', ' '))
        ax.set_xlabel('Predicted')
        ax.grid(alpha=0.25)

    axes[0].set_ylabel('Residual')
    fig.suptitle('Prediction Residuals by Model')
    _save(fig, 'prediction_residuals.png')


def plot_correlation_heatmap() -> None:
    corr = pd.read_csv(RESULTS_DIR / 'correlation_matrix.csv', index_col=0)
    corr = corr.apply(pd.to_numeric, errors='coerce')

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, cmap='coolwarm', vmin=-1, vmax=1, annot=False, ax=ax)
    ax.set_title('Result Correlation Heatmap')
    _save(fig, 'correlation_heatmap.png')


def main() -> None:
    plot_model_metrics()
    plot_predictions_vs_actual()
    plot_actual_vs_model_predictions()
    plot_prediction_residuals()
    plot_correlation_heatmap()
    print(f'Saved figures to {FIGURE_DIR}')


if __name__ == '__main__':
    main()
