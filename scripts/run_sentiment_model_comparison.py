from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / '.cache'
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(CACHE_DIR / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(CACHE_DIR))

from sklearn.model_selection import train_test_split

from src.feature_engineering import create_features
from src.train_models import train_linear_regression, train_random_forest, train_xgboost
from src.evaluate_models import evaluate_model

RESULTS_DIR = PROJECT_ROOT / 'outputs' / 'results'
FIGURE_DIR = RESULTS_DIR / 'figures'
BASE_DATA_PATH = RESULTS_DIR / 'final_merged_dataset.csv'

SENTIMENT_SPECS = [
    ('Avg_Sentiment', 'VADER'),
    ('TextBlob_Sentiment', 'TextBlob'),
    ('FinBERT_Sentiment', 'FinBERT'),
]

MODEL_SPECS = [
    ('Linear Regression', train_linear_regression, 'Linear_Regression_Predictions'),
    ('Random Forest', train_random_forest, 'Random_Forest_Predictions'),
    ('XGBoost', train_xgboost, 'XGBoost_Predictions'),
]


def _load_base_dataset() -> pd.DataFrame:
    if not BASE_DATA_PATH.exists():
        raise FileNotFoundError(
            f'{BASE_DATA_PATH} does not exist. Run the main preprocessing pipeline first.'
        )

    df = pd.read_csv(BASE_DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    return df


def _run_one_sentiment(base_df: pd.DataFrame, sentiment_column: str, sentiment_label: str):
    feature_df = create_features(base_df, sentiment_column=sentiment_column)
    feature_df['Date'] = pd.to_datetime(feature_df['Date'], errors='coerce')
    feature_df = feature_df.sort_values('Date').reset_index(drop=True)

    feature_columns = [
        'Close',
        'Volume',
        'Return',
        'Volatility_7D',
        'Turnover_Ratio',
        sentiment_column,
        'Sentiment_Lag_1',
        'Sentiment_Lag_2',
        'information_diffusion_score',
        'tesla_relevance',
    ]
    X = feature_df[feature_columns]
    y = feature_df['Illiquidity']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
        random_state=42,
    )

    target_scale = 1e10
    y_train_scaled = y_train * target_scale

    result_rows = []
    prediction_frame = pd.DataFrame({
        'Date': feature_df.loc[X_test.index, 'Date'].dt.strftime('%Y-%m-%d').reset_index(drop=True),
        'Actual': y_test.reset_index(drop=True),
    })

    for model_label, trainer, pred_col in MODEL_SPECS:
        model = trainer(X_train, y_train_scaled)
        metrics = evaluate_model(model, X_test, y_test, prediction_scale=1 / target_scale)

        result_rows.append({
            'Sentiment_Model': sentiment_label,
            'Model': model_label,
            **metrics,
            'Test_Samples': len(X_test),
        })

        prediction_frame[f'{sentiment_label}_{pred_col}'] = model.predict(X_test) / target_scale

    return pd.DataFrame(result_rows), prediction_frame


def _save_rmse_figure(results_df: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pivot = results_df.pivot(index='Model', columns='Sentiment_Model', values='RMSE')
    pivot = pivot[['VADER', 'TextBlob', 'FinBERT']]

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind='bar', ax=ax, color=['#2563eb', '#16a34a', '#dc2626'], width=0.8)
    ax.set_title('RMSE by Model and Sentiment Score')
    ax.set_xlabel('Machine Learning Model')
    ax.set_ylabel('RMSE')
    ax.grid(axis='y', alpha=0.25)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Sentiment Model')
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / 'sentiment_model_rmse_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    base_df = _load_base_dataset()

    all_results = []
    prediction_frames = []
    for sentiment_column, sentiment_label in SENTIMENT_SPECS:
        sentiment_results, sentiment_predictions = _run_one_sentiment(
            base_df,
            sentiment_column,
            sentiment_label,
        )
        all_results.append(sentiment_results)
        prediction_frames.append(sentiment_predictions)

    results_df = pd.concat(all_results, ignore_index=True)
    predictions_df = prediction_frames[0]
    for frame in prediction_frames[1:]:
        predictions_df = predictions_df.merge(frame, on=['Date', 'Actual'], how='inner')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / 'sentiment_model_results.csv', index=False)
    predictions_df.to_csv(RESULTS_DIR / 'sentiment_model_predictions.csv', index=False)

    summary = (
        results_df.sort_values(['Sentiment_Model', 'RMSE'])
        .groupby('Sentiment_Model', as_index=False)
        .first()[['Sentiment_Model', 'Model', 'RMSE', 'MAE', 'R2']]
    )
    summary.to_csv(RESULTS_DIR / 'sentiment_model_best_summary.csv', index=False)

    _save_rmse_figure(results_df)

    print(results_df)
    print(summary)
    print(f'Saved comparison outputs to {RESULTS_DIR}')


if __name__ == '__main__':
    main()
