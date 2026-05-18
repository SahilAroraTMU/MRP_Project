import os
import sys
from pathlib import Path

project_cache_dir = Path(__file__).resolve().parent / '.cache'
project_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(project_cache_dir / 'matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(project_cache_dir))
os.environ.setdefault('MPLBACKEND', 'Agg')

if os.environ.get('MRP_OFFLINE') == '1':
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import pandas as pd

from sklearn.model_selection import train_test_split

from src.fetch_data import (
    fetch_yahoo_finance_data
)

from src.preprocess_data import (
    preprocess_financial_data,
    preprocess_reddit_data,
    combine_reddit_data,
    aggregate_reddit_sentiment,
    merge_datasets
)

from src.sentiment_analysis import (
    apply_fast_sentiment,
    aggregate_daily_text,
    apply_daily_finbert
)

from src.feature_engineering import (
    create_features
)

from src.eda import (
    plot_close_price,
    plot_volume,
    plot_sentiment_distribution,
    boxplot_outlier_detection,
    correlation_heatmap
)

from src.lag_analysis import (
    lag_analysis
)

from src.train_models import (
    train_linear_regression,
    train_random_forest,
    train_xgboost
)

from src.evaluate_models import (
    evaluate_model
)


def run_outputs_and_models(merged_df):
    merged_df = create_features(
        merged_df
    )

    merged_df.to_csv(
        'outputs/results/final_merged_dataset.csv',
        index=False
    )

    plot_close_price(merged_df)

    plot_volume(merged_df)

    plot_sentiment_distribution(
        merged_df
    )

    boxplot_outlier_detection(
        merged_df
    )

    correlation_heatmap(
        merged_df
    )

    lag_analysis(merged_df)

    X = merged_df[[
        'Close',
        'Volume',
        'Return',
        'Volatility_7D',
        'Turnover_Ratio',
        'Avg_Sentiment',
        'TextBlob_Sentiment',
        'FinBERT_Sentiment',
        'Comment_Count',
        'Avg_Reddit_Score',
        'Sentiment_Lag_1',
        'Sentiment_Lag_2'
    ]]

    y = merged_df['Illiquidity']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False,
        random_state=42
    )

    lr_model = train_linear_regression(
        X_train,
        y_train
    )

    rf_model = train_random_forest(
        X_train,
        y_train
    )

    xgb_model = train_xgboost(
        X_train,
        y_train
    )

    lr_results = evaluate_model(
        lr_model,
        X_test,
        y_test
    )

    rf_results = evaluate_model(
        rf_model,
        X_test,
        y_test
    )

    xgb_results = evaluate_model(
        xgb_model,
        X_test,
        y_test
    )

    print('\nLinear Regression')
    print(lr_results)

    print('\nRandom Forest')
    print(rf_results)

    print('\nXGBoost')
    print(xgb_results)

    results_df = pd.DataFrame([
        {
            'Model': 'Linear Regression',
            **lr_results
        },
        {
            'Model': 'Random Forest',
            **rf_results
        },
        {
            'Model': 'XGBoost',
            **xgb_results
        }
    ])

    results_df.to_csv(
        'outputs/results/model_results.csv',
        index=False
    )

    predictions_df = pd.DataFrame({

        'Actual': y_test,

        'Linear_Regression_Predictions':
            lr_model.predict(X_test),

        'Random_Forest_Predictions':
            rf_model.predict(X_test),

        'XGBoost_Predictions':
            xgb_model.predict(X_test)

    })

    predictions_df.to_csv(
        'outputs/results/model_predictions.csv',
        index=False
    )

    print(
        '\nProject execution completed successfully.'
    )


cached_merged_path = Path('data/processed/merged_dataset.csv')
cached_final_path = Path('outputs/results/final_merged_dataset.csv')

if cached_merged_path.exists() or cached_final_path.exists():
    cached_path = cached_merged_path
    cached_merged_df = pd.read_csv(cached_path) if cached_path.exists() else pd.DataFrame()

    if cached_merged_df.empty and cached_final_path.exists():
        cached_path = cached_final_path
        cached_merged_df = pd.read_csv(
            cached_path
        )

    finbert_is_placeholder = (
        'FinBERT_Sentiment' in cached_merged_df.columns
        and cached_merged_df['FinBERT_Sentiment'].fillna(0).eq(0).all()
    )

    if not cached_merged_df.empty and not finbert_is_placeholder:
        print(f'Using cached merged dataset: {cached_path}')
        cached_merged_df['Date'] = pd.to_datetime(
            cached_merged_df['Date']
        ).dt.date
        run_outputs_and_models(
            cached_merged_df
        )
        sys.exit(0)

    if finbert_is_placeholder:
        print('Cached FinBERT sentiment is all neutral; rebuilding from raw data.')

    print('Cached merged datasets are empty; rebuilding from raw data.')


# ---------------------------------------------------
# STEP 1 - FETCH TSLA DATA
# ---------------------------------------------------

fetch_yahoo_finance_data()


# ---------------------------------------------------
# STEP 2 - LOAD REDDIT DATASETS
# ---------------------------------------------------

comments_df = preprocess_reddit_data(
    'data/raw/reddit_comments.csv'
)

posts_df = preprocess_reddit_data(
    'data/raw/reddit_posts.csv'
)


# ---------------------------------------------------
# STEP 3 - COMBINE REDDIT DATA
# ---------------------------------------------------

reddit_df = combine_reddit_data(
    comments_df,
    posts_df
)


# ---------------------------------------------------
# STEP 4 - FAST SENTIMENT
# VADER + TEXTBLOB
# ---------------------------------------------------

reddit_df = apply_fast_sentiment(
    reddit_df
)


# ---------------------------------------------------
# STEP 5 - DAILY TEXT AGGREGATION
# FOR FINBERT
# ---------------------------------------------------

daily_text = aggregate_daily_text(
    reddit_df
)


# ---------------------------------------------------
# STEP 6 - APPLY FINBERT DAILY
# ---------------------------------------------------

daily_text = apply_daily_finbert(
    daily_text
)


# ---------------------------------------------------
# STEP 7 - AGGREGATE DAILY SENTIMENT
# ---------------------------------------------------

reddit_daily = (
    reddit_df
    .groupby('Date')
    .agg({
        'VADER_Sentiment': 'mean',
        'TextBlob_Sentiment': 'mean',
        'score': 'mean',
        'text': 'count'
    })
    .reset_index()
)

reddit_daily.rename(
    columns={
        'VADER_Sentiment': 'Avg_Sentiment',
        'score': 'Avg_Reddit_Score',
        'text': 'Comment_Count'
    },
    inplace=True
)


# ---------------------------------------------------
# STEP 8 - MERGE FINBERT RESULTS
# ---------------------------------------------------

reddit_daily = reddit_daily.merge(
    daily_text[
        ['Date', 'FinBERT_Sentiment']
    ],
    on='Date',
    how='left'
)


# ---------------------------------------------------
# STEP 9 - LOAD FINANCIAL DATA
# ---------------------------------------------------

financial_df = preprocess_financial_data()


# ---------------------------------------------------
# STEP 10 - MERGE DATASETS
# ---------------------------------------------------

merged_df = merge_datasets(
    financial_df,
    reddit_daily
)


# ---------------------------------------------------
# STEP 11 - FEATURE ENGINEERING
# ---------------------------------------------------

merged_df = create_features(
    merged_df
)


# ---------------------------------------------------
# STEP 12 - SAVE MERGED DATASET
# ---------------------------------------------------

merged_df.to_csv(
    'outputs/results/final_merged_dataset.csv',
    index=False
)


# ---------------------------------------------------
# STEP 13 - EDA
# ---------------------------------------------------

plot_close_price(merged_df)

plot_volume(merged_df)

plot_sentiment_distribution(
    merged_df
)

boxplot_outlier_detection(
    merged_df
)

correlation_heatmap(
    merged_df
)


# ---------------------------------------------------
# STEP 14 - LAG ANALYSIS
# ---------------------------------------------------

lag_analysis(merged_df)


# ---------------------------------------------------
# STEP 15 - FEATURES
# ---------------------------------------------------

X = merged_df[[
    'Close',
    'Volume',
    'Return',
    'Volatility_7D',
    'Turnover_Ratio',
    'Avg_Sentiment',
    'TextBlob_Sentiment',
    'FinBERT_Sentiment',
    'Comment_Count',
    'Avg_Reddit_Score',
    'Sentiment_Lag_1',
    'Sentiment_Lag_2'
]]


# ---------------------------------------------------
# TARGET VARIABLE
# ---------------------------------------------------

y = merged_df['Illiquidity']


# ---------------------------------------------------
# STEP 16 - TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False,
    random_state=42
)


# ---------------------------------------------------
# STEP 17 - TRAIN MODELS
# ---------------------------------------------------

lr_model = train_linear_regression(
    X_train,
    y_train
)

rf_model = train_random_forest(
    X_train,
    y_train
)

xgb_model = train_xgboost(
    X_train,
    y_train
)


# ---------------------------------------------------
# STEP 18 - MODEL EVALUATION
# ---------------------------------------------------

lr_results = evaluate_model(
    lr_model,
    X_test,
    y_test
)

rf_results = evaluate_model(
    rf_model,
    X_test,
    y_test
)

xgb_results = evaluate_model(
    xgb_model,
    X_test,
    y_test
)


# ---------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------

print('\nLinear Regression')
print(lr_results)

print('\nRandom Forest')
print(rf_results)

print('\nXGBoost')
print(xgb_results)


# ---------------------------------------------------
# SAVE MODEL RESULTS
# ---------------------------------------------------

results_df = pd.DataFrame([
    {
        'Model': 'Linear Regression',
        **lr_results
    },
    {
        'Model': 'Random Forest',
        **rf_results
    },
    {
        'Model': 'XGBoost',
        **xgb_results
    }
])

results_df.to_csv(
    'outputs/results/model_results.csv',
    index=False
)


# ---------------------------------------------------
# SAVE PREDICTIONS
# ---------------------------------------------------

predictions_df = pd.DataFrame({

    'Actual': y_test,

    'Linear_Regression_Predictions':
        lr_model.predict(X_test),

    'Random_Forest_Predictions':
        rf_model.predict(X_test),

    'XGBoost_Predictions':
        xgb_model.predict(X_test)

})

predictions_df.to_csv(
    'outputs/results/model_predictions.csv',
    index=False
)


print(
    '\nProject execution completed successfully.'
)
