import os
import sys
from pathlib import Path

project_cache_dir = Path(__file__).resolve().parent / '.cache'
project_cache_dir.mkdir(exist_ok=True)

os.environ.setdefault(
    'MPLCONFIGDIR',
    str(project_cache_dir / 'matplotlib')
)

os.environ.setdefault(
    'XDG_CACHE_HOME',
    str(project_cache_dir)
)

os.environ.setdefault(
    'MPLBACKEND',
    'Agg'
)

if os.environ.get('MRP_OFFLINE') == '1':

    os.environ.setdefault(
        'HF_HUB_OFFLINE',
        '1'
    )

    os.environ.setdefault(
        'TRANSFORMERS_OFFLINE',
        '1'
    )

import pandas as pd

from sklearn.model_selection import (
    train_test_split
)

# ===================================================
# DATA FETCHING
# ===================================================

from src.fetch_data import (
    fetch_yahoo_finance_data
)

# ===================================================
# PREPROCESSING
# ===================================================

from src.preprocess_data import (
    preprocess_financial_data,
    preprocess_reddit_data,
    combine_reddit_data,
    aggregate_reddit_sentiment,
    merge_datasets
)

# ===================================================
# SENTIMENT ANALYSIS
# ===================================================

from src.sentiment_analysis import (
    vader_sentiment,
    apply_fast_sentiment,
    aggregate_daily_text,
    apply_daily_finbert
)

# ===================================================
# FEATURE ENGINEERING
# ===================================================

from src.feature_engineering import (
    create_features
)

# ===================================================
# EDA
# ===================================================

from src.eda import (
    plot_close_price,
    plot_volume,
    plot_sentiment_distribution,
    boxplot_outlier_detection,
    correlation_heatmap,
    duplicate_repost_distribution,
    unique_author_distribution,
    information_diffusion_distribution,
    engagement_score_distribution,
    engagement_score_vs_trading_volume,
    engagement_score_vs_illiquidity
)

from src.lag_analysis import (
    lag_analysis
)

# ===================================================
# MACHINE LEARNING
# ===================================================

from src.train_models import (
    train_linear_regression,
    train_random_forest,
    train_xgboost
)

from src.evaluate_models import (
    evaluate_model
)

# ===================================================
# ADVANCED PREPROCESSING IMPORTS
# ===================================================

from src.preprocessing.reddit_preprocessing import (
    load_reddit_data,
    preprocess_timestamps,
    preprocess_post_reposts,
    preprocess_comment_reposts
)

from src.preprocessing.relationship_preprocessing import (
    build_post_comment_relationship,
    calculate_comment_counts,
    detect_orphan_comments
)

from src.preprocessing.engagement_preprocessing import (
    calculate_engagement_score
)

from src.preprocessing.topic_preprocessing import (
    tesla_relevance,
    detect_topic
)

from src.preprocessing.kalman_preprocessing import (
    apply_kalman_filter
)

from src.preprocessing.final_merge_preprocessing import (
    merge_reddit_financial
)

# ===================================================
# SENTIMENT VALIDATION IMPORTS
# ===================================================

from src.sentiment_validation_module.src.sentiment_validation.sentiment_comparison import (
    sentiment_distribution_comparison,
    sentiment_histograms,
    sentiment_correlation,
    sentiment_scatterplot
)

from src.sentiment_validation_module.src.sentiment_validation.reliability_analysis import (
    create_reliability_dataset
)

from src.sentiment_validation_module.src.sentiment_validation.sarcasm_detection import (
    generate_sarcasm_analysis
)

from src.sentiment_validation_module.src.sentiment_validation.kappa_analysis import (
    calculate_kappa_scores
)

EDA_STORYTELLING_DIR = Path('outputs/eda_storytelling')
DAILY_FINBERT_CANDIDATES = [
    EDA_STORYTELLING_DIR / 'daily_finbert_sentiment.csv',
    EDA_STORYTELLING_DIR / 'daily_aggregated_finbert.csv',
]


def load_external_daily_finbert():

    finbert_path = next(
        (path for path in DAILY_FINBERT_CANDIDATES if path.exists()),
        DAILY_FINBERT_CANDIDATES[0]
    )

    if not finbert_path.exists():
        return None

    df = pd.read_csv(finbert_path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

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
        grouped['FinBERT_Sentiment'] = grouped['total_score'] / grouped['total_count'].replace(0, pd.NA)
        return grouped[['Date', 'FinBERT_Sentiment']]

    if 'FinBERT_Sentiment' not in df.columns and 'Avg_FinBERT_Sentiment' in df.columns:
        df = df.rename(columns={'Avg_FinBERT_Sentiment': 'FinBERT_Sentiment'})

    return (
        df[['Date', 'FinBERT_Sentiment']]
        .dropna(subset=['Date'])
        .groupby('Date', as_index=False)['FinBERT_Sentiment']
        .mean()
    )

# ===================================================
# OUTPUT + MODEL FUNCTION
# ===================================================

def run_outputs_and_models(merged_df):

    merged_df = create_features(
        merged_df
    )

    merged_df.to_csv(
        'outputs/results/final_merged_dataset.csv',
        index=False
    )

    # ---------------------------------------------------
    # EDA
    # ---------------------------------------------------

    plot_close_price(
        merged_df
    )

    plot_volume(
        merged_df
    )

    plot_sentiment_distribution(
        merged_df
    )

    boxplot_outlier_detection(
        merged_df
    )

    correlation_heatmap(
        merged_df
    )

    duplicate_repost_distribution(
        merged_df
    )

    unique_author_distribution(
        merged_df
    )

    information_diffusion_distribution(
        merged_df
    )

    engagement_score_distribution(
        merged_df
    )

    engagement_score_vs_trading_volume(
        merged_df
    )

    engagement_score_vs_illiquidity(
        merged_df
    )

    lag_analysis(
        merged_df
    )

    # ---------------------------------------------------
    # MODEL FEATURES
    # ---------------------------------------------------

    X = merged_df[[
        'Close',
        'Volume',
        'Return',
        'Volatility_7D',
        'Turnover_Ratio',
        'FinBERT_Sentiment',
        'Sentiment_Lag_1',
        'Sentiment_Lag_2',
        'information_diffusion_score',
        'tesla_relevance'
    ]]

    y = merged_df['Illiquidity']

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            shuffle=False,
            random_state=42
        )
    )

    target_scale = 1e10
    y_train_scaled = y_train * target_scale

    # ---------------------------------------------------
    # TRAIN MODELS
    # ---------------------------------------------------

    lr_model = train_linear_regression(
        X_train,
        y_train_scaled
    )

    rf_model = train_random_forest(
        X_train,
        y_train_scaled
    )

    xgb_model = train_xgboost(
        X_train,
        y_train_scaled
    )

    # ---------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------

    lr_results = evaluate_model(
        lr_model,
        X_test,
        y_test,
        prediction_scale=1 / target_scale
    )

    rf_results = evaluate_model(
        rf_model,
        X_test,
        y_test,
        prediction_scale=1 / target_scale
    )

    xgb_results = evaluate_model(
        xgb_model,
        X_test,
        y_test,
        prediction_scale=1 / target_scale
    )

    print('\nLinear Regression')
    print(lr_results)

    print('\nRandom Forest')
    print(rf_results)

    print('\nXGBoost')
    print(xgb_results)

    # ---------------------------------------------------
    # SAVE RESULTS
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


# ===================================================
# STEP 1 - FETCH TSLA DATA
# ===================================================

fetch_yahoo_finance_data()

# ===================================================
# STEP 2 - REDDIT PREPROCESSING
# ===================================================

print(
    '\nRunning advanced preprocessing pipeline...'
)

posts_df, comments_df = load_reddit_data(
    'data/raw/reddit_posts.csv',
    'data/raw/reddit_comments.csv'
)

posts_df = preprocess_timestamps(
    posts_df
)

comments_df = preprocess_timestamps(
    comments_df
)

posts_df = preprocess_post_reposts(
    posts_df
)

comments_df = preprocess_comment_reposts(
    comments_df
)

print(f'Cleaned posts after repost aggregation: {len(posts_df)}')
print(f'Cleaned comments after repost aggregation: {len(comments_df)}')

# ===================================================
# STEP 3 - RELATIONSHIP ANALYSIS
# ===================================================

merged_relation = (
    build_post_comment_relationship(
        posts_df,
        comments_df
    )
)

posts_df = calculate_comment_counts(
    posts_df,
    merged_relation
)

merged_relation = detect_orphan_comments(
    merged_relation
)

orphan_count = (
    (~merged_relation['has_parent_post'])
    .sum()
)

print(
    f'Orphan comments detected: {orphan_count}'
)

# ===================================================
# STEP 4 - TESLA RELEVANCE + TOPICS
# ===================================================

posts_df['tesla_relevance'] = (
    (
        posts_df['title'].fillna('').astype(str) + ' ' +
        posts_df.get(
            'selftext',
            pd.Series('', index=posts_df.index)
        ).fillna('').astype(str)
    )
    .apply(tesla_relevance)
)

posts_df['context_topic'] = (
    posts_df['title']
    .apply(detect_topic)
)

# ===================================================
# STEP 5 - COMBINE REDDIT DATA
# ===================================================

reddit_df = combine_reddit_data(
    comments_df,
    posts_df
)

# ===================================================
# STEP 6 - SENTIMENT ANALYSIS
# ===================================================

reddit_df = apply_fast_sentiment(
    reddit_df
)

daily_text = aggregate_daily_text(
    reddit_df
)

external_daily_finbert = load_external_daily_finbert()

if external_daily_finbert is not None:

    daily_text = daily_text.merge(
        external_daily_finbert,
        on='Date',
        how='left'
    )

else:

    daily_text = apply_daily_finbert(
        daily_text
    )

# ===================================================
# STEP 6.5 - MERGE FINBERT BACK
# ===================================================

reddit_df = reddit_df.merge(

    daily_text[
        ['Date', 'FinBERT_Sentiment']
    ],

    on='Date',
    how='left'

)

posts_df['VADER_Sentiment'] = (
    (
        posts_df['title'].fillna('').astype(str) + ' ' +
        posts_df.get(
            'selftext',
            pd.Series('', index=posts_df.index)
        ).fillna('').astype(str)
    )
    .apply(vader_sentiment)
)

posts_df = calculate_engagement_score(
    posts_df
)

# ===================================================
# STEP 7 - SAVE PREPROCESSED DATASET 1
# ===================================================

with pd.ExcelWriter(
    'data/processed/preprocessed_data_1.xlsx'
) as writer:

    posts_df.to_excel(
        writer,
        sheet_name='cleaned_posts',
        index=False
    )

    comments_df.to_excel(
        writer,
        sheet_name='cleaned_comments',
        index=False
    )

    merged_relation.to_excel(
        writer,
        sheet_name='post_comment_links',
        index=False
    )

print(
    'Saved preprocessed_data_1.xlsx'
)

# ===================================================
# STEP 8 - SENTIMENT VALIDATION
# ===================================================

print(
    '\nRunning sentiment validation analysis...'
)

sentiment_distribution_comparison(
    reddit_df
)

sentiment_histograms(
    reddit_df
)

sentiment_correlation(
    reddit_df
)

sentiment_scatterplot(
    reddit_df
)

create_reliability_dataset(
    reddit_df
)

calculate_kappa_scores(
    reddit_df
)

generate_sarcasm_analysis(
    reddit_df
)



print(
    'Sentiment validation completed.'
)

# ===================================================
# STEP 9 - DAILY REDDIT AGGREGATION
# ===================================================

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

        'VADER_Sentiment':
            'Avg_Sentiment',

        'score':
            'Avg_Reddit_Score',

        'text':
            'Comment_Count'

    },
    inplace=True
)

reddit_daily = reddit_daily.merge(

    daily_text[
        ['Date', 'FinBERT_Sentiment']
    ],

    on='Date',
    how='left'

)

advanced_daily = (
    posts_df
    .groupby('Date')
    .agg({

        'engagement_score': 'mean',

        'tesla_relevance': 'mean',

        'comment_count': 'mean',

        'repost_count': 'mean',

        'unique_author_count': 'mean',

        'discussion_intensity': 'mean',

        'information_diffusion_score': 'mean',

        'sentiment_magnitude': 'mean',

        'VADER_Sentiment': 'mean'

    })
    .reset_index()
)

advanced_daily = advanced_daily.rename(
    columns={
        'VADER_Sentiment': 'Post_VADER_Sentiment'
    }
)

reddit_daily = reddit_daily.merge(
    advanced_daily,
    on='Date',
    how='left'
)

# ===================================================
# STEP 11 - FINANCIAL PREPROCESSING
# ===================================================

financial_df = preprocess_financial_data()

financial_df = apply_kalman_filter(
    financial_df
)

financial_df.to_csv(
    'data/processed/preprocessed_data_2.csv',
    index=False
)

print(
    'Saved preprocessed_data_2.csv'
)

# ===================================================
# STEP 12 - FINAL MERGE
# ===================================================

merged_df = merge_datasets(
    financial_df,
    reddit_daily
)

merged_df.to_csv(
    'data/processed/preprocessed_data_3.csv',
    index=False
)

print(
    'Saved preprocessed_data_3.csv'
)

# ===================================================
# STEP 13 - FEATURE ENGINEERING
# ===================================================

merged_df = create_features(
    merged_df
)

# ===================================================
# STEP 14 - RUN EDA + ML
# ===================================================

run_outputs_and_models(
    merged_df
)
