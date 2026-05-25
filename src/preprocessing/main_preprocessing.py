
from reddit_preprocessing import *
from relationship_preprocessing import *
from engagement_preprocessing import *
from topic_preprocessing import *
from financial_preprocessing import *
from kalman_preprocessing import *
from final_merge_preprocessing import *

posts_df, comments_df = load_reddit_data(
    'data/raw/reddit_posts.csv',
    'data/raw/reddit_comments.csv'
)

posts_df = preprocess_timestamps(posts_df)
comments_df = preprocess_timestamps(comments_df)

posts_df = remove_post_duplicates(posts_df)
comments_df = remove_comment_duplicates(comments_df)

merged_relation = build_post_comment_relationship(
    posts_df,
    comments_df
)

posts_df = calculate_comment_counts(
    posts_df,
    merged_relation
)

merged_relation = detect_orphan_comments(
    merged_relation
)

posts_df['tesla_relevance'] = (
    posts_df['title'].apply(tesla_relevance)
)

posts_df['context_topic'] = (
    posts_df['title'].apply(detect_topic)
)

posts_df = calculate_engagement_factor(posts_df)

posts_df.to_excel(
    'data/processed/preprocessed_data_1.xlsx',
    index=False
)

financial_df = preprocess_financial_data(
    'data/raw/tsla_yahoo_finance.csv'
)

financial_df = apply_kalman_filter(financial_df)

financial_df.to_csv(
    'data/processed/preprocessed_data_2.csv',
    index=False
)

reddit_daily = (
    posts_df.groupby('Date')
    .agg({
        'engagement_factor': 'mean',
        'tesla_relevance': 'mean',
        'comment_count': 'mean'
    })
    .reset_index()
)

final_df = merge_reddit_financial(
    reddit_daily,
    financial_df
)

final_df.to_csv(
    'data/processed/preprocessed_data_3.csv',
    index=False
)

print('All preprocessing completed successfully.')
