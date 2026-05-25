
import pandas as pd

def _ensure_comment_link_id(comments_df):
    comments_df = comments_df.copy()

    if 'link_id' not in comments_df.columns:
        comments_df['link_id'] = (
            comments_df['permalink']
            .astype(str)
            .str.extract(r'/comments/([^/]+)/')[0]
        )

    comments_df['link_id'] = (
        comments_df['link_id']
        .astype(str)
        .str.replace('^t3_', '', regex=True)
    )

    return comments_df

def build_post_comment_relationship(posts_df, comments_df):
    comments_df = _ensure_comment_link_id(comments_df)

    return comments_df.merge(
        posts_df,
        left_on='link_id',
        right_on='id',
        how='left',
        suffixes=('_comment', '_post')
    )

def calculate_comment_counts(posts_df, merged_relation):
    comment_counts = (
        merged_relation.groupby('link_id')
        .size()
        .reset_index(name='comment_count')
    )

    posts_df = posts_df.merge(
        comment_counts,
        left_on='id',
        right_on='link_id',
        how='left'
    )

    posts_df['comment_count'] = posts_df['comment_count'].fillna(0)

    return posts_df

def detect_orphan_comments(merged_relation):
    merged_relation['has_parent_post'] = (
        merged_relation['title'].notna()
    )
    return merged_relation
