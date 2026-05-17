import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def lag_analysis(df):

    plt.figure(figsize=(8,5))

    plt.scatter(
        df['Sentiment_Lag_1'],
        df['Illiquidity']
    )

    plt.xlabel('Lagged Sentiment')

    plt.ylabel('Illiquidity')

    plt.title(
        'Lagged Sentiment vs Illiquidity'
    )

    plt.savefig(
        'outputs/figures/lag_analysis.png'
    )

    plt.close()
