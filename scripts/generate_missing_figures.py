from pathlib import Path
import os

PROJECT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache"
PROJECT_CACHE_DIR.mkdir(exist_ok=True)
MATPLOTLIB_CACHE_DIR = PROJECT_CACHE_DIR / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_PATH = Path("outputs/results/final_merged_dataset.csv")
FIGURE_DIR = Path("outputs/figures")


def _prepare_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    numeric_cols = [
        "Volume",
        "Volatility_7D",
        "Illiquidity",
        "Avg_Sentiment",
        "Sentiment_Lag_1",
        "Sentiment_Lag_2",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values("Date")


def _save(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def trading_volume(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Volume"], color="#2563eb", linewidth=1.5)
    plt.title("TSLA Daily Trading Volume")
    plt.xlabel("Date")
    plt.ylabel("Trading Volume")
    plt.grid(alpha=0.25)
    _save(FIGURE_DIR / "trading_volume.png")


def rolling_volatility(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(df["Date"], df["Volatility_7D"], color="#dc2626", linewidth=1.5)
    plt.title("7-Day Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(alpha=0.25)
    _save(FIGURE_DIR / "rolling_volatility.png")


def sentiment_vs_illiquidity(df: pd.DataFrame) -> None:
    plot_df = df[["Avg_Sentiment", "Illiquidity"]].dropna()
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=plot_df,
        x="Avg_Sentiment",
        y="Illiquidity",
        scatter_kws={"alpha": 0.65, "s": 36},
        line_kws={"color": "#dc2626"},
    )
    plt.title("Sentiment vs Market Illiquidity")
    plt.xlabel("Average Sentiment")
    plt.ylabel("Illiquidity")
    plt.grid(alpha=0.2)
    _save(FIGURE_DIR / "sentiment_vs_illiquidity.png")


def combined_trends(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    ax1.plot(df["Date"], df["Volume"], color="#2563eb", linewidth=1.3, label="Trading Volume")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Trading Volume", color="#2563eb")
    ax1.tick_params(axis="y", labelcolor="#2563eb")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["Avg_Sentiment"], color="#16a34a", linewidth=1.5, label="Average Sentiment")
    ax2.set_ylabel("Average Sentiment", color="#16a34a")
    ax2.tick_params(axis="y", labelcolor="#16a34a")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left")
    plt.title("Trading Volume and Sentiment Trends")
    _save(FIGURE_DIR / "combined_trends.png")


def sentiment_vs_volume(df: pd.DataFrame) -> None:
    plot_df = df[["Avg_Sentiment", "Volume"]].dropna()
    plt.figure(figsize=(8, 5))
    sns.regplot(
        data=plot_df,
        x="Avg_Sentiment",
        y="Volume",
        scatter_kws={"alpha": 0.65, "s": 36},
        line_kws={"color": "#dc2626"},
    )
    plt.title("Sentiment vs Trading Volume")
    plt.xlabel("Average Sentiment")
    plt.ylabel("Trading Volume")
    plt.grid(alpha=0.2)
    _save(FIGURE_DIR / "sentiment_vs_volume.png")


def lag_plots(df: pd.DataFrame) -> None:
    for lag_col, output_name, title in [
        ("Sentiment_Lag_1", "lag1.png", "Lag-1 Sentiment vs Illiquidity"),
        ("Sentiment_Lag_2", "lag2.png", "Lag-2 Sentiment vs Illiquidity"),
    ]:
        plot_df = df[[lag_col, "Illiquidity"]].dropna()
        plt.figure(figsize=(8, 5))
        sns.regplot(
            data=plot_df,
            x=lag_col,
            y="Illiquidity",
            scatter_kws={"alpha": 0.65, "s": 36},
            line_kws={"color": "#dc2626"},
        )
        plt.title(title)
        plt.xlabel(lag_col.replace("_", " "))
        plt.ylabel("Illiquidity")
        plt.grid(alpha=0.2)
        _save(FIGURE_DIR / output_name)


def main() -> None:
    df = _prepare_data(DATA_PATH)
    trading_volume(df)
    rolling_volatility(df)
    sentiment_vs_illiquidity(df)
    combined_trends(df)
    sentiment_vs_volume(df)
    lag_plots(df)


if __name__ == "__main__":
    main()
