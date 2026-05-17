from pathlib import Path

import yfinance as yf


def has_cached_rows(path):
    if not path.exists():
        return False

    with path.open() as file:
        return sum(1 for _, _ in zip(range(3), file)) >= 3


def fetch_yahoo_finance_data():
    output_path = Path('data/raw/tsla_yahoo_finance.csv')
    processed_fallback_path = Path('data/processed/merged_dataset.csv')

    if has_cached_rows(output_path):
        print('Using cached TSLA data.')
        return

    try:
        df = yf.download(
            'TSLA',
            start='2021-07-05',
            end='2022-07-04'
        )

        if df.empty:
            raise RuntimeError('Yahoo Finance returned no TSLA rows.')

        df.reset_index(inplace=True)

        df.to_csv(
            output_path,
            index=False
        )

        print('TSLA data downloaded.')

    except Exception as exc:
        if has_cached_rows(output_path):
            print(
                "TSLA download unavailable; using cached data/raw/tsla_yahoo_finance.csv. "
                f"Reason: {exc}"
            )
            return

        if has_cached_rows(processed_fallback_path):
            print(
                "TSLA download unavailable and raw cache is incomplete; "
                "using data/processed/merged_dataset.csv as the financial fallback. "
                f"Reason: {exc}"
            )
            return

        raise
