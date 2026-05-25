
import pandas as pd

def preprocess_financial_data(financial_path):

    df = pd.read_csv(financial_path)

    df['Date'] = pd.to_datetime(df['Date'])

    numeric_cols = ['Open','High','Low','Close','Volume']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    df = df.sort_values('Date')

    return df
