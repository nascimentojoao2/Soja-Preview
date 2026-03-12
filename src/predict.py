from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / 'models' / 'soybean_model.pkl'
PROCESSED_DATA_PATH = BASE_DIR / 'models' / 'processed_dataset.csv'


def load_artifact(model_path: str | Path = MODEL_PATH):
    return joblib.load(model_path)


def _build_row_from_latest(future_date: str | pd.Timestamp, latest_row: pd.Series) -> pd.DataFrame:
    future_date = pd.to_datetime(future_date)
    row = {
        'day': future_date.day,
        'month': future_date.month,
        'year': future_date.year,
        'day_of_week': future_date.dayofweek,
        'day_of_year': future_date.dayofyear,
        'week_of_year': int(future_date.isocalendar().week),
        'quarter': future_date.quarter,
        'usd_brl': latest_row['usd_brl'],
        'rainfall_index': latest_row['rainfall_index'],
        'supply_index': latest_row['supply_index'],
        'price_lag_1': latest_row['price'],
        'price_lag_5': latest_row['price_lag_5'],
        'price_lag_20': latest_row['price_lag_20'],
        'rolling_mean_5': latest_row['rolling_mean_5'],
        'rolling_mean_20': latest_row['rolling_mean_20'],
        'rolling_std_20': latest_row['rolling_std_20'],
    }
    return pd.DataFrame([row])


def predict_price(future_date: str | pd.Timestamp):
    artifact = load_artifact()
    model = artifact['model']
    feature_cols = artifact['feature_cols']

    processed = pd.read_csv(PROCESSED_DATA_PATH)
    processed['date'] = pd.to_datetime(processed['date'])
    latest_row = processed.sort_values('date').iloc[-1]

    row = _build_row_from_latest(future_date, latest_row)
    prediction = model.predict(row[feature_cols])[0]
    return float(prediction)


if __name__ == '__main__':
    example_date = '2026-04-01'
    predicted = predict_price(example_date)
    print(f'Predicted soybean price for {example_date}: {predicted:.2f}')
