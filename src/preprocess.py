from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load soybean price dataset."""
    df = pd.read_csv(csv_path)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for training and visualization."""
    data = df.copy()
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values('date').reset_index(drop=True)

    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['day_of_week'] = data['date'].dt.dayofweek
    data['day_of_year'] = data['date'].dt.dayofyear
    data['week_of_year'] = data['date'].dt.isocalendar().week.astype(int)
    data['quarter'] = data['date'].dt.quarter

    # Lag features for a more realistic time-series-style regression approach.
    data['price_lag_1'] = data['price'].shift(1)
    data['price_lag_5'] = data['price'].shift(5)
    data['price_lag_20'] = data['price'].shift(20)
    data['rolling_mean_5'] = data['price'].rolling(window=5).mean().shift(1)
    data['rolling_mean_20'] = data['price'].rolling(window=20).mean().shift(1)
    data['rolling_std_20'] = data['price'].rolling(window=20).std().shift(1)

    data = data.dropna().reset_index(drop=True)
    return data


def build_features(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Split features and target."""
    feature_cols = [
        'day', 'month', 'year', 'day_of_week', 'day_of_year', 'week_of_year', 'quarter',
        'usd_brl', 'rainfall_index', 'supply_index',
        'price_lag_1', 'price_lag_5', 'price_lag_20',
        'rolling_mean_5', 'rolling_mean_20', 'rolling_std_20'
    ]
    X = data[feature_cols]
    y = data['price']
    return X, y, feature_cols
