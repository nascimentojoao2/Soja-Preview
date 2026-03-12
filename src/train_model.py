from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from preprocess import load_data, preprocess_data, build_features

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'data' / 'soybean_prices.csv'
MODELS_DIR = BASE_DIR / 'models'
MODELS_DIR.mkdir(exist_ok=True)


def time_split(X: pd.DataFrame, y: pd.Series, train_ratio: float = 0.8):
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def evaluate_model(name: str, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        'model': name,
        'mse': float(mean_squared_error(y_test, preds)),
        'rmse': float(mean_squared_error(y_test, preds) ** 0.5),
        'mae': float(mean_absolute_error(y_test, preds)),
        'r2': float(r2_score(y_test, preds)),
    }
    return model, preds, metrics


def main():
    raw_df = load_data(DATA_PATH)
    data = preprocess_data(raw_df)
    X, y, feature_cols = build_features(data)
    X_train, X_test, y_train, y_test = time_split(X, y)

    models = {
        'LinearRegression': LinearRegression(),
        'RandomForestRegressor': RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    trained = {}
    test_predictions = {}

    for name, model in models.items():
        trained_model, preds, metrics = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
        trained[name] = trained_model
        test_predictions[name] = preds.tolist()
        print(f"{name}: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R2={metrics['r2']:.4f}")

    best_result = max(results, key=lambda item: item['r2'])
    best_model_name = best_result['model']
    best_model = trained[best_model_name]

    artifact = {
        'model': best_model,
        'feature_cols': feature_cols,
        'metrics': results,
        'best_model_name': best_model_name,
        'test_index': X_test.index.tolist(),
        'test_actuals': y_test.tolist(),
        'test_predictions': test_predictions[best_model_name],
        'processed_data_path': str(MODELS_DIR / 'processed_dataset.csv'),
    }

    joblib.dump(artifact, MODELS_DIR / 'soybean_model.pkl')
    data.to_csv(MODELS_DIR / 'processed_dataset.csv', index=False)
    with open(MODELS_DIR / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Saved to: {MODELS_DIR / 'soybean_model.pkl'}")


if __name__ == '__main__':
    main()
