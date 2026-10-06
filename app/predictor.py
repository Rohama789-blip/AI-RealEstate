import json
import joblib, pandas as pd
from .data import MODEL_PATH, METRICS_PATH, FEATURES, load_clean

_model = None

def _get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError("Model not found. Run: python scripts/train_model.py")
        _model = joblib.load(MODEL_PATH)
    return _model

def metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

def predict_price(features: dict) -> dict:
    row = pd.DataFrame([features])[FEATURES]
    price = float(max(_get_model().predict(row)[0], 0))
    best = metrics().get("best_model")
    mae = metrics().get("results", {}).get(best, {}).get("MAE", price * 0.1)
    df = load_clean()
    area_avg = df[df.location == features["location"]].price_per_sqft.mean()
    return {
        "predicted_price": round(price, -2),
        "range_low": round(max(price - mae, 0), -2),
        "range_high": round(price + mae, -2),
        "price_per_sqft": round(price / features["area_sqft"], 1),
        "area_avg_price_per_sqft": round(float(area_avg), 1),
        "model": best,
    }
