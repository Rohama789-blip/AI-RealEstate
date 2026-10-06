"""Shared paths, feature lists and the data-cleaning function."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "properties_raw.csv"
CLEAN_CSV = ROOT / "data" / "properties_clean.csv"
MODEL_PATH = ROOT / "models" / "price_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"

CATEGORICAL = ["location", "property_type"]
NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "age_years", "parking", "garden"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, fill gaps, drop outliers, add price_per_sqft."""
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=["price", "area_sqft", "location", "property_type"])

    # Fill missing values with the median of similar properties
    df["bathrooms"] = df["bathrooms"].fillna(df.groupby("bedrooms")["bathrooms"].transform("median"))
    df["bathrooms"] = df["bathrooms"].fillna(1)
    df["age_years"] = df["age_years"].fillna(df["age_years"].median())

    df["price_per_sqft"] = df["price"] / df["area_sqft"]

    # Outliers: keep rows within 1.5 * IQR on area and price_per_sqft
    for col in ["area_sqft", "price_per_sqft"]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        df = df[df[col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)]

    int_cols = ["bedrooms", "bathrooms", "age_years", "parking", "garden", "listing_year"]
    df[int_cols] = df[int_cols].astype(int)
    return df.reset_index(drop=True)


def load_clean() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        clean_data(pd.read_csv(RAW_CSV)).to_csv(CLEAN_CSV, index=False)
    return pd.read_csv(CLEAN_CSV)
