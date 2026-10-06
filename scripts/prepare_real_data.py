"""Download Kaggle's House Prices dataset (Ames, Iowa) and convert it to this project's schema.

Run:  python scripts/prepare_real_data.py
Then: python scripts/eda.py && python scripts/train_model.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from app.data import RAW_CSV, CLEAN_CSV

TYPE_NAMES = {"1Fam": "Single-family", "2fmCon": "Two-family conversion", "Duplx": "Duplex",
              "TwnhsE": "Townhouse (end unit)", "TwnhsI": "Townhouse (inside unit)"}


def convert(src: pd.DataFrame) -> pd.DataFrame:
    """Map Kaggle column names to the columns the app expects."""
    return pd.DataFrame({
        "location": src["Neighborhood"],
        "property_type": src["BldgType"].map(TYPE_NAMES).fillna(src["BldgType"]),
        "area_sqft": src["GrLivArea"],                                   # above-ground living area
        "bedrooms": src["BedroomAbvGr"].clip(lower=1),
        "bathrooms": (src["FullBath"] + src["HalfBath"]).clip(lower=1),
        "age_years": (src["YrSold"] - src["YearBuilt"]).clip(lower=0),
        "parking": src["GarageCars"].fillna(0),                          # garage capacity (cars)
        # The dataset has no "garden" field. Proxy: lot space beyond the ground floor of at least 5,000 sq ft.
        "garden": ((src["LotArea"] - src["1stFlrSF"]) >= 5000).astype(int),
        "listing_year": src["YrSold"],
        "price": src["SalePrice"],
    })


def main():
    import kagglehub
    path = Path(kagglehub.competition_download("house-prices-advanced-regression-techniques"))
    train = next(path.rglob("train.csv"))
    out = convert(pd.read_csv(train))
    out.to_csv(RAW_CSV, index=False)
    CLEAN_CSV.unlink(missing_ok=True)   # force re-cleaning
    print(f"Saved {len(out)} rows to {RAW_CSV}")


if __name__ == "__main__":
    main()