"""Create a synthetic property dataset with realistic quirks (missing values, duplicates, outliers).
Replace data/properties_raw.csv with a real dataset (e.g. from Kaggle) if you prefer;
just keep the same column names or adjust app/data.py."""
import numpy as np, pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 3000
LOCATIONS = {"Downtown": 1.45, "Riverside": 1.25, "Greenfield": 1.10, "Lakeview": 1.30,
             "Old Town": 0.85, "Hillcrest": 1.15, "Tech Park": 1.20, "Seaside": 1.50}
TYPES = {"Apartment": 0.90, "House": 1.15, "Villa": 1.60, "Townhouse": 1.00}

loc = rng.choice(list(LOCATIONS), N)
ptype = rng.choice(list(TYPES), N, p=[0.40, 0.30, 0.10, 0.20])
area = rng.normal(1500, 500, N).clip(450, 5000).round()
bedrooms = np.clip(np.round(area / 600 + rng.normal(0, 0.7, N)), 1, 6).astype(int)
bathrooms = np.clip(bedrooms - rng.integers(0, 2, N), 1, 5)
age = rng.integers(0, 40, N)
parking = rng.integers(0, 3, N)
garden = rng.integers(0, 2, N) * np.isin(ptype, ["House", "Villa", "Townhouse"])
year = rng.integers(2022, 2026, N)

base = (area * 95 * np.array([LOCATIONS[l] for l in loc]) * np.array([TYPES[t] for t in ptype])
        * (1 - 0.008 * age) * (1 + 0.06 * (year - 2022))
        + bedrooms * 4000 + bathrooms * 3000 + parking * 6000 + garden * 9000)
price = (base * rng.lognormal(0, 0.08, N)).round(-2)

df = pd.DataFrame(dict(location=loc, property_type=ptype, area_sqft=area, bedrooms=bedrooms,
                       bathrooms=bathrooms, age_years=age, parking=parking, garden=garden,
                       listing_year=year, price=price))

# Add messiness so the cleaning step has real work to do
df.loc[rng.choice(N, 60, replace=False), "bathrooms"] = np.nan
df.loc[rng.choice(N, 90, replace=False), "age_years"] = np.nan
df.loc[rng.choice(N, 10, replace=False), "area_sqft"] = 40000      # unit mistakes
df = pd.concat([df, df.sample(25, random_state=1)], ignore_index=True)  # duplicates

out = Path(__file__).resolve().parent.parent / "data" / "properties_raw.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
