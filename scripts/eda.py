"""Clean the data and produce analysis charts in reports/figures."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from app.data import RAW_CSV, CLEAN_CSV, ROOT, clean_data

FIG = ROOT / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(RAW_CSV)
print("Raw shape:", raw.shape); print("Missing values:\n", raw.isna().sum()[raw.isna().sum() > 0])
df = clean_data(raw); df.to_csv(CLEAN_CSV, index=False)
print("Clean shape:", df.shape); print(df.describe().round(1).T)
sns.set_theme(style="whitegrid", palette="crest")

def save(name): plt.tight_layout(); plt.savefig(FIG / name, dpi=130); plt.close()

plt.figure(figsize=(7, 4)); sns.histplot(df.price, bins=40); plt.title("Price distribution"); save("price_distribution.png")
plt.figure(figsize=(7, 5)); sns.scatterplot(data=df, x="area_sqft", y="price", hue="property_type", alpha=.5)
plt.title("Area vs price"); save("area_vs_price.png")
order = df.groupby("location").price_per_sqft.mean().sort_values(ascending=False).index
plt.figure(figsize=(8, 4)); sns.barplot(data=df, x="location", y="price_per_sqft", order=order, errorbar=None)
plt.xticks(rotation=30); plt.title("Average price per sq ft by location"); save("location_comparison.png")
plt.figure(figsize=(7, 4)); sns.boxplot(data=df, x="bedrooms", y="price"); plt.title("Price by bedrooms"); save("bedrooms_vs_price.png")
plt.figure(figsize=(7, 4)); sns.lineplot(data=df, x="listing_year", y="price_per_sqft", errorbar=None, marker="o")
plt.xticks(sorted(df.listing_year.unique())); plt.title("Price per sq ft trend"); save("price_trend.png")
plt.figure(figsize=(7, 5.5)); sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="crest")
plt.title("Correlation heatmap"); save("correlation_heatmap.png")
print("Charts saved to", FIG)
