"""Rule-based recommender: hard filters, weighted score, and a human-readable explanation."""
import pandas as pd
from .data import load_clean

def recommend(budget: float, location: str = "Any", bedrooms: int = 0, property_type: str = "Any",
              need_parking: bool = False, need_garden: bool = False, top_n: int = 5) -> list[dict]:
    df = load_clean()
    loc_avg = df.groupby("location").price_per_sqft.mean()

    d = df[df.price <= budget].copy()
    if location != "Any":
        d = d[d.location == location]
    if bedrooms:
        d = d[d.bedrooms >= bedrooms]
    if property_type != "Any":
        d = d[d.property_type == property_type]
    if need_parking:
        d = d[d.parking >= 1]
    if need_garden:
        d = d[d.garden == 1]
    if d.empty:
        return []

    d["budget_fit"] = 1 - (budget - d.price) / budget                       # closer to budget = more home for the money
    d["value"] = ((d.location.map(loc_avg) - d.price_per_sqft) / d.location.map(loc_avg)).clip(-0.3, 0.3) / 0.3
    d["newness"] = 1 - d.age_years / 40
    d["bed_match"] = (d.bedrooms == bedrooms).astype(int) if bedrooms else 0.5
    d["score"] = 0.35 * d.budget_fit + 0.30 * (d.value + 1) / 2 + 0.20 * d.newness + 0.15 * d.bed_match

    results = []
    for idx, r in d.sort_values("score", ascending=False).head(top_n).iterrows():
        why = []
        saving = (budget - r.price) / budget * 100
        why.append(f"Priced {saving:.0f}% under your budget" if saving >= 1 else "Right at your budget")
        diff = (loc_avg[r.location] - r.price_per_sqft) / loc_avg[r.location] * 100
        if diff >= 3:
            why.append(f"{diff:.0f}% cheaper per sq ft than the {r.location} average")
        elif diff <= -3:
            why.append(f"{abs(diff):.0f}% above the {r.location} average per sq ft, so expect premium features")
        if r.age_years <= 10:
            why.append(f"Relatively new ({int(r.age_years)} years old)")
        elif r.age_years >= 25:
            why.append(f"Older building ({int(r.age_years)} years), so budget for maintenance")
        if bedrooms and r.bedrooms == bedrooms:
            why.append(f"Exactly {bedrooms} bedrooms as requested")
        if need_garden and r.garden:
            why.append("Has a garden")
        if need_parking and r.parking:
            why.append(f"Includes {int(r.parking)} parking space(s)")
        results.append({
            "id": int(idx), "location": r.location, "property_type": r.property_type,
            "area_sqft": int(r.area_sqft), "bedrooms": int(r.bedrooms), "bathrooms": int(r.bathrooms),
            "age_years": int(r.age_years), "parking": int(r.parking), "garden": bool(r.garden),
            "price": int(r.price), "score": round(float(r.score) * 100), "why": why,
        })
    return results
