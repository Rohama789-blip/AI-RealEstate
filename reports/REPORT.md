# Project Report: EstateIQ

## 1. Problem
Buyers struggle to judge whether a price is fair, find properties that fit a budget, and understand legal and financial steps. EstateIQ combines prediction, recommendation, and a document-grounded assistant in one product.

## 2. Data
3,025 raw listings (synthetic, 8 locations, 4 property types). Cleaning: removed 25 duplicates, filled missing bathrooms (median by bedroom count) and age (median), removed unit-error and IQR outliers, added price per sq ft. Result: 2,896 clean rows. Figures are in `reports/figures/`.

## 3. Model results (20% test split)
| Model | MAE | RMSE | R2 |
|---|---|---|---|
| Linear Regression | 17,312 | 22,575 | 0.911 |
| Decision Tree | 24,819 | 33,605 | 0.8028 |
| Random Forest | 18,117 | 24,049 | 0.899 |

Best model: **Linear Regression**. Linear Regression performs well because the synthetic prices follow near-linear relationships; on real, messier data Random Forest usually wins, so rerun the comparison with your dataset. The saved artifact is a full scikit-learn Pipeline (encoding, scaling, model), so the API passes raw inputs directly.

## 4. Recommender
Hard filters (budget, location, bedrooms, type, parking, garden), then a score: 35% budget fit, 30% value versus area average, 20% newness, 15% bedroom match. Explanations are built from the same components, so they are always consistent with the ranking.

## 5. Assistant and RAG
System prompt defines role, scope, grounding rules, citation of sources, refusal to invent data, and a professional-advice disclaimer. Context per question = top 3 retrieved passages + location price statistics. Documents can be uploaded in the UI and are indexed immediately.

## 6. Architecture
Browser (Jinja2 pages + fetch) -> FastAPI -> predictor / recommender / RAG + Groq LLM. Data and model files live on disk.

## 7. Limitations and next steps
Synthetic data; no authentication; recommendations use a static dataset rather than live listings. Next: real dataset, map view, saved searches, model explainability (feature importance), deployment, evaluation of RAG answer quality.
