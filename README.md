# EstateIQ: AI Real Estate Intelligence Platform

A web app that estimates property prices, recommends properties with explanations, and answers real estate questions using generative AI and RAG.

| Feature | How it works |
|---|---|
| Data analysis | Pandas/NumPy cleaning, Matplotlib/Seaborn charts (`scripts/eda.py`) |
| Price prediction | scikit-learn pipeline comparing Linear Regression, Decision Tree, Random Forest (`scripts/train_model.py`) |
| Recommendations | Filters + weighted score; explanations are generated from the score components (`app/recommender.py`) |
| AI assistant | Groq LLM with a role-based system prompt, market statistics, and retrieved context (`app/chat.py`) |
| RAG | Chunk, embed (Sentence Transformers), store in ChromaDB, retrieve top 3; TF-IDF fallback (`app/rag.py`) |
| Web app | FastAPI + Jinja2 pages for predict, recommend, and chat (`app/main.py`) |

## Project structure
```
app/            FastAPI app, predictor, recommender, RAG, chat, templates, CSS
data/           properties_raw.csv, properties_clean.csv
docs/           knowledge base documents for RAG (uploads go to docs/uploads)
models/         saved model + metrics.json
reports/        figures from the analysis, REPORT.md
scripts/        generate_dataset.py, eda.py, train_model.py
screenshots/    add your app screenshots here
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # add your GROQ_API_KEY
python scripts/eda.py            # clean data + create charts
python scripts/train_model.py    # train, compare, save the model
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000. Without a `GROQ_API_KEY`, the chat runs in retrieval-only demo mode.

## Using a real dataset
`data/properties_raw.csv` is synthetic so the project runs anywhere. To use a real dataset, replace it with a CSV that has these columns: `location, property_type, area_sqft, bedrooms, bathrooms, age_years, parking, garden, listing_year, price` (rename or adapt columns in `app/data.py` if yours differ), then rerun the two scripts.

## API
`POST /api/predict`, `POST /api/recommend`, `POST /api/chat`, `POST /api/upload-doc`, `GET /api/health`. Interactive docs at `/docs`.

## Limitations
Predictions are estimates from sample data, not valuations. The assistant is not legal or financial advice.
