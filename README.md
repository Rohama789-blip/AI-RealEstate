# 🏡 EstateIQ — AI Real Estate Intelligence Platform

An AI-powered real estate application that predicts property prices, recommends suitable properties, and answers real estate questions using Generative AI and RAG.

## ✨ Features

- 🏠 **Price Prediction** — Estimate property prices using Machine Learning.
- 🎯 **Property Recommendations** — Get personalized recommendations with explanations.
- 🤖 **AI Real Estate Advisor** — Ask questions using the Groq LLM.
- 📚 **RAG-Based Search** — Retrieve relevant information from documents.
- 📊 **Data Analysis** — Explore property trends and compare ML models.
- 🌐 **Interactive Web App** — Simple interface built with FastAPI and Jinja2.

## 🛠️ Tech Stack

- 🐍 Python
- ⚡ FastAPI
- 🧠 Scikit-learn
- 📊 Pandas & NumPy
- 🔎 ChromaDB
- 🔤 Sentence Transformers
- 🤖 Groq API
- 🎨 HTML, CSS & Jinja2

## 🚀 Installation

**1. Clone the repository**

```bash
git clone https://github.com/Rohama789-blip/AI-RealEstate.git
cd EstateIQ
```

**2. Create a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

Copy `.env.example` to `.env` and add your Groq API key:

```env
GROQ_API_KEY=your_groq_api_key
```

**5. Prepare the data and train the model**

```bash
python scripts/eda.py
python scripts/train_model.py
```

**6. Run the application**

```bash
uvicorn app.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

## 🔗 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | ❤️ Health check |
| POST | `/api/predict` | 🏠 Predict property prices |
| POST | `/api/recommend` | 🎯 Recommend properties |
| POST | `/api/chat` | 🤖 AI chat assistant |
| POST | `/api/upload-doc` | 📄 Upload a document |

Interactive API documentation: `http://127.0.0.1:8000/docs`

## ⚠️ Limitations

Predictions are estimates based on sample data, not professional property valuations. AI responses are for informational purposes and should not be treated as legal or financial advice.

