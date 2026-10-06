from pathlib import Path
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from . import chat, predictor, recommender
from .data import load_clean
from .rag import UPLOAD_DIR, get_kb

BASE = Path(__file__).resolve().parent
app = FastAPI(title="EstateIQ - AI Real Estate Intelligence Platform")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")


def options() -> dict:
    df = load_clean()
    return {"locations": sorted(df.location.unique()), "types": sorted(df.property_type.unique())}


# ---------- Pages ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    df = load_clean()
    stats = {"listings": len(df), "avg_price": int(df.price.mean()), "locations": df.location.nunique()}
    return templates.TemplateResponse(request, "home.html", {"page": "home", "stats": stats, "m": predictor.metrics(), **options()})

@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse(request, "predict.html", {"page": "predict", **options()})

@app.get("/recommend", response_class=HTMLResponse)
def recommend_page(request: Request):
    return templates.TemplateResponse(request, "recommend.html", {"page": "recommend", **options()})

@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return templates.TemplateResponse(request, "chat.html", {"page": "chat"})


# ---------- API ----------
class PredictIn(BaseModel):
    location: str
    property_type: str
    area_sqft: float = Field(gt=200, lt=10000)
    bedrooms: int = Field(ge=1, le=10)
    bathrooms: int = Field(ge=1, le=10)
    age_years: int = Field(ge=0, le=100)
    parking: int = Field(ge=0, le=10)
    garden: int = Field(ge=0, le=1)

@app.post("/api/predict")
def api_predict(body: PredictIn):
    try:
        return predictor.predict_price(body.model_dump())
    except RuntimeError as e:
        raise HTTPException(503, str(e))


class RecommendIn(BaseModel):
    budget: float = Field(gt=0)
    location: str = "Any"
    bedrooms: int = Field(0, ge=0, le=10)
    property_type: str = "Any"
    need_parking: bool = False
    need_garden: bool = False

@app.post("/api/recommend")
def api_recommend(body: RecommendIn):
    return {"results": recommender.recommend(**body.model_dump())}


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=1500)
    history: list[dict] = []

@app.post("/api/chat")
def api_chat(body: ChatIn):
    history = [h for h in body.history if h.get("role") in {"user", "assistant"} and isinstance(h.get("content"), str)]
    return chat.answer(body.message, history)


@app.post("/api/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".md", ".txt", ".pdf"}:
        raise HTTPException(400, "Upload a .pdf, .txt or .md file.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    (UPLOAD_DIR / Path(file.filename).name).write_bytes(await file.read())
    kb = get_kb(); kb.reindex()
    return {"status": "indexed", "chunks": len(kb.chunks), "backend": kb.backend}


@app.get("/api/health")
def health():
    kb = get_kb()
    return {"status": "ok", "rag_backend": kb.backend, "chunks": len(kb.chunks),
            "llm": "groq" if chat.os.getenv("GROQ_API_KEY") else "demo-mode"}
