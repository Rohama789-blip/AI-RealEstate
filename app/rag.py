"""RAG knowledge base. Uses ChromaDB + Sentence Transformers when available;
falls back to TF-IDF retrieval so the app still runs without them."""
import re
from pathlib import Path
from .data import ROOT

DOCS_DIR = ROOT / "docs"
UPLOAD_DIR = DOCS_DIR / "uploads"
CHROMA_DIR = ROOT / "chroma_db"


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, source: str, size: int = 700) -> list[dict]:
    """Split on paragraphs, then pack paragraphs into ~size character chunks."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if cur and len(cur) + len(p) > size:
            chunks.append(cur); cur = p
        else:
            cur = f"{cur}\n\n{p}".strip()
    if cur:
        chunks.append(cur)
    return [{"text": c, "source": source} for c in chunks]


class KnowledgeBase:
    def __init__(self):
        self.chunks: list[dict] = []
        self.backend = "none"
        self._collection = None
        self._embedder = None
        self._tfidf = None
        self.reindex()

    def reindex(self):
        self.chunks = []
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(DOCS_DIR.glob("*.md")) + sorted(DOCS_DIR.glob("*.txt")) + \
                sorted(UPLOAD_DIR.glob("*.*"))
        for f in files:
            if f.suffix.lower() in {".md", ".txt", ".pdf"}:
                self.chunks += chunk_text(read_file(f), f.name)
        if not self.chunks:
            return
        if not self._build_chroma():
            self._build_tfidf()

    def _build_chroma(self) -> bool:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            self._embedder = self._embedder or SentenceTransformer("all-MiniLM-L6-v2")
            client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            try:
                client.delete_collection("real_estate")
            except Exception:
                pass
            self._collection = client.create_collection("real_estate")
            texts = [c["text"] for c in self.chunks]
            self._collection.add(
                ids=[str(i) for i in range(len(texts))], documents=texts,
                embeddings=self._embedder.encode(texts).tolist(),
                metadatas=[{"source": c["source"]} for c in self.chunks])
            self.backend = "chromadb"
            return True
        except Exception as e:  # missing package, offline model download, etc.
            print(f"[rag] ChromaDB backend unavailable ({type(e).__name__}); using TF-IDF fallback")
            return False

    def _build_tfidf(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._matrix = self._tfidf.fit_transform([c["text"] for c in self.chunks])
        self.backend = "tfidf"

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.chunks:
            return []
        if self.backend == "chromadb":
            res = self._collection.query(query_embeddings=self._embedder.encode([query]).tolist(), n_results=k)
            return [{"text": t, "source": m["source"], "score": round(1 - d, 3)}
                    for t, m, d in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(self._tfidf.transform([query]), self._matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": round(float(sims[i]), 3)} for i in top if sims[i] > 0]


_kb = None
def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
