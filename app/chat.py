"""LLM assistant: prompt engineering + RAG context + market statistics."""
import os
from dotenv import load_dotenv
from .data import ROOT, load_clean
from .rag import get_kb

load_dotenv(ROOT / ".env")

SYSTEM_PROMPT = """You are EstateIQ, an AI real estate advisor inside a property platform.

Role and scope:
- Answer property questions: buying guidance, market trends, area comparisons, legal basics, investment tips.
- Summarise properties when the user provides details.

Rules:
1. Ground answers in the KNOWLEDGE CONTEXT and MARKET DATA below. When you use the context, name the source file in brackets, e.g. [legal_checklist.md].
2. If the context does not cover the question, say so, then give general guidance and mark it as general knowledge.
3. Never invent listings, prices, laws, or statistics. Use only numbers present in MARKET DATA.
4. You are not a lawyer or financial advisor. For legal or financial decisions, recommend a professional.
5. Be concise: short paragraphs or a few bullets. End buying guidance with one clear next step.
6. Currency amounts are in the dataset's currency unit shown as $."""


def market_summary() -> str:
    df = load_clean()
    by_loc = df.groupby("location").agg(avg_price=("price", "mean"), ppsf=("price_per_sqft", "mean"))
    lines = [f"- {loc}: avg price ${r.avg_price:,.0f}, ${r.ppsf:,.0f}/sq ft" for loc, r in by_loc.iterrows()]
    trend = df.groupby("listing_year").price_per_sqft.mean()
    lines.append("- Price per sq ft by year: " + ", ".join(f"{y}: ${v:,.0f}" for y, v in trend.items()))
    return "\n".join(lines)


def build_messages(question: str, history: list[dict], chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks) or "(no relevant documents found)"
    system = f"{SYSTEM_PROMPT}\n\nMARKET DATA:\n{market_summary()}\n\nKNOWLEDGE CONTEXT:\n{context}"
    return [{"role": "system", "content": system}, *history[-6:], {"role": "user", "content": question}]


def answer(question: str, history: list[dict] | None = None) -> dict:
    chunks = get_kb().search(question, k=3)
    sources = sorted({c["source"] for c in chunks})
    key = os.getenv("GROQ_API_KEY")
    if not key:
        # Graceful fallback so the demo works without an API key
        body = "\n\n".join(f"**{c['source']}**: {c['text'][:400]}" for c in chunks) or \
               "I couldn't find anything relevant in the knowledge base."
        return {"answer": body, "sources": sources, "mode": "retrieval-only"}
    try:
        from groq import Groq
        resp = Groq(api_key=key).chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=build_messages(question, history or [], chunks), temperature=0.3, max_tokens=1500)
        return {"answer": resp.choices[0].message.content, "sources": sources, "mode": "llm"}
    except Exception as e:
        return {"answer": f"The AI service had a problem ({type(e).__name__}). Please try again.",
                "sources": sources, "mode": "error"}
