import os
import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------------------------------------------------------------------------
# Rate limiter — in-memory per instance (good enough for a portfolio)
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute", "100/day"])

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# CORS — only allow your portfolio domain (+ localhost for dev)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ganeshan.dev",
        "http://localhost:3000",
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-002")
SYSTEM_PROMPT  = os.environ.get("SYSTEM_PROMPT", (
    "You are a helpful AI assistant on Ganeshan Arumuganainar's portfolio website. "
    "Ganeshan is an AI Engineer specializing in LLMs, RAG systems, and Cloud AI. "
    "Answer questions about his work, skills, and projects helpfully and concisely. "
    "Keep responses short and friendly."
))


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/")
async def health():
    return {"status": "ok", "model": GEMINI_MODEL}


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------
@app.post("/chat")
@limiter.limit("10/minute;100/day")
async def chat(request: Request):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    message = str(body.get("message", "")).strip()[:4000]
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )

    payload: dict = {
        "contents": [{"role": "user", "parts": [{"text": message}]}],
    }
    if SYSTEM_PROMPT:
        payload["systemInstruction"] = {"parts": [{"text": SYSTEM_PROMPT}]}

    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            res = await client.post(endpoint, json=payload)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Upstream request timed out")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream fetch failed: {str(e)}")

    if not res.is_success:
        raise HTTPException(status_code=502, detail=f"Gemini error: {res.status_code}")

    data = res.json()
    parts = (
        data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
    )
    reply = "\n".join(p.get("text", "") for p in parts if p.get("text")).strip()

    return JSONResponse({"reply": reply or "(no content)"})
