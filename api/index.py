import os
import json
import boto3
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
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("MODEL_ID", "us.amazon.nova-lite-v1:0")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", (
    "You are PortfolioAI helping visitors learn about Ganeshan Arumuganainar. "
    "RULES: 1) MAX 35 words. 2) For 'hi/hello': Just say 'Hi! Ask me about Ganeshan's AI work.' (8 words). "
    "3) Be direct - no fluff. 4) Use simple dashes (-) for lists, NOT asterisks or bold. "
    "5) End with 1 short question. 6) Third person only (he/his, not I/my). "
    "INFO: GenAI Engineer building RAG pipelines & LLM systems on AWS. "
    "Skills: LangChain, LlamaIndex, Bedrock, FastAPI, VectorDBs. "
    "Projects: GenAI-In-A-Box framework, RAG for insurance/HR/hospitality, satellite ML. "
    "Education: B.E. Computer (AI/ML). Certs: GCP ML Pro, AWS ML Associate."
))

# Initialize Bedrock client
bedrock_runtime = None
if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/")
async def health():
    return {"status": "ok", "model": MODEL_ID, "region": AWS_REGION}


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------
@app.post("/chat")
@limiter.limit("10/minute;100/day")
async def chat(request: Request):
    if not bedrock_runtime:
        raise HTTPException(status_code=500, detail="AWS credentials not configured")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    message = str(body.get("message", "")).strip()[:4000]
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Prepare the request for Amazon Nova
    messages = [{"role": "user", "content": [{"text": message}]}]
    
    payload = {
        "messages": messages,
        "inferenceConfig": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }
    
    # Add system prompt if configured
    if SYSTEM_PROMPT:
        payload["system"] = [{"text": SYSTEM_PROMPT}]

    try:
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )
        
        response_body = json.loads(response["body"].read())
        
        # Extract text from Nova response
        reply = ""
        if "output" in response_body and "message" in response_body["output"]:
            content = response_body["output"]["message"].get("content", [])
            for item in content:
                if "text" in item:
                    reply += item["text"]
        
        return JSONResponse({"reply": reply.strip() or "(no content)"})
        
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Bedrock API error: {str(e)}")
