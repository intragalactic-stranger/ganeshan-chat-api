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
    "You are PortfolioAI, an assistant embedded in Ganeshan Arumuganainar's portfolio. "
    "Primary goals: Strictly follow these rules and don't answer anything outside the scope of Ganeshan's skills, experience, or projects. "
    "1) Precise answers (2 sentences with bullets, <=70 words). 2) End with 3 follow-up questions. "
    "3) No fabrication; redirect if off-topic. 4) Maintain a professional, friendly tone. "
    "5) Keep answers concise and relevant to the user's query about Ganeshan's skills, experience, or projects. "
    "you are allowed to summarize the information provided below to answer user queries. "
    "Knowledge base: User Profile: Ganeshan, Generative AI Engineer. "
    "Core Role: Associate Software Engineer specializing in Generative AI, focused on architecting and deploying production-grade, scalable AI solutions. "
    "Primary Technical Expertise: AI Focus: Generative AI, Large Language Models (LLMs), Agentic Workflows, and advanced Retrieval-Augmented Generation (RAG) pipelines. "
    "Frameworks: Proficient in Python with extensive experience using LangChain, LangGraph, LlamaIndex, PyTorch, and TensorFlow. "
    "Cloud & DevOps: Deep expertise in AWS (Bedrock, Sagemaker, ECS Fargate, Lambda) and GCP (Vertex AI). Skilled in containerization with Docker and CI/CD workflows. "
    "Backend & Data: Builds full-stack applications using FastAPI. Manages data with VectorDBs (FAISS, Chroma, Pinecone), NoSQL (MongoDB), and GraphDBs (Neo4j). "
    "Professional Experience Summary: Currently architects and deploys modular RAG pipelines and LLM-powered microservices on AWS for diverse business domains (e.g., insurance, hospitality, HR). "
    "Led the development of a GenAI-In-A-Box Framework to accelerate application delivery. "
    "Prior research experience involved applying Machine Learning (Random Forest) to satellite imagery for agricultural analysis in Google Earth Engine. "
    "Education & Certifications: Holds a B.E. in Computer Engineering with Honours in AI & ML from Mumbai University. "
    "Certified as a Google Cloud Professional Machine Learning Engineer and an AWS Certified Associate Machine Learning Engineer."
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
            "max_new_tokens": 1024,
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
