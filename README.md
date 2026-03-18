# Ganeshan's Chat API

FastAPI backend for the AI chat feature on [ganeshan.dev](https://ganeshan.dev).

## Features

- **Gemini API Integration**: Powered by Google's Gemini 1.5 Flash
- **Rate Limiting**: 10 requests/minute, 100 requests/day per IP
- **CORS Protection**: Locked to ganeshan.dev and localhost
- **Vercel-Ready**: Serverless deployment configuration included

## Deployment

### Prerequisites

- Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)
- Vercel account

### Deploy to Vercel

1. **Push to GitHub** (if not already done):
   ```bash
   git remote add origin https://github.com/intragalactic-stranger/ganeshan-chat-api.git
   git push -u origin main
   ```

2. **Deploy via Vercel CLI**:
   ```bash
   vercel --prod
   ```

3. **Set Environment Variables** in Vercel Dashboard:
   - `GEMINI_API_KEY`: Your Gemini API key
   - `GEMINI_MODEL`: `gemini-1.5-flash-002` (optional, has default)
   - `SYSTEM_PROMPT`: Custom prompt (optional, has default)

4. **Update Portfolio**: Add the Vercel URL to your portfolio's `NEXT_PUBLIC_CHAT_API_URL` secret

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API key

# Run locally
uvicorn api.index:app --reload --port 8000
```

## API Endpoints

- `GET /` - Health check
- `POST /chat` - Send message, get AI reply

## Tech Stack

- **FastAPI**: Modern Python web framework
- **httpx**: Async HTTP client
- **slowapi**: Rate limiting
- **Gemini API**: Google's generative AI
