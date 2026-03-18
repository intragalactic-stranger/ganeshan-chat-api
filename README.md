# Ganeshan's Chat API

FastAPI backend for the AI chat feature on [ganeshan.dev](https://ganeshan.dev).

## Features

- **Amazon Nova 2 Lite**: Powered by AWS Bedrock Nova Lite model
- **Rate Limiting**: 10 requests/minute, 100 requests/day per IP
- **CORS Protection**: Locked to ganeshan.dev and localhost
- **Vercel-Ready**: Serverless deployment configuration included

## Deployment

### Prerequisites

- AWS credentials (Access Key ID and Secret Access Key) with Bedrock access
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
   - `AWS_ACCESS_KEY_ID`: Your AWS access key ID
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key
   - `AWS_REGION`: `us-east-1` (or your preferred region)
   - `MODEL_ID`: `us.amazon.nova-lite-v1:0` (optional, has default)
   - `SYSTEM_PROMPT`: Custom prompt (optional, has default)

4. **Update Portfolio**: Add the Vercel URL to your portfolio's `NEXT_PUBLIC_CHAT_API_URL` secret

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your AWS credentials

# Run locally
uvicorn api.index:app --reload --port 8000
```

## API Endpoints

- `GET /` - Health check
- `POST /chat` - Send message, get AI reply

## Tech Stack

- **FastAPI**: Modern Python web framework
- **boto3**: AWS SDK for Python
- **slowapi**: Rate limiting
- **Amazon Nova 2 Lite**: AWS Bedrock generative AI
