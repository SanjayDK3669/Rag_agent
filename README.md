# RAG Agent Backend — Vercel Deployment Guide

## Project Structure

```
vercel-rag-backend/
├── api/
│   ├── __init__.py       # Makes api/ a Python package
│   ├── index.py          # ← Vercel entrypoint (FastAPI app lives here)
│   ├── agent.py          # LangGraph agent logic
│   ├── config.py         # Environment variable loading
│   └── vectorstore.py    # Pinecone vector store helpers
├── vercel.json           # Vercel routing + build config
├── requirements.txt      # Python dependencies
├── .env.example          # Template for environment variables
└── .gitignore
```

## ⚠️ Important Vercel Limitations

| Concern | Detail |
|---|---|
| **Execution timeout** | Hobby plan: 10s max. Pro plan: 60s max. LLM chains can be slow — upgrade to Pro or use streaming carefully. |
| **No persistent filesystem** | `/tmp` works within a single invocation only. PDF uploads use `tempfile` and are deleted after processing — this is fine. |
| **Cold starts** | First request after idle may be slow (~3–5s). |
| **Memory** | 1024 MB max per function. LangGraph + LLM clients are heavy — monitor usage. |
| **No background tasks** | Long-running tasks must complete within the timeout window. |

---

## Deployment Steps

### 1. Install Vercel CLI
```bash
npm install -g vercel
```

### 2. Log in
```bash
vercel login
```

### 3. Set Environment Variables on Vercel Dashboard

Go to: **Vercel Dashboard → Your Project → Settings → Environment Variables**

Add every key from `.env.example`:

- `PINECONE_API_KEY`
- `PINECONE_ENVIRONMENT`
- `PINECONE_INDEX_NAME`
- `GROQ_API_KEY`
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

> **Never commit `.env` to git.** Vercel injects these at runtime automatically.

### 4. Deploy

From the project root:
```bash
vercel --prod
```

Vercel will detect `vercel.json`, install `requirements.txt`, and deploy `api/index.py` as a serverless function.

### 5. Test the deployment
```bash
curl https://your-project.vercel.app/health
# Expected: {"status":"ok"}

curl -X POST https://your-project.vercel.app/chat/ \
  -H "Content-Type: application/json" \
  -d '{"session_id":"test-1","query":"Hello","enable_web_search":false}'
```

---

## Local Development

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in environment variables
cp .env.example .env
# Edit .env with your actual API keys

# 4. Run locally with uvicorn
uvicorn api.index:app --reload --port 8000
```

API docs will be available at: http://localhost:8000/docs

---

## Notes on `MemorySaver` (LangGraph Checkpointing)

The agent uses `MemorySaver` for in-memory conversation checkpointing. On Vercel serverless functions, **each invocation may run in a fresh process**, meaning memory is not shared between requests.

**Solutions (pick one):**
- Use Vercel KV (Redis) + a LangGraph `RedisCheckpointer` for persistent memory.
- Use `RedisSaver` from `langgraph-checkpoint-redis`.
- Accept stateless behavior (memory resets between cold starts) if conversation history isn't critical.
