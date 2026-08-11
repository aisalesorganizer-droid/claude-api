# Claude API — Render Deployment Guide

## Files
- `claude_client.py` — Core client (multi-account, file upload, SSE streaming)
- `main.py` — FastAPI wrapper (OpenAI-compatible endpoints)
- `requirements.txt` — Dependencies

## Local Setup (First Time)

1. **Install deps:**
   ```bash
   pip install -r requirements.txt
   python -m playwright install chromium
   ```

2. **Add accounts locally:**
   ```bash
   python claude_client.py --add-account
   ```
   Repeat for each Google account (up to 10).

3. **Encode session for Render:**
   ```bash
   python -c "import base64,json; print(base64.b64encode(open('claude_pool/account_00/session.json','rb').read()).decode())"
   ```
   Copy the output.

## Render Deployment

1. **Create new Web Service** on Render, connect your GitHub repo
2. **Set Build Command:** `pip install -r requirements.txt`
3. **Set Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Add Environment Variables:**
   - `CLAUDE_ACCOUNT` — base64-encoded session.json (single account)
   - `CLAUDE_ACCOUNT_01` — base64-encoded session.json (account 1)
   - `CLAUDE_ACCOUNT_02` — base64-encoded session.json (account 2)
   - ... up to `CLAUDE_ACCOUNT_10`

   For multiple accounts, use `_01` through `_10`. The server auto-detects pool mode.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/status` | Account pool health, expiry, cookie status |
| GET | `/v1/models` | List supported models |
| POST | `/chat` | Simple chat `{message}` → `{response}` |
| POST | `/v1/chat/completions` | OpenAI-compatible (streaming + non-streaming) |
| POST | `/v1/files/upload` | Upload PDF/image → returns `file_id` |

## File Upload Flow

1. `POST /v1/files/upload` with multipart form-data
2. Get `file_id` from response
3. `POST /v1/chat/completions` with header: `X-Claude-File-Ids: <file_id>`

## Important: Session Expiry

Claude sessions are **cookie-based** and expire. There is **no auto-refresh** like Kimi.

- **sessionKey**: ~28 days (but may die sooner on datacenter IPs)
- **cf_clearance**: Hours to days (IP-sensitive)

**When cookies expire:** The API returns `503` with `"Claude session expired"`.

**To fix:** Re-run `--add-account` locally, encode the new session, update the env var on Render.

## Testing

```bash
curl -X POST https://your-app.onrender.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hello"}]}'
```
