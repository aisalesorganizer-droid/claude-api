# claude-api

A self-hosted FastAPI server that reverse-engineers the **claude.ai browser session** to expose an OpenAI-compatible API. Supports multi-account pooling, automatic rotation, file upload, and streaming — all without using Anthropic's official API or API keys.

---

## How It Works

Claude.ai's web client communicates with `claude.ai` using **browser session cookies** (not API keys). This project captures those cookies via a Playwright login flow, stores them per-account, and replays them using `curl_cffi` with Chrome impersonation to make authenticated requests on your behalf.

### Core flow per request

```
POST /v1/chat/completions
    → pick a healthy account from the pool
    → POST /api/organizations/{org_id}/chat_conversations   (register new conversation UUID)
    → POST /api/organizations/{org_id}/chat_conversations/{conv_id}/completion  (stream response)
    → parse SSE stream → yield OpenAI-format chunks
```

Every conversation is disposable — a new `conv_id` is registered per session, and the full message history is sent in the completion body each time. No server-side conversation state is maintained.

---

## File Structure

```
.
├── claude_client.py     # Core engine: session management, signing, streaming, CLI
├── main.py              # FastAPI wrapper: OpenAI-compatible endpoints, account pool, auth
├── requirements.txt     # Python dependencies
├── .env                 # Account pool (base64-encoded sessions, one per CLAUDE_ACCOUNT_XX)
└── claude_pool/         # Local session storage (created on first --add-account run)
    └── account_N/
        └── session.json # Cookies, org_id, device_id, expiry
```

---

## Architecture

### `claude_client.py` — The Engine

**Session model:** Each account slot is a directory under `claude_pool/`. The `session.json` stores:
- `cookies` — all browser cookies: `sessionKey`, `cf_clearance`, `__cf_bm`, `anthropic-device-id`, `lastActiveOrg`, etc.
- `org_id` — the Anthropic organization UUID (required for every API call)
- `device_id` — the `anthropic-device-id` cookie value
- `anonymous_id` — from `localStorage.ajs_anonymous_id` (used as `anthropic-anonymous-id` header)
- `expires` — ISO8601 timestamp of `sessionKey` expiry

**HTTP client:** `curl_cffi` with `chrome131` impersonation. All requests carry the full browser header set:
```
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/151.0.0.0
sec-ch-ua, sec-fetch-*, origin, referer
anthropic-client-platform: web_claude_ai
anthropic-client-sha: 2aa88381b74d...
anthropic-client-version: 1.0.0
```

**Login flow (`playwright_login`):** Opens a persistent Chromium context → navigates to `claude.ai/login` → waits for the user to complete Google OAuth → captures cookies and `localStorage` values → calls `/api/organizations` to confirm the org UUID.

**Conversation lifecycle (`_stream_on_slot`):**
1. If no `conv_id` exists for the slot, registers one via `POST /api/organizations/{org}/chat_conversations` with `is_temporary: true`
2. Generates UUID7 values for `human_message_uuid` and `assistant_message_uuid`
3. Sends `POST .../completion` with `stream=True`
4. Parses SSE events: `content_block_delta` (text chunks), `message_stop`, `message_limit`, `error`
5. On success, stores `last_asst_uuid` for multi-turn continuity (`parent_message_uuid`)
6. On 401 → clears session, removes from `_slot_states`, raises for rotation

**Pool rotation (`_stream`):**
- Reads current rotation index from `claude_pool/rotation.json`
- Tries accounts round-robin; on 401/429/expired errors, skips to next
- Advances index on success

**Model resolution (`resolve_model`):** Maps display names → `(real_model_slug, thinking_mode, effort)`. Every model variant is explicitly declared in `MODEL_CONFIGS`.

**UUID7 (`uuid7`):** Time-based UUID with correct version/variant bits — required by the claude.ai API for message IDs.

---

### `main.py` — The FastAPI Wrapper

**Startup sequence:**
1. Reads `CLAUDE_ACCOUNT_01`–`CLAUDE_ACCOUNT_10` env vars (base64-encoded `session.json` JSON)
2. Decodes and writes them to a `tempfile.mkdtemp()` directory
3. Patches `claude_client.POOL_DIR` to point at that temp dir
4. Runs a pre-flight `GET /api/organizations` check per account to detect Cloudflare blocks
5. Marks accounts with 3+ failures as unhealthy (5-minute cooldown)
6. Aborts startup if zero healthy accounts

**Per-request flow:**
- `_fresh_slot_state(label)` — creates a new `_SlotState` per request (prevents shared-state corruption across concurrent requests)
- `_stream_claude()` — tries forced account first (for file attachment org-scoping), then rotates through healthy accounts
- `_to_openai_stream()` — converts Claude SSE chunks to OpenAI `chat.completion.chunk` SSE format

**Prompt assembly (OpenAI → Claude web format):**
```
[system message]  →  <s>\n{content}\n</s>

[user message]    →  \n\nHuman: {content}
[assistant msg]   →  \n\nAssistant: {content}

[trailing marker] →  \n\nAssistant:   (only if last message is from user)
```

**File upload scoping:** When a file is uploaded via `/v1/files/upload`, the account label that performed the upload is stored in `_file_account_map[file_uuid]`. On subsequent `/v1/chat/completions` calls with `X-Claude-File-Ids` header, that same account is tried first — because uploaded files are scoped to the organization that owns them.

**Auth gate:** If `CLAUDE_API_KEY` env var is set, all mutation endpoints require `Authorization: Bearer <key>`. Skipped if not set.

**Health tracking (`_ClaudeAccount`):** Thread-safe per-account failure counter. After 3 failures within 5 minutes, the account is skipped in rotation. Auto-resets after the cooldown window.

---

## Supported Models

Every model is mapped to a `(slug, thinking_mode, effort)` triple. Thinking mode and effort are sent as-is in the completion body.

| Display Name | Real Slug | Thinking | Effort |
|---|---|---|---|
| `claude-haiku-4-5` | `claude-haiku-4-5-20251001` | off | — |
| `claude-haiku-4-5 (Extended)` | `claude-haiku-4-5-20251001` | extended | — |
| `claude-sonnet-4-6 Low` | `claude-sonnet-4-6` | off | low |
| `claude-sonnet-4-6 Medium` | `claude-sonnet-4-6` | off | medium |
| `claude-sonnet-4-6 High` | `claude-sonnet-4-6` | off | high |
| `claude-sonnet-4-6 Max` | `claude-sonnet-4-6` | off | max |
| `claude-sonnet-4-6 Low + Think` | `claude-sonnet-4-6` | auto | low |
| `claude-sonnet-4-6 Medium + Think` | `claude-sonnet-4-6` | auto | medium |
| `claude-sonnet-4-6 High + Think` | `claude-sonnet-4-6` | auto | high |
| `claude-sonnet-4-6 Max + Think` | `claude-sonnet-4-6` | auto | max |
| `claude-sonnet-5 Low` | `claude-sonnet-5` | off | low |
| `claude-sonnet-5 Medium` | `claude-sonnet-5` | off | medium |
| `claude-sonnet-5 High` | `claude-sonnet-5` | off | high |
| `claude-sonnet-5 XHigh` | `claude-sonnet-5` | off | xhigh |
| `claude-sonnet-5 Max` | `claude-sonnet-5` | off | max |
| `claude-sonnet-5 [effort] + Think` | `claude-sonnet-5` | auto | varies |
| `claude-opus-4-6` | `claude-opus-4-6` | extended | medium |
| `claude-opus-4-7` | `claude-opus-4-7` | auto | xhigh |
| `claude-opus-4-8` | `claude-opus-4-8` | auto | high |
| `claude-opus-5` | `claude-opus-5` | auto | high |
| `claude-fable-5` | `claude-fable-5` | auto | high |

Pass any display name as `model` in the OpenAI request body. Unknown names fall back to `claude-sonnet-4-6` with `effort: high`.

---

## Endpoints

| Method | Path | Auth Required | Description |
|---|---|---|---|
| `GET` | `/` | No | Health check — returns `ready`, `init_error` |
| `GET` | `/status` | No | Per-account health: days remaining, `sessionKey` present, `cf_clearance` present, healthy flag, fail count |
| `GET` | `/v1/models` | No | Lists all supported model display names in OpenAI format |
| `POST` | `/chat` | Yes (if key set) | Simple chat: `{"message": "...", "model": "..."}` → `{"response": "..."}` |
| `POST` | `/v1/chat/completions` | Yes (if key set) | OpenAI-compatible, streaming and non-streaming |
| `POST` | `/v1/files/upload` | Yes (if key set) | Multipart upload → `{"file_id": "...", "filename": "...", "status": "ready"}` |

---

## File Upload Flow

Files are org-scoped — the account that uploads a file must also be the one that sends the chat request referencing it.

```bash
# 1. Upload
curl -X POST https://your-app.railway.app/v1/files/upload \
  -H "Authorization: Bearer $CLAUDE_API_KEY" \
  -F "file=@document.pdf"
# → {"file_id": "abc-123-...", "filename": "document.pdf", "status": "ready"}

# 2. Chat with attachment
curl -X POST https://your-app.railway.app/v1/chat/completions \
  -H "Authorization: Bearer $CLAUDE_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Claude-File-Ids: abc-123-..." \
  -d '{"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "Summarize this document"}]}'
```

The server automatically routes the chat request to the same account that uploaded the file.

---

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

### 2. Add accounts

```bash
python claude_client.py --add-account
```

A browser window opens. Sign in with a Google account linked to Claude Pro/Max. The window closes automatically after login. Repeat for each account (up to 10). Each session is saved to `claude_pool/account_N/session.json` and base64-appended to `.env`.

### 3. Run locally

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or use the CLI directly:

```bash
python claude_client.py "hello"          # one-shot
python claude_client.py                  # interactive REPL
python claude_client.py --pool           # show pool status
```

---

## Railway / Render Deployment

Sessions captured locally must be injected as environment variables because the server has no browser for Playwright login.

### 1. Encode a session

```bash
python -c "import base64, json; print(base64.b64encode(open('claude_pool/account_1/session.json','rb').read()).decode())"
```

Or after running `--add-account`, the value is auto-appended to `.env` under the key `CLAUDE_ACCOUNT_01`, `CLAUDE_ACCOUNT_02`, etc.

### 2. Set environment variables on Railway

| Variable | Value |
|---|---|
| `CLAUDE_ACCOUNT_01` | base64-encoded `session.json` for account 1 |
| `CLAUDE_ACCOUNT_02` | base64-encoded `session.json` for account 2 |
| ... | up to `CLAUDE_ACCOUNT_10` |
| `CLAUDE_API_KEY` | any secret string — gates all mutation endpoints |
| `CLAUDE_PROXY` | `socks5://...` if needed, or omit |

### 3. Deploy

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

`CLAUDE_ALLOW_BROWSER_LOGIN` must NOT be set to `1` on the server — it defaults to off, preventing accidental Playwright launches in production.

---

## Session Expiry and Maintenance

Claude sessions are cookie-based. There is **no auto-refresh**.

| Cookie | Lifetime | Notes |
|---|---|---|
| `sessionKey` | ~28 days | May die sooner on datacenter IPs |
| `cf_clearance` | Hours to days | **IP-bound** — a cookie captured on your home IP will fail from a Railway/Render IP |
| `__cf_bm` | Minutes | Short-lived Cloudflare bot management token |

### When sessions expire

The `/status` endpoint shows `days_remaining` and `healthy` per account. When an account hits 401, `claude_client.py` clears its `session.json` and removes it from the in-memory slot states. The server returns `503` with `"Claude session expired"`.

### To refresh

Re-run `--add-account` locally, encode the new session, update the env var on Railway, and redeploy (or restart the service).

### Cloudflare on datacenter IPs

Railway and Render IPs are datacenter ranges. Cloudflare may challenge requests even with a valid `cf_clearance` cookie, because `cf_clearance` is IP-bound. If the startup pre-flight returns `"Just a moment"` (HTTP 403), the account is marked unhealthy at startup. Options:
- Use a residential proxy (`CLAUDE_PROXY=socks5://...`)
- Regenerate cookies from the same IP as the server (not practical for most deployments)
- Accept that some accounts will be blocked and rely on the healthy ones in the pool

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `CLAUDE_MODEL` | `claude-sonnet-4-6` | Default model slug |
| `CLAUDE_PROXY` | _(auto-detect)_ | Proxy URL. `"none"` to disable. Auto-detects WARP on ports 40000/1080/8080 |
| `CLAUDE_SSE_DEBUG` | `0` | Set `1` to print raw SSE frames to stderr |
| `CLAUDE_POOL_DIR` | `./claude_pool` | Local pool directory (overridden by `main.py` at startup) |
| `CLAUDE_ALLOW_BROWSER_LOGIN` | `1` | Set to `0` on server to block accidental Playwright launch |
| `CLAUDE_API_KEY` | _(unset)_ | Bearer token for endpoint auth. All endpoints are open if unset |
| `CLAUDE_ACCOUNT_01`–`10` | _(unset)_ | Base64-encoded `session.json` blobs for Railway/Render deployment |

---

## Testing

```bash
# Health
curl https://your-app.railway.app/

# Account status
curl https://your-app.railway.app/status

# List models
curl https://your-app.railway.app/v1/models

# Chat (non-streaming)
curl -X POST https://your-app.railway.app/v1/chat/completions \
  -H "Authorization: Bearer $CLAUDE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6 High + Think",
    "messages": [{"role": "user", "content": "hello"}]
  }'

# Chat (streaming)
curl -X POST https://your-app.railway.app/v1/chat/completions \
  -H "Authorization: Bearer $CLAUDE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-6",
    "stream": true,
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

---

## Known Limitations

- **No auto-refresh.** Sessions must be manually re-captured and re-deployed when expired.
- **Cloudflare IP-binding.** `cf_clearance` cookies captured on one IP may not work on another. This is the primary failure mode on cloud deployments.
- **No message limit bypass.** When an account hits Claude's usage cap, `message_limit` events are raised and the account is treated as a failure. The pool rotates to the next account.
- **Thinking-only responses.** If a model produces thinking tokens but no text output, the conversation state is reset and the turn is discarded to avoid poisoning the `parent_message_uuid` chain.
- **File attachment type.** The file attachment body currently hardcodes `"file_type": "application/pdf"` regardless of actual MIME type. Works for PDFs and images; may need adjustment for other types.
- **`anonymous_id` is empty.** The `ajs_anonymous_id` value in `localStorage` is not reliably captured during login. The `anthropic-anonymous-id` header is omitted when empty.
## Phase 3.4 — Tool Registry, Validation, and Invocation Ledger

Phase 3.4 adds the model-side tool compatibility layer without executing tools.

- `agent_bridge/tool_registry.py` ingests the tool definitions Claude Code provides and validates model-emitted inputs against their JSON Schemas using `jsonschema`.
- Unknown tools, invalid schemas, and invalid tool inputs are rejected explicitly.
- `AgentSessionState` now accepts validated `ToolInvocation` records and tracks pending/completed/failed calls by `tool_use_id`.
- Tool execution, `tool_use` response emission, `tool_result` ingestion, and the full agent loop remain deferred to later phases.

Verification: `20 passed`.
