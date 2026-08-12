"""
main.py — FastAPI wrapper for claude_client.py
===============================================
OpenAI-compatible API endpoint for Claude.ai web client.

Account injection (Render/Railway env vars):
  CLAUDE_ACCOUNT      — single account JSON, base64-encoded
  CLAUDE_ACCOUNT_01   — pool account 1, base64-encoded
  CLAUDE_ACCOUNT_02   — pool account 2, base64-encoded
  ... (up to 10)

To encode your account JSON for Render:
  python -c "import base64,json; print(base64.b64encode(open('claude_pool/account_0/session.json','rb').read()).decode())"

Endpoints:
  GET  /                    health check
  GET  /status              account pool health + session expiry
  GET  /v1/models           model list
  POST /chat                simple {message} -> {response}
  POST /v1/chat/completions OpenAI-compatible (streaming + non-streaming)
  POST /v1/files/upload     multipart/form-data file upload -> {file_id, filename, status}
"""

import asyncio
import base64
import datetime
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Generator

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ─── Account loading from env ─────────────────────────────────────────────────

# Cloudflare cookies that are IP-bound — strip them on server to test
# if curl_cffi impersonation alone is enough
_CF_COOKIES = {"cf_clearance", "__cf_bm"}

def _load_accounts_from_env() -> list[tuple[str, dict]]:
    accounts = []
    for i in range(1, 11):
        key = f"CLAUDE_ACCOUNT_{i:02d}"
        val = os.environ.get(key, "").strip()
        if val:
            try:
                data = json.loads(base64.b64decode(val))
                # STRIP IP-bound Cloudflare cookies for server deployment
                cookies = data.get("cookies", {})
                stripped = {k: v for k, v in cookies.items() if k not in _CF_COOKIES}
                removed = set(cookies.keys()) - set(stripped.keys())
                if removed:
                    print(f"[startup] {key}: stripped IP-bound cookies: {removed}")
                data["cookies"] = stripped
                accounts.append((f"account_{i:02d}", data))
                print(f"[startup] ✓ Loaded {key} (cookies: {list(stripped.keys())})")
            except Exception as e:
                print(f"[startup] ✗ Failed to decode {key}: {e}")

    if not accounts:
        val = os.environ.get("CLAUDE_ACCOUNT", "").strip()
        if val:
            try:
                data = json.loads(base64.b64decode(val))
                cookies = data.get("cookies", {})
                stripped = {k: v for k, v in cookies.items() if k not in _CF_COOKIES}
                removed = set(cookies.keys()) - set(stripped.keys())
                if removed:
                    print(f"[startup] CLAUDE_ACCOUNT: stripped IP-bound cookies: {removed}")
                data["cookies"] = stripped
                accounts.append(("default", data))
                print(f"[startup] ✓ Loaded CLAUDE_ACCOUNT (cookies: {list(stripped.keys())})")
            except Exception as e:
                print(f"[startup] ✗ Failed to decode CLAUDE_ACCOUNT: {e}")

    return accounts


def _write_accounts_to_disk(accounts: list[tuple[str, dict]]) -> Path:
    """Write accounts to temp dir in the structure claude_client expects:
       /tmp/claude_accounts_xxx/account_01/session.json
    """
    tmpdir = Path(tempfile.mkdtemp(prefix="claude_accounts_"))
    for label, data in accounts:
        slot_dir = tmpdir / label
        slot_dir.mkdir(parents=True, exist_ok=True)
        path = slot_dir / "session.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"[startup] ✓ Wrote {label}/session.json → {tmpdir}")
    return tmpdir


# ─── Patch claude_client paths BEFORE importing ──────────────────────────────

_accounts_from_env = _load_accounts_from_env()

if not _accounts_from_env:
    print("[startup] ✗ No account env vars found.")
    print("[startup]   Set CLAUDE_ACCOUNT or CLAUDE_ACCOUNT_01 in Render.")
    sys.exit(1)

_tmpdir = _write_accounts_to_disk(_accounts_from_env)

import claude_client as _cc_module
_cc_module.POOL_DIR = _tmpdir

from claude_client import (
    _make_curl_session, create_conversation, _stream_on_slot, _SlotState,
    upload_file, list_accounts, load_slot_session, ensure_slot,
    BASE_URL, MODEL, CURL_AVAILABLE, MODEL_CONFIGS, resolve_model
)

# ─── Supported models ─────────────────────────────────────────────────────────

SUPPORTED_MODELS = [
    # ── Haiku 4.5 — mode-only, no effort field ───────────────────────────────
    "claude-haiku-4-5",
    "claude-haiku-4-5 (Extended)",

    # ── Sonnet 4.6 — no thinking: low → medium → high → max ─────────────────
    "claude-sonnet-4-6 Low",
    "claude-sonnet-4-6 Medium",
    "claude-sonnet-4-6 High",
    "claude-sonnet-4-6 Max",
    # ── Sonnet 4.6 — thinking on: low → medium → high → max ──────────────────
    "claude-sonnet-4-6 Low + Think",
    "claude-sonnet-4-6 Medium + Think",
    "claude-sonnet-4-6 High + Think",
    "claude-sonnet-4-6 Max + Think",

    # ── Sonnet 5 — no thinking: low → medium → high → xhigh → max ───────────
    "claude-sonnet-5 Low",
    "claude-sonnet-5 Medium",
    "claude-sonnet-5 High",
    "claude-sonnet-5 XHigh",
    "claude-sonnet-5 Max",
    # ── Sonnet 5 — thinking on: low → medium → high → xhigh → max ───────────
    "claude-sonnet-5 Low + Think",
    "claude-sonnet-5 Medium + Think",
    "claude-sonnet-5 High + Think",
    "claude-sonnet-5 XHigh + Think",
    "claude-sonnet-5 Max + Think",

    # ── Opus 4.6 / 4.7 / 4.8 ────────────────────────────────────────────────
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",

    # ── Opus 5 ───────────────────────────────────────────────────────────────
    "claude-opus-5",

    # ── Fable 5 (may 403 on free accounts) ───────────────────────────────────
    "claude-fable-5",
]

# ─── Global state ─────────────────────────────────────────────────────────────

_provider_ready = False
_init_error: str | None = None
_accounts: list[str] = []
_slot_states: dict[str, _SlotState] = {}
_index_lock = threading.Lock()
_current_index = 0


# ─── Account wrapper ──────────────────────────────────────────────────────────

class _ClaudeAccount:
    """Wraps a raw account dict with health tracking."""
    def __init__(self, label: str, data: dict):
        self.label = label
        self.data = data
        self._fail_count = 0
        self._last_fail = 0.0
        self._lock = threading.Lock()

    @property
    def is_healthy(self) -> bool:
        with self._lock:
            if self._fail_count >= 3:
                if time.time() - self._last_fail < 300:  # 5 min bench
                    return False
                self._fail_count = 0
            return True

    def mark_failure(self):
        with self._lock:
            self._fail_count += 1
            self._last_fail = time.time()

    def mark_success(self):
        with self._lock:
            self._fail_count = 0

    def __repr__(self):
        return f"_ClaudeAccount({self.label})"


_account_objects: list[_ClaudeAccount] = []


def _init_provider():
    global _provider_ready, _init_error, _accounts, _account_objects, _slot_states
    try:
        print("[startup] Initializing Claude provider ...")
        _accounts = list_accounts()
        if not _accounts:
            raise RuntimeError("No accounts found in pool directory.")

        for label in _accounts:
            data = load_slot_session(label)
            if data is None:
                raise RuntimeError(f"No session.json found for {label}")
            _account_objects.append(_ClaudeAccount(label, data))
            _slot_states[label] = _SlotState(label, data, None)
            print(f"[startup]   Loaded: {label}")

        _provider_ready = True
        print(f"[startup] ✓ Claude ready — {len(_accounts)} account(s)")
    except Exception as e:
        _init_error = str(e)
        print(f"[startup] ✗ Claude init failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _init_provider)
    yield
    print("[shutdown] ✓ Claude API stopped")


app = FastAPI(title="claude-api", lifespan=lifespan)


# ─── Guard ────────────────────────────────────────────────────────────────────

def _guard():
    if _init_error:
        raise HTTPException(status_code=503, detail=f"Init failed: {_init_error}")
    if not _provider_ready:
        raise HTTPException(status_code=503, detail="Still initializing — try again in a moment")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _next_healthy_account() -> Optional[_ClaudeAccount]:
    global _current_index
    with _index_lock:
        n = len(_account_objects)
        for _ in range(n):
            acc = _account_objects[_current_index % n]
            _current_index += 1
            if acc.is_healthy:
                return acc
    return None


def _consume_stream(prompt: str, model: str, file_attachments: Optional[List[dict]] = None,
                    timeout: int = 120) -> str:
    """Non-streaming: collect all chunks into a single string."""
    chunks: list[str] = []
    deadline = time.time() + timeout

    for chunk in _stream_claude(prompt, model, file_attachments):
        if time.time() > deadline:
            raise RuntimeError(f"Claude response timed out after {timeout}s")
        chunks.append(chunk)

    return "".join(chunks)


def _stream_claude(prompt: str, model: str, file_attachments: Optional[List[dict]] = None) -> Generator[str, None, None]:
    """Yield text chunks, rotating through healthy accounts."""
    n = len(_account_objects)
    if n == 0:
        raise RuntimeError("No accounts available")

    for _ in range(n):
        acc = _next_healthy_account()
        if acc is None:
            raise RuntimeError("All Claude accounts are exhausted. Wait 5 minutes or re-run --add-account locally.")

        label = acc.label
        try:
            if label not in _slot_states:
                data = load_slot_session(label)
                if data is None:
                    raise RuntimeError(f"{label} session missing")
                _slot_states[label] = _SlotState(label, data, None)

            state = _slot_states[label]
            state.conv_id = None
            state.last_asst_uuid = None
            for chunk in _stream_on_slot(state, prompt, model, file_attachments):
                yield chunk

            acc.mark_success()
            return

        except RuntimeError as e:
            err_msg = str(e)
            print(f"[stream] {label} failed: {err_msg}", flush=True)
            acc.mark_failure()

            if "401" in err_msg or "expired" in err_msg.lower():
                _slot_states.pop(label, None)
                continue
            if "429" in err_msg:
                time.sleep(2)
                continue
            if "overloaded" in err_msg.lower() or "temporarily" in err_msg.lower():
                time.sleep(3)
                continue
            if "403" in err_msg:
                time.sleep(2)
                continue
            raise

    raise RuntimeError("All Claude accounts failed after full rotation.")


def _to_openai_stream(claude_chunks: Generator[str, None, None], model: str) -> Generator[str, None, None]:
    """Convert Claude text chunks to OpenAI SSE format."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    for text_chunk in claude_chunks:
        if text_chunk:
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": text_chunk}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(payload)}\n\n"

    done_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(done_payload)}\n\n"
    yield "data: [DONE]\n\n"


# ─── Pydantic models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "claude-sonnet-4-6"


class ChatResponse(BaseModel):
    response: str
    model: str | None = None


class StatusResponse(BaseModel):
    ready: bool
    error: str | None = None
    accounts: list | None = None


class OAIMessage(BaseModel):
    role: str
    content: str


class OAIRequest(BaseModel):
    model: Optional[str] = "claude-sonnet-4-6"
    messages: List[OAIMessage]
    stream: Optional[bool] = False
    model_config = {"extra": "ignore"}


class OAIChoice(BaseModel):
    index: int
    message: OAIMessage
    finish_reason: str


class OAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OAIResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OAIChoice]
    usage: OAIUsage


class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    status: str
    mime_type: str | None = None


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "claude-api",
        "ready": _provider_ready,
        "init_error": _init_error,
    }


@app.get("/status", response_model=StatusResponse)
def status():
    if _init_error:
        return StatusResponse(ready=False, error=_init_error)
    if not _provider_ready:
        return StatusResponse(ready=False, error="Still initializing ...")

    results = []
    now = datetime.datetime.now(datetime.timezone.utc)
    for acc in _account_objects:
        try:
            data = acc.data
            exp = datetime.datetime.fromisoformat(data.get("expires", ""))
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=datetime.timezone.utc)
            days_left = (exp - now).days
            ck = data.get("cookies", {})
            results.append({
                "label": acc.label,
                "status": "ok" if days_left >= 7 else "expiring_soon",
                "days_remaining": days_left,
                "healthy": acc.is_healthy,
                "fail_count": acc._fail_count,
                "has_sessionKey": "sessionKey" in ck,
                "has_cf_clearance": "cf_clearance" in ck,
            })
        except Exception as e:
            results.append({"label": acc.label, "status": "error", "error": str(e)})

    healthy_count = sum(1 for r in results if r.get("healthy", False))
    return StatusResponse(
        ready=True,
        accounts=results,
    )


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": 1700000000,
                "owned_by": "anthropic",
            }
            for m in SUPPORTED_MODELS
        ],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    _guard()
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    model = req.model if req.model in MODEL_CONFIGS else MODEL

    try:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            lambda: _consume_stream(req.message, model),
        )
        return ChatResponse(response=answer, model=model)
    except RuntimeError as e:
        err = str(e)
        if "expired" in err.lower() or "401" in err:
            raise HTTPException(status_code=503, detail=f"Claude session expired: {err}. Re-run --add-account locally.")
        if "429" in err:
            raise HTTPException(status_code=429, detail="Claude rate limited. Try again later.")
        print(f"[chat] ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        print(f"[chat] ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/files/upload", response_model=FileUploadResponse)
async def upload_file_endpoint(file: UploadFile = File(...)):
    """
    Upload a file to Claude. Returns file metadata for use in chat attachments.
    Pass the file UUID via X-Claude-File-Ids header on /v1/chat/completions.
    """
    _guard()

    file_bytes = await file.read()
    filename = file.filename or "upload.pdf"
    mime_type = file.content_type or "application/octet-stream"

    print(f"[upload] Received {filename} — {len(file_bytes)} bytes — {mime_type}")

    acc = _next_healthy_account()
    if acc is None:
        raise HTTPException(status_code=503, detail="No healthy accounts available")

    label = acc.label
    try:
        if label not in _slot_states:
            data = load_slot_session(label)
            if data is None:
                raise RuntimeError(f"{label} session missing")
            _slot_states[label] = _SlotState(label, data, None)

        state = _slot_states[label]
        s = state.http

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: upload_file(s, state.org_id, file_bytes, filename, mime_type),
        )

        file_uuid = result.get("uuid") or result.get("file_uuid") or result.get("id", "")
        print(f"[upload] ✓ {filename} → {file_uuid}")
        return FileUploadResponse(
            file_id=file_uuid,
            filename=filename,
            status="ready",
            mime_type=mime_type,
        )

    except RuntimeError as e:
        print(f"[upload] ERROR: {e}")
        raise HTTPException(status_code=503, detail=f"Upload failed: {e}")
    except Exception as e:
        print(f"[upload] ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions")
async def oai_chat(request: Request):
    body = await request.json()
    print(f"[oai] RAW BODY: {json.dumps(body)[:200]}")

    try:
        req = OAIRequest(**body)
    except Exception as e:
        print(f"[oai] PARSE ERROR: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    _guard()

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Extract file attachments from header
    file_ids_header = request.headers.get("x-claude-file-ids", "").strip()
    file_attachments = None
    if file_ids_header:
        file_uuids = [fid.strip() for fid in file_ids_header.split(",") if fid.strip()]
        file_attachments = [{"file_uuid": fid, "file_type": "application/pdf"} for fid in file_uuids]
        print(f"[oai] file_attachments: {file_attachments}")

    # Accept both display names (SUPPORTED_MODELS) and canonical keys (MODEL_CONFIGS)
    model = req.model if req.model in MODEL_CONFIGS else MODEL

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # ── Prompt assembly: Claude web API format ────────────────────────────────
    # The completion endpoint has a single `prompt` field — no native system param.
    # Correct format: <system> block first, then Human:/Assistant: turn pairs.
    parts = []

    # 1. Extract system message (if any) → wrap in <system> XML tag
    system_content = next(
        (m["content"] for m in messages if m["role"] == "system"), None
    )
    if system_content:
        parts.append(f"<system>\n{system_content}\n</system>")

    # 2. Build Human/Assistant turns from non-system messages
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user":
            parts.append(f"\n\nHuman: {m['content']}")
        elif m["role"] == "assistant":
            parts.append(f"\n\nAssistant: {m['content']}")

    # 3. Trailing Assistant: marker so Claude knows it's its turn to respond
    parts.append("\n\nAssistant:")

    prompt = "".join(parts)

    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    print(f"[oai] model={model} prompt={last_user[:80]}")

    if req.stream:
        def stream_generator():
            """Thread-safe generator that runs Claude streaming in a thread and yields SSE."""
            import queue
            q = queue.Queue()
            done = threading.Event()
            error_holder = [None]

            def _run_in_thread():
                try:
                    for chunk in _to_openai_stream(_stream_claude(prompt, model, file_attachments), model):
                        q.put(chunk)
                except Exception as e:
                    error_holder[0] = e
                finally:
                    done.set()

            t = threading.Thread(target=_run_in_thread)
            t.start()

            while not done.is_set() or not q.empty():
                try:
                    chunk = q.get(timeout=0.1)
                    yield chunk
                except queue.Empty:
                    if done.is_set():
                        break

            t.join(timeout=5)

            if error_holder[0]:
                err = str(error_holder[0])
                print(f"[oai stream] ERROR: {err}")
                err_payload = {
                    "error": {
                        "message": err,
                        "type": "server_error",
                        "code": "session_expired" if "expired" in err.lower() else "unknown",
                    }
                }
                yield f"data: {json.dumps(err_payload)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Non-streaming
    try:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            lambda: _consume_stream(prompt, model, file_attachments),
        )

        prompt_tokens = sum(len(m["content"].split()) for m in messages)
        completion_tokens = len(answer.split())

        print(f"[oai] final_answer={answer[:200]}")

        return OAIResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                OAIChoice(
                    index=0,
                    message=OAIMessage(role="assistant", content=answer),
                    finish_reason="stop",
                )
            ],
            usage=OAIUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except RuntimeError as e:
        err = str(e)
        print(f"[oai] AIProviderError: {err}")
        if "expired" in err.lower() or "401" in err:
            raise HTTPException(status_code=503, detail=f"Claude session expired: {err}. Re-run --add-account locally.")
        if "429" in err:
            raise HTTPException(status_code=429, detail="Claude rate limited. Try again later.")
        raise HTTPException(status_code=503, detail=err)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[oai] ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))