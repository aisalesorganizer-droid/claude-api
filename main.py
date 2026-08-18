"""
main.py — FastAPI wrapper for claude_client.py
===============================================
Dual-protocol API endpoint for Claude.ai web client.

Account injection (Render/Railway env vars):
  CLAUDE_ACCOUNT      — single account JSON, base64-encoded
  CLAUDE_ACCOUNT_01   — pool account 1, base64-encoded
  ... (up to 10)

Env:
  CLAUDE_API_KEY      — Bearer token required for all mutation endpoints
  CLAUDE_ALLOW_BROWSER_LOGIN — set "1" locally; server should omit this

Endpoints:
  GET  /                    health check
  GET  /status              account pool health + session expiry
  GET  /v1/models           model list
  POST /chat                simple {message} -> {response}
  POST /v1/chat/completions OpenAI-compatible (streaming + non-streaming)
  POST /v1/messages         Anthropic Messages API-compatible (for Claude Code)
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

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ─── Account loading from env ─────────────────────────────────────────────────

def _load_accounts_from_env() -> list[tuple[str, dict]]:
    """Load accounts from env vars. Keep ALL cookies including cf_clearance.
    Cloudflare cookies are IP-bound but without them datacenter IPs get 403.
    """
    accounts = []
    for i in range(1, 11):
        key = f"CLAUDE_ACCOUNT_{i:02d}"
        val = os.environ.get(key, "").strip()
        if val:
            try:
                data = json.loads(base64.b64decode(val))
                cookies = data.get("cookies", {})
                accounts.append((f"account_{i:02d}", data))
                print(f"[startup] ✓ Loaded {key} (cookies: {list(cookies.keys())})")
            except Exception as e:
                print(f"[startup] ✗ Failed to decode {key}: {e}")

    if not accounts:
        val = os.environ.get("CLAUDE_ACCOUNT", "").strip()
        if val:
            try:
                data = json.loads(base64.b64decode(val))
                cookies = data.get("cookies", {})
                accounts.append(("default", data))
                print(f"[startup] ✓ Loaded CLAUDE_ACCOUNT (cookies: {list(cookies.keys())})")
            except Exception as e:
                print(f"[startup] ✗ Failed to decode CLAUDE_ACCOUNT: {e}")

    return accounts


def _write_accounts_to_disk(accounts: list[tuple[str, dict]]) -> Path:
    tmpdir = Path(tempfile.mkdtemp(prefix="claude_accounts_"))
    for label, data in accounts:
        slot_dir = tmpdir / label
        slot_dir.mkdir(parents=True, exist_ok=True)
        path = slot_dir / "session.json"
        path.write_text(json.dumps(data, indent=2))
        print(f"[startup] ✓ Wrote {label}/session.json → {tmpdir}")
    return tmpdir


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
    "claude-haiku-4-5",
    "claude-haiku-4-5 (Extended)",
    "claude-sonnet-4-6 Low",
    "claude-sonnet-4-6 Medium",
    "claude-sonnet-4-6 High",
    "claude-sonnet-4-6 Max",
    "claude-sonnet-4-6 Low + Think",
    "claude-sonnet-4-6 Medium + Think",
    "claude-sonnet-4-6 High + Think",
    "claude-sonnet-4-6 Max + Think",
    "claude-sonnet-5 Low",
    "claude-sonnet-5 Medium",
    "claude-sonnet-5 High",
    "claude-sonnet-5 XHigh",
    "claude-sonnet-5 Max",
    "claude-sonnet-5 Low + Think",
    "claude-sonnet-5 Medium + Think",
    "claude-sonnet-5 High + Think",
    "claude-sonnet-5 XHigh + Think",
    "claude-sonnet-5 Max + Think",
    "claude-opus-4-6",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-fable-5",
]

# ─── Global state ─────────────────────────────────────────────────────────────

_provider_ready = False
_init_error: str | None = None
_accounts: list[str] = []
_index_lock = threading.Lock()
_current_index = 0

# FIX: Track which account uploaded which file so chat can route to the same org.
_file_account_map: dict[str, str] = {}
_file_map_lock = threading.Lock()

# FIX: API key gate — set CLAUDE_API_KEY on Railway/Render
API_KEY = os.environ.get("CLAUDE_API_KEY", "").strip()


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
                if time.time() - self._last_fail < 300:
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
    global _provider_ready, _init_error, _accounts, _account_objects
    try:
        print("[startup] Initializing Claude provider ...")
        _accounts = list_accounts()
        if not _accounts:
            raise RuntimeError("No accounts found in pool directory.")

        for label in _accounts:
            data = load_slot_session(label)
            if data is None:
                raise RuntimeError(f"No session.json found for {label}")

            # Pre-flight: test if session can hit Claude without Cloudflare 403
            s = _make_curl_session(data.get("cookies", {}), None)
            try:
                r = s.get(BASE_URL + "/api/organizations", timeout=15)
                if r.status_code == 403 and "Just a moment" in r.text:
                    print(f"[startup] ⚠ {label}: Cloudflare challenge (403) — session likely IP-bound.")
                    # Still load it; may work if IP is close, but mark as unhealthy
                    acc = _ClaudeAccount(label, data)
                    acc.mark_failure()
                    _account_objects.append(acc)
                    continue
                elif r.status_code != 200:
                    print(f"[startup] ⚠ {label}: org lookup returned HTTP {r.status_code}")
            except Exception as e:
                print(f"[startup] ⚠ {label}: pre-flight check failed: {e}")

            _account_objects.append(_ClaudeAccount(label, data))
            print(f"[startup]   Loaded: {label}")

        healthy = sum(1 for a in _account_objects if a.is_healthy)
        if healthy == 0:
            raise RuntimeError(
                "All accounts blocked by Cloudflare. "
                "Cookies were generated from a different IP than this server. "
                "Regenerate cookies from the same IP, or use a proxy (CLAUDE_PROXY)."
            )

        _provider_ready = True
        print(f"[startup] ✓ Claude ready — {len(_accounts)} account(s), {healthy} healthy")
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


# ─── Auth guard ───────────────────────────────────────────────────────────────

async def require_auth(request: Request):
    """Bearer-token gate. Skipped if CLAUDE_API_KEY is not set."""
    if not API_KEY:
        return
    auth = request.headers.get("authorization", "")
    if not auth.startswith("Bearer ") or auth[7:] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


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


# FIX: Create a fresh _SlotState per request to prevent concurrent-request corruption.
def _fresh_slot_state(label: str) -> _SlotState:
    data = load_slot_session(label)
    if data is None:
        raise RuntimeError(f"{label} session missing")
    return _SlotState(label, data, None)


def _stream_claude(prompt: str, model: str, file_attachments: Optional[List[dict]] = None,
                   forced_label: Optional[str] = None) -> Generator[str, None, None]:
    """Yield text chunks, rotating through healthy accounts.

    If forced_label is provided, that account is tried first (for file-attachment consistency).
    """
    n = len(_account_objects)
    if n == 0:
        raise RuntimeError("No accounts available")

    tried: set[str] = set()

    # 1. Try forced account first (e.g. the one that uploaded the file)
    if forced_label:
        acc = next((a for a in _account_objects if a.label == forced_label), None)
        if acc and acc.is_healthy and acc.label not in tried:
            tried.add(acc.label)
            try:
                state = _fresh_slot_state(acc.label)
                for chunk in _stream_on_slot(state, prompt, model, file_attachments):
                    yield chunk
                acc.mark_success()
                return
            except RuntimeError as e:
                err_msg = str(e)
                print(f"[stream] {acc.label} failed: {err_msg}", flush=True)
                acc.mark_failure()
                if "401" in err_msg or "expired" in err_msg.lower():
                    pass
                elif "429" in err_msg:
                    time.sleep(2)
                elif "overloaded" in err_msg.lower() or "temporarily" in err_msg.lower():
                    time.sleep(3)
                elif "403" in err_msg:
                    time.sleep(2)
                else:
                    raise

    # 2. Normal rotation for remaining healthy accounts
    for _ in range(n):
        acc = _next_healthy_account()
        if acc is None:
            break
        if acc.label in tried:
            continue
        tried.add(acc.label)
        try:
            state = _fresh_slot_state(acc.label)
            for chunk in _stream_on_slot(state, prompt, model, file_attachments):
                yield chunk
            acc.mark_success()
            return
        except RuntimeError as e:
            err_msg = str(e)
            print(f"[stream] {acc.label} failed: {err_msg}", flush=True)
            acc.mark_failure()
            if "401" in err_msg or "expired" in err_msg.lower():
                continue
            if "429" in err_msg:
                time.sleep(2)
                continue
            if "overloaded" in err_msg.lower() or "temporarily" in err_msg.lower():
                time.sleep(3)
                continue
            if "403" in err_msg or "Just a moment" in err_msg or "cloudflare" in err_msg.lower():
                print(f"[stream] {acc.label} blocked by Cloudflare challenge (403). "
                      f"cf_clearance may be stale or IP-mismatched.", flush=True)
                continue
            raise

    raise RuntimeError("All Claude accounts failed after full rotation.")


def _consume_stream(prompt: str, model: str, file_attachments: Optional[List[dict]] = None,
                    forced_label: Optional[str] = None, timeout: int = 120) -> str:
    """Non-streaming: collect all chunks into a single string."""
    chunks: list[str] = []
    deadline = time.time() + timeout
    for chunk in _stream_claude(prompt, model, file_attachments, forced_label):
        if time.time() > deadline:
            raise RuntimeError(f"Claude response timed out after {timeout}s")
        chunks.append(chunk)
    return "".join(chunks)


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
async def chat(req: ChatRequest, _=Depends(require_auth)):
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
async def upload_file_endpoint(file: UploadFile = File(...), _=Depends(require_auth)):
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
        state = _fresh_slot_state(label)
        s = state.http

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: upload_file(s, state.org_id, file_bytes, filename, mime_type),
        )

        file_uuid = result.get("uuid") or result.get("file_uuid") or result.get("id", "")

        # FIX: Remember which account owns this file so chat routes to the same org.
        with _file_map_lock:
            _file_account_map[file_uuid] = label

        print(f"[upload] ✓ {filename} → {file_uuid} (account={label})")
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
async def oai_chat(request: Request, _=Depends(require_auth)):
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

    # FIX: Only append trailing Assistant: marker if the last non-system message
    # is from the user. If it ends with assistant, we want continuation, not a new turn.
    last_non_system = next((m for m in reversed(messages) if m["role"] != "system"), None)
    if last_non_system is None or last_non_system["role"] == "user":
        parts.append("\n\nAssistant:")

    prompt = "".join(parts)

    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    )
    print(f"[oai] model={model} prompt={last_user[:80]}")

    # FIX: If file attachments exist, determine which account uploaded them
    # and force the chat to use that account (files are org-scoped).
    forced_label = None
    if file_attachments:
        for fa in file_attachments:
            fid = fa.get("file_uuid", "")
            with _file_map_lock:
                fl = _file_account_map.get(fid)
            if fl:
                forced_label = fl
                print(f"[oai] Forcing account {forced_label} for file {fid}")
                break

    if req.stream:
        def stream_generator():
            """Thread-safe generator that runs Claude streaming in a thread and yields SSE."""
            import queue
            q = queue.Queue()
            done = threading.Event()
            error_holder = [None]

            def _run_in_thread():
                try:
                    for chunk in _to_openai_stream(_stream_claude(prompt, model, file_attachments, forced_label), model):
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
            lambda: _consume_stream(prompt, model, file_attachments, forced_label),
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


# ═══════════════════════════════════════════════════════════════════════════════
# Anthropic Messages API  —  /v1/messages
# ═══════════════════════════════════════════════════════════════════════════════
#
# Claude Code points ANTHROPIC_BASE_URL at this server and calls POST /v1/messages
# using the native Anthropic wire format.  We translate that into the same
# _stream_claude() pipeline used by /v1/chat/completions, then emit proper
# Anthropic SSE events so Claude Code gets exactly what it expects.
#
# Anthropic streaming event sequence:
#   message_start
#   content_block_start   (index=0, type="text")
#   ping                  (keepalive)
#   content_block_delta   (repeating, type="text_delta")
#   content_block_stop
#   message_delta         (stop_reason="end_turn", usage)
#   message_stop
#
# Non-streaming response mirrors the /v1/messages shape:
#   {id, type, role, content:[{type,text}], model, stop_reason, usage}
# ───────────────────────────────────────────────────────────────────────────────

class _AntContentBlock(BaseModel):
    type: str
    text: str


class _AntMessage(BaseModel):
    role: str
    content: str | list  # str for simple text; list for content-block arrays


class AntRequest(BaseModel):
    model: str = "claude-sonnet-4-6 High"
    messages: List[_AntMessage]
    system: Optional[str | list] = None   # str or content-block list (Claude Code sends list with cache hints)
    max_tokens: Optional[int] = 8096
    stream: Optional[bool] = False
    temperature: Optional[float] = None   # accepted, ignored (web API doesn't expose it)
    top_p: Optional[float] = None         # accepted, ignored
    top_k: Optional[int] = None           # accepted, ignored
    model_config = {"extra": "ignore"}    # swallow any other Claude Code fields


def _ant_extract_text(content) -> str:
    """Pull plain text out of either a str or a content-block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _ant_build_prompt(req: AntRequest) -> str:
    """
    Assemble the Claude web-API prompt string from an Anthropic Messages request.

    System prompt priority:
      1. Top-level `system` field  (Claude Code always uses this)
      2. A message with role="system" in the messages list  (legacy fallback)

    Prompt format matches what _stream_on_slot() expects — same as oai_chat.
    """
    parts: list[str] = []

    # 1. System prompt
    system_text: Optional[str] = None
    if req.system:
        system_text = _ant_extract_text(req.system).strip()
    else:
        for m in req.messages:
            if m.role == "system":
                system_text = _ant_extract_text(m.content).strip()
                break

    if system_text:
        parts.append(f"<s>\n{system_text}\n</s>")

    # 2. Conversation turns (skip system role — already handled above)
    for m in req.messages:
        if m.role == "system":
            continue
        text = _ant_extract_text(m.content)
        if m.role == "user":
            parts.append(f"\n\nHuman: {text}")
        elif m.role == "assistant":
            parts.append(f"\n\nAssistant: {text}")

    # 3. Trailing assistant turn marker (only when last non-system turn is from user)
    last_non_sys = next(
        (m for m in reversed(req.messages) if m.role != "system"), None
    )
    if last_non_sys is None or last_non_sys.role == "user":
        parts.append("\n\nAssistant:")

    return "".join(parts)


def _to_ant_stream(
    claude_chunks: Generator[str, None, None],
    model: str,
    msg_id: str,
) -> Generator[str, None, None]:
    """
    Convert Claude text-chunk generator → Anthropic SSE event stream.

    Emits the exact sequence Claude Code expects:
      message_start → content_block_start → ping →
      content_block_delta* → content_block_stop →
      message_delta → message_stop
    """
    input_tokens  = 0   # we don't have real counts; send 0 — Claude Code ignores them
    output_tokens = 0

    def _evt(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    # ── message_start ──────────────────────────────────────────────────────────
    yield _evt("message_start", {
        "type": "message_start",
        "message": {
            "id":           msg_id,
            "type":         "message",
            "role":         "assistant",
            "content":      [],
            "model":        model,
            "stop_reason":  None,
            "stop_sequence": None,
            "usage": {
                "input_tokens":  input_tokens,
                "output_tokens": output_tokens,
            },
        },
    })

    # ── content_block_start ────────────────────────────────────────────────────
    yield _evt("content_block_start", {
        "type":  "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    })

    # ── ping (keepalive — Claude Code expects at least one) ────────────────────
    yield _evt("ping", {"type": "ping"})

    # ── content_block_delta* ───────────────────────────────────────────────────
    for chunk in claude_chunks:
        if chunk:
            output_tokens += len(chunk.split())
            yield _evt("content_block_delta", {
                "type":  "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk},
            })

    # ── content_block_stop ─────────────────────────────────────────────────────
    yield _evt("content_block_stop", {
        "type":  "content_block_stop",
        "index": 0,
    })

    # ── message_delta (stop reason + final usage) ──────────────────────────────
    yield _evt("message_delta", {
        "type":  "message_delta",
        "delta": {
            "stop_reason":   "end_turn",
            "stop_sequence": None,
        },
        "usage": {"output_tokens": output_tokens},
    })

    # ── message_stop ───────────────────────────────────────────────────────────
    yield _evt("message_stop", {"type": "message_stop"})


def _ant_error_event(message: str, err_type: str = "server_error") -> str:
    """Return a well-formed Anthropic error SSE event so Claude Code can surface it."""
    return (
        f"event: error\n"
        f"data: {json.dumps({'type': 'error', 'error': {'type': err_type, 'message': message}})}\n\n"
    )


@app.post("/v1/messages")
async def ant_messages(request: Request, _=Depends(require_auth)):
    """
    Anthropic Messages API endpoint.

    Accepts the native Anthropic POST /v1/messages body and returns the native
    Anthropic response format (streaming or non-streaming).

    Set in Claude Code's ~/.claude/settings.json:
        {
          "env": {
            "ANTHROPIC_BASE_URL": "https://<your-railway-app>.railway.app",
            "ANTHROPIC_AUTH_TOKEN": "<your-CLAUDE_API_KEY>"
          }
        }
    """
    raw_body = await request.json()
    print(f"[ant] RAW BODY: {json.dumps(raw_body)[:300]}")

    try:
        req = AntRequest(**raw_body)
    except Exception as e:
        print(f"[ant] PARSE ERROR: {e}")
        raise HTTPException(status_code=422, detail=str(e))

    _guard()

    if not req.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Models confirmed unsupported on these accounts (plan restriction).
    # Claude Code sends these by default; remap them to the working fallback.
    _ANT_MODEL_DENYLIST = {
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-fable-5",
    }
    _ANT_FALLBACK = "claude-sonnet-4-6 High"

    if req.model in _ANT_MODEL_DENYLIST:
        model = _ANT_FALLBACK
        print(f"[ant] model={req.model!r} not supported — remapped to {_ANT_FALLBACK!r}")
    elif req.model in MODEL_CONFIGS:
        model = req.model
    else:
        model = MODEL

    # Build the Claude web-API prompt
    prompt = _ant_build_prompt(req)

    last_user = next(
        (_ant_extract_text(m.content) for m in reversed(req.messages) if m.role == "user"),
        "",
    )
    print(f"[ant] model={model} stream={req.stream} prompt_tail={last_user[:80]}")

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    # ── Streaming ──────────────────────────────────────────────────────────────
    if req.stream:
        def stream_generator():
            import queue as _queue
            q     = _queue.Queue()
            done  = threading.Event()
            error_holder: list[Optional[Exception]] = [None]

            def _run_in_thread():
                try:
                    for evt in _to_ant_stream(
                        _stream_claude(prompt, model, None, None),
                        model,
                        msg_id,
                    ):
                        q.put(evt)
                except Exception as exc:
                    error_holder[0] = exc
                finally:
                    done.set()

            t = threading.Thread(target=_run_in_thread, daemon=True)
            t.start()

            while not done.is_set() or not q.empty():
                try:
                    yield q.get(timeout=0.1)
                except _queue.Empty:
                    if done.is_set():
                        break

            t.join(timeout=5)

            if error_holder[0]:
                err = str(error_holder[0])
                print(f"[ant stream] ERROR: {err}")
                if "expired" in err.lower() or "401" in err:
                    yield _ant_error_event(f"Session expired: {err}", "authentication_error")
                elif "429" in err:
                    yield _ant_error_event("Rate limited. Try again later.", "rate_limit_error")
                elif "overloaded" in err.lower():
                    yield _ant_error_event("Claude is overloaded. Try again shortly.", "overloaded_error")
                else:
                    yield _ant_error_event(err, "server_error")

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":   "no-cache",
                "X-Accel-Buffering": "no",       # disable nginx buffering on Railway
            },
        )

    # ── Non-streaming ──────────────────────────────────────────────────────────
    try:
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(
            None,
            lambda: _consume_stream(prompt, model, None, None),
        )

        input_tokens  = sum(
            len(_ant_extract_text(m.content).split()) for m in req.messages
        )
        output_tokens = len(answer.split())

        print(f"[ant] answer={answer[:200]}")

        return {
            "id":            msg_id,
            "type":          "message",
            "role":          "assistant",
            "content":       [{"type": "text", "text": answer}],
            "model":         model,
            "stop_reason":   "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens":  input_tokens,
                "output_tokens": output_tokens,
            },
        }

    except RuntimeError as e:
        err = str(e)
        print(f"[ant] RuntimeError: {err}")
        if "expired" in err.lower() or "401" in err:
            raise HTTPException(status_code=401, detail=f"Session expired: {err}")
        if "429" in err:
            raise HTTPException(status_code=429, detail="Rate limited. Try again later.")
        if "overloaded" in err.lower():
            raise HTTPException(status_code=529, detail="Claude overloaded. Try again.")
        raise HTTPException(status_code=503, detail=err)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ant] ERROR: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))