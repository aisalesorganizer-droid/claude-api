#!/usr/bin/env python3
"""
claude.ai — Multi-Account Rotating Client (Server + CLI)
==========================================================
Production-ready: import-safe, env-var loading, file upload, health checks.

Setup (once):
  pip install curl_cffi playwright
  python -m playwright install chromium

Add accounts (local):
  python claude_client.py --add-account   (repeat per Google account)

CLI:
  python claude_client.py "hello"          one-shot
  python claude_client.py                interactive
  python claude_client.py --pool         show pool status

Env:
  CLAUDE_MODEL       model slug (default: claude-sonnet-4-6)
  CLAUDE_PROXY       socks5://... or "none"  (auto-detects WARP on 40000)
  CLAUDE_SSE_DEBUG   1 = raw SSE frames to stderr
  CLAUDE_POOL_DIR    pool directory (default: ./claude_pool)
"""

from __future__ import annotations

import os, sys, json, time, socket, datetime, argparse, uuid as _uuid, base64
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Generator

# ─── Lazy imports (safe for server import) ──────────────────────────────────

try:
    from curl_cffi import requests as curl_requests
    CURL_AVAILABLE = True
except ImportError:
    CURL_AVAILABLE = False
    curl_requests = None  # type: ignore

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


# ─── Config ────────────────────────────────────────────────────────────────────

MODEL     = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
PROXY_ENV = os.environ.get("CLAUDE_PROXY", "")
SSE_DEBUG = os.environ.get("CLAUDE_SSE_DEBUG", "0") == "1"
POOL_DIR  = Path(os.environ.get("CLAUDE_POOL_DIR", "claude_pool"))

# ─── Model configuration (source: HAR analysis 2026-08-12) ────────────────────
# thinking_mode : "off" | "auto" | "extended"
# effort        : "low" | "medium" | "high" | "xhigh" | "max" | None
#                 None  → field must be OMITTED (Haiku uses mode-only, no effort)
MODEL_CONFIGS: dict = {
    # Haiku 4.5 — mode-only thinking schema, effort field must not be sent
    "claude-haiku-4-5-20251001": {"thinking_mode": "off",      "effort": None},

    # Sonnet 4.6 — effort_and_mode, 4 levels (low/medium/high/max), no xhigh
    "claude-sonnet-4-6":         {"thinking_mode": "off",      "effort": "high"},

    # Sonnet 5 — effort_and_mode, 5 levels (adds xhigh between high and max)
    "claude-sonnet-5":           {"thinking_mode": "auto",     "effort": "medium"},

    # Opus 4.6 — extended thinking mode by default
    "claude-opus-4-6":           {"thinking_mode": "extended", "effort": "medium"},

    # Opus 4.7 — xhigh effort default
    "claude-opus-4-7":           {"thinking_mode": "auto",     "effort": "xhigh"},

    # Opus 4.8
    "claude-opus-4-8":           {"thinking_mode": "auto",     "effort": "high"},

    # Frontier models (in selector state; may 403 on free accounts)
    "claude-fable-5":            {"thinking_mode": "auto",     "effort": "high"},
    "claude-opus-5":             {"thinking_mode": "auto",     "effort": "high"},
}
_DEFAULT_MODEL_CONFIG: dict = {"thinking_mode": "off", "effort": "high"}

BASE_URL         = "https://claude.ai"
REFRESH_DAYS     = 0
ROTATION_FILE    = POOL_DIR / "rotation.json"

_CLIENT_SHA      = "2aa88381b74d2cd481c2dc6b403ca7cfbb8f5c2d"
_CLIENT_VERSION  = "1.0.0"
_CLIENT_PLATFORM = "web_claude_ai"

_WANTED_COOKIES = {
    "sessionKey", "routingHint", "lastActiveOrg",
    "anthropic-device-id", "__ssid", "activitySessionId",
    "cf_clearance", "__cf_bm", "CH-prefers-color-scheme",
    "intercom-session-lupk8zyo", "user-sidebar-visible-on-load",
}


# ─── UUID v7 ─────────────────────────────────────────────────────────────────

def uuid7() -> str:
    ts = int(time.time() * 1000)
    b  = ts.to_bytes(6, "big") + os.urandom(10)
    ba = bytearray(b)
    ba[6] = (ba[6] & 0x0F) | 0x70
    ba[8] = (ba[8] & 0x3F) | 0x80
    h = ba.hex()
    return "{}-{}-{}-{}-{}".format(h[:8], h[8:12], h[12:16], h[16:20], h[20:])


# ─── Proxy ─────────────────────────────────────────────────────────────────────

def detect_proxy() -> Optional[str]:
    if PROXY_ENV.lower() == "none":
        return None
    if PROXY_ENV:
        return PROXY_ENV
    for port in [40000, 1080, 8080]:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                return "socks5://127.0.0.1:{}".format(port)
        except OSError:
            pass
    return None


# ─── Pool management ───────────────────────────────────────────────────────────

def list_accounts(pool_dir: Optional[Path] = None) -> list:
    pd = pool_dir or POOL_DIR
    if not pd.exists():
        return []
    return sorted(d.name for d in pd.iterdir()
                  if d.is_dir() and d.name.startswith("account_"))

def next_slot_name(pool_dir: Optional[Path] = None) -> str:
    return "account_{}".format(len(list_accounts(pool_dir)))

def session_file(slot: str, pool_dir: Optional[Path] = None) -> Path:
    return (pool_dir or POOL_DIR) / slot / "session.json"

def profile_dir(slot: str, pool_dir: Optional[Path] = None) -> Path:
    return (pool_dir or POOL_DIR) / slot / "chrome_profile"


# ─── Rotation ──────────────────────────────────────────────────────────────────

def get_rotation_index(pool_dir: Optional[Path] = None) -> int:
    rf = (pool_dir or POOL_DIR) / "rotation.json"
    if not rf.exists():
        return 0
    try:
        return json.loads(rf.read_text()).get("index", 0)
    except Exception:
        return 0

def advance_rotation(current_idx: int, pool_dir: Optional[Path] = None):
    pd = pool_dir or POOL_DIR
    pd.mkdir(exist_ok=True)
    ((pd / "rotation.json")).write_text(json.dumps({"index": current_idx + 1}))


# ─── Session cache ─────────────────────────────────────────────────────────────

def load_slot_session(slot: str, pool_dir: Optional[Path] = None) -> Optional[dict]:
    sf = session_file(slot, pool_dir)
    if not sf.exists():
        return None
    try:
        data = json.loads(sf.read_text())
        exp  = datetime.datetime.fromisoformat(data["expires"])
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=datetime.timezone.utc)
        days_left = (exp - datetime.datetime.now(datetime.timezone.utc)).days
        if days_left < REFRESH_DAYS:
            print("[{}] Session expires in {}d — needs refresh".format(slot, days_left), file=sys.stderr)
            return None
        print("[{}] Session valid ({}d remaining)".format(slot, days_left), file=sys.stderr)
        return data
    except Exception as e:
        print("[{}] Cache bad ({}) — needs re-login".format(slot, e), file=sys.stderr)
        return None

def save_slot_session(slot: str, cookies: dict, org_id: str, device_id: str,
                      anonymous_id: str, expires_iso: str, pool_dir: Optional[Path] = None):
    sf = session_file(slot, pool_dir)
    sf.parent.mkdir(parents=True, exist_ok=True)
    sf.write_text(json.dumps({
        "cookies":      cookies,
        "org_id":       org_id,
        "device_id":    device_id,
        "anonymous_id": anonymous_id,
        "expires":      expires_iso,
        "slot":         slot,
    }, indent=2))
    try:
        sf.chmod(0o600)
    except Exception:
        pass
    print("[{}] Session saved → {}".format(slot, sf), file=sys.stderr)

def clear_slot_session(slot: str, pool_dir: Optional[Path] = None):
    session_file(slot, pool_dir).unlink(missing_ok=True)
    print("[{}] Session cleared".format(slot), file=sys.stderr)


# ─── Playwright login ──────────────────────────────────────────────────────────

def playwright_login(slot: str, proxy: Optional[str], pool_dir: Optional[Path] = None) -> dict:
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright not installed: pip install playwright && python -m playwright install chromium")

    pd = profile_dir(slot, pool_dir)
    pd.mkdir(parents=True, exist_ok=True)
    proxy_cfg = {"server": proxy} if proxy else None

    print("[{}] Opening browser — sign in with Google. Window closes automatically.".format(slot), file=sys.stderr)

    result = {}

    with sync_playwright() as pw:
        ctx = pw.chromium.launch_persistent_context(
            user_data_dir=str(pd),
            headless=False,
            proxy=proxy_cfg,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            ignore_default_args=["--enable-automation"],
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151.0.0.0 Safari/537.36"
            ),
        )

        page = ctx.new_page()
        page.goto(BASE_URL + "/login", wait_until="domcontentloaded", timeout=30_000)
        time.sleep(2)

        url_now = page.url
        print("[{}] URL: {}".format(slot, url_now), file=sys.stderr)

        already_in = (
            "claude.ai" in url_now
            and "/login" not in url_now
            and "accounts.google.com" not in url_now
        )

        if not already_in:
            print("[{}] Waiting for login (up to 5 min)...".format(slot), file=sys.stderr)
            try:
                page.wait_for_url(
                    lambda u: (
                        "claude.ai" in u
                        and "/login" not in u
                        and "accounts.google.com" not in u
                        and "account.google.com" not in u
                    ),
                    timeout=300_000,
                )
            except Exception:
                if "/login" in page.url or "google.com" in page.url:
                    raise RuntimeError("[{}] Login timed out".format(slot))

        print("[{}] Landed: {}".format(slot, page.url), file=sys.stderr)
        time.sleep(3)

        all_cookies = ctx.cookies()
        cookies = {}
        expires_iso = None
        for c in all_cookies:
            name = c.get("name", "")
            if name in _WANTED_COOKIES:
                cookies[name] = c.get("value", "")
                if name == "sessionKey":
                    raw_exp = c.get("expires", -1)
                    if raw_exp and raw_exp > 0:
                        exp_dt = datetime.datetime.fromtimestamp(raw_exp, tz=datetime.timezone.utc)
                        expires_iso = exp_dt.isoformat()

        print("[{}] Captured {} cookies: {}".format(slot, len(cookies), list(cookies.keys())), file=sys.stderr)

        if not cookies.get("sessionKey"):
            try:
                sk_js = page.evaluate(
                    "() => document.cookie.split(';').map(c=>c.trim())"
                    ".find(c=>c.startsWith('sessionKey='))?.split('=').slice(1).join('=')"
                )
                if sk_js:
                    cookies["sessionKey"] = sk_js
                    print("[{}] sessionKey via JS eval".format(slot), file=sys.stderr)
            except Exception as e:
                print("[{}] JS fallback: {}".format(slot, e), file=sys.stderr)

        if not cookies.get("sessionKey"):
            raise RuntimeError("[{}] sessionKey not found — did login complete?".format(slot))

        org_id = cookies.get("lastActiveOrg", "").strip()
        if not org_id or len(org_id) != 36:
            import re
            m = re.search(r"/organizations/([0-9a-f-]{36})", page.url)
            if m:
                org_id = m.group(1)
        if not org_id or len(org_id) != 36:
            try:
                org_id = page.evaluate(
                    "() => { try { return localStorage.getItem('lastActiveOrg') || ''; } catch(e) { return ''; } }"
                ) or ""
            except Exception:
                pass
        if not org_id or len(org_id) != 36:
            raise RuntimeError("[{}] Cannot determine org UUID".format(slot))

        device_id = cookies.get("anthropic-device-id", "")
        anonymous_id = ""
        try:
            anonymous_id = page.evaluate(
                "() => { try { return (localStorage.getItem('ajs_anonymous_id') || '').replace(/^\"|\"$/g, ''); } catch(e) { return ''; } }"
            ) or ""
        except Exception:
            pass

        ctx.close()

    if not expires_iso:
        exp_dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=28)
        expires_iso = exp_dt.isoformat()

    s = _make_curl_session(cookies, proxy)
    r = s.get(BASE_URL + "/api/organizations", timeout=15)
    print("[{}] Orgs: HTTP {}".format(slot, r.status_code), file=sys.stderr)
    if r.status_code != 200:
        raise RuntimeError("[{}] org lookup failed: {}".format(slot, r.status_code))
    orgs = r.json()
    if not orgs:
        raise RuntimeError("[{}] No organisations returned".format(slot))
    org_id = orgs[0]["uuid"]
    print("[{}] Org: {}".format(slot, org_id), file=sys.stderr)
    cookies["lastActiveOrg"] = org_id

    return {
        "cookies":      cookies,
        "org_id":       org_id,
        "device_id":    device_id,
        "anonymous_id": anonymous_id,
        "expires":      expires_iso,
    }


# ─── curl_cffi session builder ─────────────────────────────────────────────────

def _make_curl_session(cookies: dict, proxy: Optional[str]) -> curl_requests.Session:
    if not CURL_AVAILABLE:
        raise RuntimeError("curl_cffi not installed. Run: pip install curl_cffi")
    s = curl_requests.Session(
        impersonate="chrome131",
        proxies={"https": proxy, "http": proxy} if proxy else {},
    )
    for name, value in cookies.items():
        s.cookies.set(name, value, domain=".claude.ai")
        s.cookies.set(name, value, domain="claude.ai")

    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/151.0.0.0 Safari/537.36"
        ),
        "Accept-Language":           "en-US,en;q=0.9",
        "sec-ch-ua":                 '"Not=A?Brand";v="99", "Google Chrome";v="151", "Chromium";v="151"',
        "sec-ch-ua-mobile":          "?0",
        "sec-ch-ua-platform":        '"Windows"',
        "sec-fetch-dest":            "empty",
        "sec-fetch-mode":            "cors",
        "sec-fetch-site":            "same-origin",
        "anthropic-client-platform": _CLIENT_PLATFORM,
        "anthropic-client-sha":      _CLIENT_SHA,
        "anthropic-client-version":  _CLIENT_VERSION,
        "origin":                    BASE_URL,
        "referer":                   BASE_URL + "/new",
    })
    return s


# ─── Ensure slot has a valid session ──────────────────────────────────────────

def ensure_slot(slot: str, proxy: Optional[str], pool_dir: Optional[Path] = None) -> dict:
    cached = load_slot_session(slot, pool_dir)
    if cached:
        return cached
    sess = playwright_login(slot, proxy, pool_dir)
    save_slot_session(slot, sess["cookies"], sess["org_id"], sess["device_id"],
                      sess["anonymous_id"], sess["expires"], pool_dir)
    return sess


# ─── Conversation API ──────────────────────────────────────────────────────────

def create_conversation(s: curl_requests.Session, org_id: str, device_id: str, model: str = MODEL) -> str:
    conv_id = str(_uuid.uuid4())
    url = BASE_URL + "/api/organizations/{}/chat_conversations".format(org_id)
    r = s.post(url, json={
        "uuid":                             conv_id,
        "name":                             "",
        "model":                            model,
        "include_conversation_preferences": True,
        "is_temporary":                     True,
        "enabled_imagine":                  True,
    }, headers={
        "anthropic-device-id": device_id,
        "content-type":        "application/json",
    }, timeout=15)
    print("[conv] HTTP {}".format(r.status_code), file=sys.stderr)
    if r.status_code not in (200, 201):
        raise RuntimeError("create_conversation: {} {}".format(r.status_code, r.text[:300]))
    return conv_id


# ─── File Upload ───────────────────────────────────────────────────────────────

def upload_file(s: curl_requests.Session, org_id: str, file_bytes: bytes, filename: str,
                mime_type: str = "application/octet-stream") -> dict:
    """Upload a file to Claude and return attachment metadata."""
    url = BASE_URL + "/api/organizations/{}/upload".format(org_id)

    # Claude web uses multipart/form-data
    files = {"file": (filename, file_bytes, mime_type)}

    r = s.post(url, files=files, timeout=60)
    print("[upload] HTTP {}".format(r.status_code), file=sys.stderr)

    if r.status_code not in (200, 201):
        raise RuntimeError("upload_file: {} {}".format(r.status_code, r.text[:300]))

    data = r.json()
    print("[upload] file_id={}".format(data.get("uuid", data.get("file_uuid", "?"))), file=sys.stderr)
    return data


# ─── Slot State ────────────────────────────────────────────────────────────────

class _SlotState:
    """Holds live state for one account slot across turns."""
    def __init__(self, slot: str, sess: dict, proxy: Optional[str]):
        self.slot         = slot
        self.sess         = sess
        self.http         = _make_curl_session(sess["cookies"], proxy)
        self.conv_id:  Optional[str] = None
        self.last_asst_uuid: Optional[str] = None

    @property
    def org_id(self):       return self.sess["org_id"]
    @property
    def device_id(self):    return self.sess["device_id"]
    @property
    def anonymous_id(self): return self.sess["anonymous_id"]

_slot_states: dict[str, _SlotState] = {}

def _get_slot_state(slot: str, proxy: Optional[str]) -> _SlotState:
    if slot not in _slot_states:
        sess = ensure_slot(slot, proxy)
        _slot_states[slot] = _SlotState(slot, sess, proxy)
    return _slot_states[slot]


def _stream_on_slot(state: _SlotState, prompt: str, model: str = MODEL,
                    file_attachments: Optional[List[dict]] = None) -> Generator[str, None, None]:
    """Send prompt on the given slot state. Yields text chunks."""
    slot = state.slot
    s    = state.http

    if state.conv_id is None:
        state.conv_id       = create_conversation(s, state.org_id, state.device_id, model)
        state.last_asst_uuid = None

    human_uuid = uuid7()
    asst_uuid  = uuid7()

    _cfg = MODEL_CONFIGS.get(model, _DEFAULT_MODEL_CONFIG)
    body: dict = {
        "prompt":             prompt,
        "timezone":           os.environ.get("ZAI_TIMEZONE", "Asia/Taipei"),
        "locale":             "en-US",
        "model":              model,
        "thinking_mode":      _cfg["thinking_mode"],
        "attachments":        file_attachments or [],
        "files":              [],
        "sync_sources":       [],
        "rendering_mode":     "messages",
        "turn_message_uuids": {
            "human_message_uuid":     human_uuid,
            "assistant_message_uuid": asst_uuid,
        },
    }
    if _cfg["effort"] is not None:          # Haiku: omit; all others: include
        body["effort"] = _cfg["effort"]
    if state.last_asst_uuid:
        body["parent_message_uuid"] = state.last_asst_uuid

    referer = "{}/chat/{}".format(BASE_URL, state.conv_id) if state.last_asst_uuid else BASE_URL + "/new"

    headers = {
        "accept":                    "text/event-stream",
        "content-type":              "application/json",
        "anthropic-device-id":       state.device_id,
        "referer":                   referer,
    }
    if state.anonymous_id:
        headers["anthropic-anonymous-id"] = state.anonymous_id

    url = BASE_URL + "/api/organizations/{}/chat_conversations/{}/completion".format(
        state.org_id, state.conv_id
    )

    r = s.post(url, json=body, headers=headers, stream=True, timeout=120)
    print("[{}][completion] HTTP {}".format(slot, r.status_code), file=sys.stderr)

    if r.status_code == 401:
        clear_slot_session(slot)
        _slot_states.pop(slot, None)
        raise RuntimeError("[{}] 401 — session expired, cleared".format(slot))

    if r.status_code == 429:
        raise RuntimeError("[{}] 429 — rate limited".format(slot))

    if r.status_code != 200:
        raise RuntimeError("[{}] HTTP {} {}".format(slot, r.status_code, r.text[:300]))

    chunks = []
    got_content = False

    for raw in r.iter_lines():
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        if SSE_DEBUG:
            print("[SSE] {!r}".format(line), file=sys.stderr)
        if not line or line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str:
            continue
        try:
            evt = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        etype = evt.get("type")

        if etype == "content_block_delta":
            delta = evt.get("delta", {})
            if delta.get("type") == "text_delta":
                chunk = delta.get("text", "")
                if chunk:
                    chunks.append(chunk)
                    got_content = True
                    yield chunk

        elif etype == "message_stop":
            if not got_content:
                state.conv_id        = None
                state.last_asst_uuid = None
                raise RuntimeError("[{}] message_stop with no content — session may be expired".format(slot))
            state.last_asst_uuid = asst_uuid
            return

        elif etype == "message_limit":
            ml = evt.get("message_limit", {})
            if ml.get("type") == "exceeded_limit":
                raise RuntimeError("[{}] message limit exceeded".format(slot))

        elif etype == "error":
            err = evt.get("error", {})
            raise RuntimeError("[{}] API error: {}".format(slot, err.get("message", str(err))))

    if chunks:
        state.last_asst_uuid = asst_uuid
        return

    state.conv_id        = None
    state.last_asst_uuid = None
    raise RuntimeError("[{}] SSE stream closed with no content".format(slot))


# ─── Rotating send ─────────────────────────────────────────────────────────────

def send(prompt: str, proxy: Optional[str] = None, model: str = MODEL,
         file_attachments: Optional[List[dict]] = None) -> str:
    """Send prompt and return full response text (non-streaming)."""
    return "".join(_stream(prompt, proxy, model, file_attachments))


def _stream(prompt: str, proxy: Optional[str] = None, model: str = MODEL,
            file_attachments: Optional[List[dict]] = None) -> Generator[str, None, None]:
    """Yield text chunks (streaming)."""
    accounts = list_accounts()
    if not accounts:
        raise RuntimeError("No accounts. Run: python claude_client.py --add-account")

    n     = len(accounts)
    start = get_rotation_index() % n

    for i in range(n):
        idx   = (start + i) % n
        slot  = accounts[idx]
        try:
            state  = _get_slot_state(slot, proxy)
            for chunk in _stream_on_slot(state, prompt, model, file_attachments):
                yield chunk
            advance_rotation(idx)
            return
        except RuntimeError as e:
            msg = str(e)
            if any(x in msg for x in ["429", "401", "expired", "no content"]):
                print("[rotate] {} failed ({}), trying next...".format(slot, msg.split('\n')[0]),
                      file=sys.stderr)
                continue
            raise

    raise RuntimeError("All {} accounts failed. Add more or wait.".format(n))


# ─── CLI ───────────────────────────────────────────────────────────────────────

def cmd_add_account(proxy: Optional[str]):
    slot = next_slot_name()
    print("\nAdding slot: {}".format(slot))
    print("Browser opening — sign in with a DIFFERENT Google account than any already in pool.\n")
    sess = playwright_login(slot, proxy)
    save_slot_session(slot, sess["cookies"], sess["org_id"], sess["device_id"],
                      sess["anonymous_id"], sess["expires"])
    accounts = list_accounts()
    print("\n[pool] {} added. Pool size: {}".format(slot, len(accounts)))

def cmd_show_pool():
    accounts = list_accounts()
    if not accounts:
        print("Pool empty.  python claude_client.py --add-account")
        return
    print("Pool: {} account(s)".format(len(accounts)))
    now = datetime.datetime.now(datetime.timezone.utc)
    for slot in accounts:
        sf = session_file(slot)
        if sf.exists():
            data = json.loads(sf.read_text())
            exp  = datetime.datetime.fromisoformat(data["expires"])
            if exp.tzinfo is None:
                exp = exp.replace(tzinfo=datetime.timezone.utc)
            days_left = (exp - now).days
            ck = data.get("cookies", {})
            has_cf = "cf_clearance" in ck
            has_sk = "sessionKey" in ck
            tag  = "OK ({}d)".format(days_left) if days_left >= REFRESH_DAYS else "EXPIRING"
            print("  {} — org={}... — {} | sessionKey={} cf_clearance={}".format(
                slot, data.get("org_id","?")[:8], tag, has_sk, has_cf))
        else:
            print("  {} — NO SESSION".format(slot))


def main():
    parser = argparse.ArgumentParser(description="claude.ai multi-account rotating client")
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--add-account", action="store_true")
    parser.add_argument("--pool",        action="store_true")
    args = parser.parse_args()

    proxy = detect_proxy()
    print("[proxy] {}".format(proxy if proxy else "none"), file=sys.stderr)

    if args.pool:
        cmd_show_pool(); return

    if args.add_account:
        cmd_add_account(proxy); return

    accounts = list_accounts()
    if not accounts:
        print("\nNo accounts yet.\n  python claude_client.py --add-account\n", file=sys.stderr)
        sys.exit(1)

    print("[pool] {} account(s)".format(len(accounts)), file=sys.stderr)
    print("", file=sys.stderr)

    if args.prompt:
        prompt = " ".join(args.prompt)
        print("[you] {}\n".format(prompt), file=sys.stderr)
        print(send(prompt, proxy))
        return

    print("Claude.ai  |  {} account(s)  |  Ctrl+C or 'exit' to quit".format(len(accounts)))
    print("-" * 60)

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye."); break
        if not prompt or prompt.lower() in ("exit", "quit"):
            break
        print("\nClaude: ", end="", flush=True)
        try:
            for chunk in _stream(prompt, proxy):
                print(chunk, end="", flush=True)
            print()
        except RuntimeError as e:
            print("\n[error] {}".format(e), file=sys.stderr); break


if __name__ == "__main__":
    main()
