import base64
import json
import os

# main.py intentionally requires an account at import time. Tests use a
# credential-free stub so the application module can be imported without
# touching real session material.
os.environ.setdefault(
    "CLAUDE_ACCOUNT_01",
    base64.b64encode(json.dumps({"cookies": {}}).encode()).decode(),
)
os.environ.setdefault("CLAUDE_PROTOCOL_CAPTURE", "0")
os.environ.setdefault("PROBE_TOOL_USE", "0")
