"""
NVIDIA NIM Proxy
================
A thin, FAST OpenAI-compatible proxy that sits between an OpenAI-compatible
client (e.g. OpenCode) and NVIDIA's NIM API (https://integrate.api.nvidia.com/v1).

Focused on chat completions for GLM.

Performance
-----------
* A SINGLE persistent upstream httpx.AsyncClient with connection POOLING and
  HTTP keep-alive (and HTTP/2) is reused across every request. This avoids a
  fresh TLS handshake to NVIDIA on every call (~100-250ms saved per request),
  which is by far the biggest latency win.
* Raw byte streaming (aiter_raw) — no parsing/re-serializing of SSE chunks.
* Run with uvloop + httptools (see Dockerfile) for a faster event loop + parser.

Features
--------
* POST /v1/chat/completions  - forwards requests to NVIDIA NIM, streaming SSE
  chunks back without buffering the whole response.
* Multi-key rotation across however many NVIDIA_KEY_n variables are set
  (1+ supported) with automatic retry on HTTP 429 and a 15-minute cool-down
  per key (NVIDIA's real reset window).
* Automatic injection of `chat_template_kwargs: {enable_thinking: true}` at the
  ROOT of the payload for any model whose name contains "glm".
* Bearer-token auth (PROXY_AUTH_TOKEN) for the proxy itself.
* GET /health  -> {"status": "ok"}
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Optional, Union

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

# --- Configuration ---------------------------------------------------------
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
COOLDOWN_SECONDS = int(os.environ.get("KEY_COOLDOWN_SECONDS", str(15 * 60)))  # 15 minutes

# Tight connect timeout (NIM is fast once warm); generous read timeout for long
# streams. Small pool timeout so we never wait long for a free connection.
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0)
# Reuse up to 100 warm keep-alive connections to NVIDIA -> no TLS handshake on
# repeat requests. keepalive_expiry lets idle connections live 2 minutes.
UPSTREAM_LIMITS = httpx.Limits(
    max_keepalive_connections=100, max_connections=200, keepalive_expiry=120.0
)

PROXY_AUTH_TOKEN = os.environ.get("PROXY_AUTH_TOKEN", "").strip()

# Collect every NVIDIA_KEY_n that is actually present and non-empty.
# Supports a VARIABLE number of keys (1+): it scans NVIDIA_KEY_1..NVIDIA_KEY_8,
# skips any that are missing or blank, and rotates only over the valid ones.
NVIDIA_KEYS: list[dict] = []
for _i in range(1, 9):
    _k = os.environ.get(f"NVIDIA_KEY_{_i}")
    if _k and _k.strip():  # skip missing OR empty/whitespace-only values
        NVIDIA_KEYS.append({"index": _i, "key": _k.strip()})

# key index -> epoch time until which the key must not be used (cool-down).
cooldown_state: dict[int, float] = {}
_rr = 0  # round-robin pointer so successive requests start on different keys

# Single shared upstream client (created in lifespan). Persistent connection
# pool = no per-request TLS handshake to NVIDIA = much lower latency.
_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create one pooled, keep-alive HTTP/1.1 client for the whole app lifetime.

    NOTE: HTTP/2 (h2, pure Python) was tried but its flow-control window BUFFERS
    streaming chunks and cut throughput dramatically. HTTP/1.1 over h11 streams
    each token immediately while still pooling connections (no per-request TLS
    handshake) — best for high-throughput SSE.
    """
    global _client
    _client = httpx.AsyncClient(
        http2=False,  # HTTP/1.1: lighter parser, no flow-control buffering
        timeout=UPSTREAM_TIMEOUT,
        limits=UPSTREAM_LIMITS,
    )
    try:
        yield
    finally:
        if _client is not None:
            await _client.aclose()


app = FastAPI(title="NVIDIA NIM Proxy", version="2.0.0", lifespan=lifespan)


# --- Helpers ---------------------------------------------------------------
def _err(status: int, message: str, type_: str = "invalid_request_error", **extra) -> JSONResponse:
    body = {"error": {"message": message, "type": type_}}
    body["error"].update(extra)
    return JSONResponse(status_code=status, content=body)


def _check_auth(authorization: Optional[str]) -> Optional[JSONResponse]:
    """Validate the proxy bearer token. Returns an error response or None."""
    if not PROXY_AUTH_TOKEN:
        return _err(401, "PROXY_AUTH_TOKEN is not configured on the server.", "unauthorized")
    if not authorization:
        return _err(
            401,
            "Missing 'Authorization' header. Expected 'Authorization: Bearer <PROXY_AUTH_TOKEN>'.",
            "unauthorized",
        )
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return _err(401, "Malformed Authorization header. Use 'Bearer <token>'.", "unauthorized")
    if parts[1].strip() != PROXY_AUTH_TOKEN:
        return _err(401, "Invalid bearer token.", "unauthorized")
    return None


def _build_payload(body: dict) -> dict:
    """Normalize the raw request dict before forwarding to NVIDIA.

    The incoming body is forwarded EXACTLY as received from the client. The only
    mutations are two fixes for NVIDIA NIM compatibility:

    1. NVIDIA NIM rejects the nested ``extra_body`` wrapper with
       ``Unsupported parameter(s): extra_body``. If present, pop it out and
       merge its contents to the ROOT of the body (so e.g.
       ``extra_body["chat_template_kwargs"]`` becomes root-level
       ``chat_template_kwargs``).

    2. For any model whose name contains "glm", ensure
       ``chat_template_kwargs: {"enable_thinking": true}`` exists at the root,
       UNLESS the client explicitly set enable_thinking (then respect it).
       Sending enable_thinking=false skips GLM's reasoning trace for ~3-4x
       faster replies (the speed knob).

    Every other field passes through completely unmodified, including (but not
    limited to): tools, tool_choice, messages, system, temperature, max_tokens,
    stream, and any unknown fields. ``messages`` content may be a plain string
    OR a list of parts (multimodal); both are forwarded untouched.
    """
    # 1. Flatten extra_body -> root (NVIDIA rejects the wrapper itself).
    extra = body.pop("extra_body", None)
    if isinstance(extra, dict):
        body.update(extra)

    # 2. GLM reasoning parameter at the root.
    #    Default: thinking ON (matches original requirement). But if the client
    #    explicitly sets enable_thinking, RESPECT it -- sending false skips the
    #    long reasoning trace and makes GLM reply ~3-4x faster (content-only).
    #    This is the speed knob: {"chat_template_kwargs":{"enable_thinking":false}}
    if "glm" in str(body.get("model", "") or "").lower():
        existing = body.get("chat_template_kwargs")
        if isinstance(existing, dict):
            existing.setdefault("enable_thinking", True)
            body["chat_template_kwargs"] = existing
        else:
            body["chat_template_kwargs"] = {"enable_thinking": True}

    return body


async def _forward(resp: httpx.Response):
    """Stream the upstream response through unchanged (raw bytes)."""
    try:
        async for chunk in resp.aiter_raw():
            if chunk:
                yield chunk
    finally:
        try:
            await resp.aclose()
        except Exception:
            pass


async def _post_with_rotation(
    url: str, payload: dict, accept: str = "text/event-stream"
) -> Union[JSONResponse, httpx.Response]:
    """Send ``payload`` to an upstream NVIDIA ``url`` using the shared key pool.

    Uses the single pooled client (no per-request TLS handshake). Rotates across
    NVIDIA_KEY_1..8, retries on HTTP 429 / 5xx, and applies the 15-minute per-key
    cool-down on real 429s. Returns either a JSONResponse (error) or an OPEN
    streamed 2xx response the caller must consume and close.
    """
    if not NVIDIA_KEYS:
        return _err(503, "No NVIDIA API keys are configured on the server.", "server_error")
    if _client is None:  # pragma: no cover - lifespan always sets it
        return _err(503, "Upstream client not ready.", "server_error")

    global _rr
    n = len(NVIDIA_KEYS)
    order = [NVIDIA_KEYS[(_rr + i) % n] for i in range(n)]
    _rr = (_rr + 1) % n

    now = time.time()
    available = [e for e in order if now >= cooldown_state.get(e["index"], 0.0)]
    if not available:
        return _err(
            429,
            f"All {n} NVIDIA API keys are cooling down after rate-limit (429) responses. "
            f"They become available again after {COOLDOWN_SECONDS // 60} minutes. Please retry later.",
            "rate_limit_exceeded",
        )

    last_status: Optional[int] = None
    last_detail = "No upstream response."

    for entry in available:
        idx = entry["index"]
        headers = {
            "Authorization": f"Bearer {entry['key']}",
            "Content-Type": "application/json",
            "Accept": accept,
            # CRITICAL for streaming throughput: force NO compression. If NVIDIA
            # gzip-deflates the SSE, the bytes accumulate in a compression buffer
            # and are NOT forwarded token-by-token -> throughput collapses.
            "Accept-Encoding": "identity",
            "User-Agent": "nvidia-nim-proxy/2.0",
        }
        try:
            req = _client.build_request("POST", url, json=payload, headers=headers)
            resp = await _client.send(req, stream=True)
        except Exception as e:  # network / DNS / connect failure
            last_status = 502
            last_detail = f"Failed to reach NVIDIA upstream: {e}"
            continue  # shared client stays alive; just try next key

        status = resp.status_code
        if status == 429 or status >= 500:
            # Transient: rotate to the next key. Cool-down only on real 429s.
            if status == 429:
                cooldown_state[idx] = time.time() + COOLDOWN_SECONDS
            try:
                last_detail = (await resp.aread()).decode("utf-8", errors="replace")[:2000]
            except Exception:
                last_detail = f"Upstream returned HTTP {status}."
            last_status = status
            await resp.aclose()
            continue
        elif status >= 400:
            # Non-transient client error (e.g. 400 bad model) - surface immediately.
            try:
                detail = (await resp.aread()).decode("utf-8", errors="replace")[:2000]
            except Exception:
                detail = f"Upstream returned HTTP {status}."
            await resp.aclose()
            return _err(status, detail, "upstream_error", upstream_status=status)
        else:
            # Success - hand the OPEN response back to the caller.
            return resp

    return _err(
        last_status or 502,
        f"All available NVIDIA API keys failed. Last upstream response: "
        f"HTTP {last_status} - {last_detail}",
        "upstream_error",
        upstream_status=last_status,
    )


# --- Routes ----------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "keys_configured": len(NVIDIA_KEYS)}


@app.get("/")
async def root():
    return {
        "service": "nvidia-nim-proxy",
        "endpoints": {"health": "/health", "chat": "/v1/chat/completions"},
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # 1. Proxy auth
    auth_err = _check_auth(request.headers.get("authorization"))
    if auth_err:
        return auth_err

    # 2. Parse body as a raw dict (accept ANY fields, no Pydantic model).
    try:
        body = await request.json()
    except Exception:
        return _err(400, "Request body must be valid JSON.")
    if not isinstance(body, dict):
        return _err(400, "Request body must be a JSON object.")

    payload = _build_payload(body)
    is_stream = bool(payload.get("stream", False))

    # 3. Forward through the shared key-rotation pool. Match the upstream Accept
    #    header to the stream flag so NIM returns the right content-type.
    accept = "text/event-stream" if is_stream else "application/json"
    result = await _post_with_rotation(f"{NVIDIA_BASE_URL}/chat/completions", payload, accept=accept)
    if isinstance(result, JSONResponse):
        return result

    # 4. Success - stream straight through to the client.
    content_type = result.headers.get(
        "content-type", "text/event-stream" if is_stream else "application/json"
    )
    return StreamingResponse(
        _forward(result),
        media_type=content_type,
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable HF/nginx buffering of SSE
            "Connection": "keep-alive",
            # Hint to any intermediary that this is a real-time stream.
            "Content-Encoding": "identity",
        },
    )
