from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any

from app.config import settings


def hash_password(password: str, salt: str | None = None) -> str:
    s = salt or secrets.token_hex(8)
    digest = hashlib.sha256(f"{s}:{password}".encode("utf-8")).hexdigest()
    return f"{s}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        salt, _ = password_hash.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(password, salt), password_hash)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def create_access_token(user: dict[str, Any]) -> str:
    payload = {
        "uid": int(user["id"]),
        "username": str(user["username"]),
        "role": str(user.get("role", "student")),
        "exp": int(time.time()) + settings.auth_exp_minutes * 60,
    }
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(settings.auth_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"{_b64url_encode(body)}.{sig}"


def parse_access_token(token: str) -> dict[str, Any] | None:
    try:
        body_b64, sig = token.split(".", 1)
        body = _b64url_decode(body_b64)
        expected = hmac.new(settings.auth_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(body.decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None
