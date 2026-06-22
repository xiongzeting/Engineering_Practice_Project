"""认证工具：密码哈希与自签 token。

设计上刻意保持「零外部依赖」——不用 ``passlib``、不用 ``python-jose``，
只用标准库的 ``hashlib`` + ``hmac`` + ``base64``。这样部署环境最小化，
代价是算法相对简单（详见各函数说明），对本课程项目足够。

- **密码**：SHA-256 + 8 字节随机盐。``hash_password`` 返回 ``salt$digest``
  字符串，整串存数据库；:func:`verify_password` 用 ``hmac.compare_digest``
  做常量时间比较，防时序攻击。
- **Token**：自签的 ``<base64url(payload)>.<hex_sig>``，payload 里含
  ``uid`` / ``username`` / ``role`` / ``exp``，签名用 ``settings.auth_secret``
  做 HMAC-SHA256。前端把 token 放 ``Authorization: Bearer <token>``。
"""
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
    """把明文密码哈希成 ``salt$digest`` 字符串。

    ``salt`` 为空时自动生成 8 字节随机盐（16 个十六进制字符）。
    使用 SHA-256(``salt:password``) 作为摘要——不是 bcrypt/argon2，
    但对课程项目足够，并且无外部依赖。

    返回值整串存数据库的 ``password_hash`` 字段，验证时用同样的盐重算。
    """
    s = salt or secrets.token_hex(8)
    digest = hashlib.sha256(f"{s}:{password}".encode("utf-8")).hexdigest()
    return f"{s}${digest}"


def verify_password(password: str, password_hash: str) -> bool:
    """验证明文密码是否匹配库里存的 ``salt$digest``。

    先拆出 salt，用相同 salt 重新哈希一遍，再走 :func:`hmac.compare_digest`
    做**常量时间比较**——避免攻击者根据响应时间差判断正确长度（时序攻击）。

    格式异常（没找到 ``$``）直接返回 False，不抛异常。
    """
    try:
        salt, _ = password_hash.split("$", 1)
    except ValueError:
        return False
    return hmac.compare_digest(hash_password(password, salt), password_hash)


def _b64url_encode(data: bytes) -> str:
    """URL 安全的 base64 编码，去掉 ``=`` 填充（JWT 风格）。"""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    """对应 :func:`_b64url_encode` 的解码，先把 ``=`` 填充补齐再解码。"""
    pad = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + pad)


def create_access_token(user: dict[str, Any]) -> str:
    """为登录用户签发 access token。

    token 格式：``<base64url(payload)>.<hex_sig>``，其中 payload 含：
    - ``uid``：用户 id；
    - ``username``：用户名（供前端展示）；
    - ``role``：角色（student/teacher/admin，决定可访问的接口）；
    - ``exp``：过期时间戳（秒），由 ``settings.auth_exp_minutes``（默认 24h）推出。

    签名算法为 HMAC-SHA256，密钥来自 ``settings.auth_secret``。签名时
    JSON body 用 ``sort_keys=True`` + 极简分隔符，保证编/解码字节一致。
    """
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
    """校验并解析 access token。

    校验流程（任一步失败都返回 ``None``，不抛异常）：
    1. 按 ``.`` 拆 body / sig；
    2. 用同样的密钥重算 HMAC，与 token 里的 sig 做**常量时间比较**；
    3. base64url 解码 body 得到 JSON；
    4. 检查 ``exp`` 是否已过期。

    通过则返回 payload dict，调用方可以直接读 ``payload['uid']`` / ``payload['role']``。
    """
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
