"""Доверенный контекст владельца; не входит в схемы инструментов для модели."""

import hashlib
import hmac
import json
import time
from typing import Any


class IdentityError(ValueError):
    """Безопасный отказ при отсутствии доверенного контекста."""


def _digest(secret: str, scope: str, arguments: dict, context: dict) -> str:
    data = json.dumps(
        {"scope": scope, "arguments": arguments, "context": context},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hmac.new(secret.encode("utf-8"), data, hashlib.sha256).hexdigest()


def sign_context(
    secret: str,
    user_id: int,
    scope: str,
    arguments: dict,
    *,
    action_id: str | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    if len(secret) < 32 or type(user_id) is not int or user_id <= 0:
        raise IdentityError("Отсутствует доверенный контекст пользователя.")
    context = {
        "user_id": user_id,
        "issued_at": int(time.time() if now is None else now),
        "action_id": action_id,
    }
    return {"study_identity": {**context, "signature": _digest(secret, scope, arguments, context)}}


def verify_context(
    secret: str,
    meta: dict | None,
    scope: str,
    arguments: dict,
    *,
    now: float | None = None,
) -> tuple[int, str | None]:
    try:
        raw = (meta or {})["study_identity"]
        if not isinstance(raw, dict) or set(raw) != {
            "user_id",
            "issued_at",
            "action_id",
            "signature",
        }:
            raise ValueError
        user_id, issued_at, action_id = raw["user_id"], raw["issued_at"], raw["action_id"]
        if (
            len(secret) < 32
            or type(user_id) is not int
            or user_id <= 0
            or type(issued_at) is not int
            or not isinstance(raw["signature"], str)
            or (action_id is not None and not isinstance(action_id, str))
        ):
            raise ValueError
        instant = time.time() if now is None else now
        if not -5 <= instant - issued_at <= 60:
            raise ValueError
        context = {key: raw[key] for key in ("user_id", "issued_at", "action_id")}
        if not hmac.compare_digest(raw["signature"], _digest(secret, scope, arguments, context)):
            raise ValueError
        return user_id, action_id
    except (KeyError, TypeError, ValueError, OverflowError):
        raise IdentityError(
            "Отсутствует или недействителен доверенный контекст пользователя."
        ) from None
