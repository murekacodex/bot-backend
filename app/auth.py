from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import get_settings
from app.models import CreateUserRequest, LoginRequest, UpdateUserRequest, UserPublic


security = HTTPBearer(auto_error=False)
_lock = threading.Lock()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _state_path() -> Path:
    return Path(get_settings().auth_state_path)


def _read_state() -> dict[str, list[dict[str, Any]]]:
    path = _state_path()
    if not path.exists():
        return {"users": []}
    try:
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)
    except (OSError, json.JSONDecodeError):
        return {"users": []}
    users = state.get("users")
    normalized_state = {"users": users if isinstance(users, list) else []}
    _coerce_single_admin(normalized_state)
    return normalized_state


def _coerce_single_admin(state: dict[str, list[dict[str, Any]]]) -> None:
    admin_seen = False
    for user in state["users"]:
        if not user.get("is_admin"):
            continue
        if admin_seen:
            user["is_admin"] = False
        else:
            admin_seen = True


def _write_state(state: dict[str, list[dict[str, Any]]]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _coerce_single_admin(state)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, sort_keys=True)
    temporary.replace(path)


def _normalize_username(username: str) -> str:
    normalized = username.strip().lower()
    if len(normalized) < 3:
        raise HTTPException(status_code=422, detail="Username must be at least 3 characters")
    return normalized


def _hash_password(password: str, salt: str | None = None) -> str:
    selected_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(selected_salt), 200_000)
    return f"{selected_salt}${digest.hex()}"


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        salt, expected = password_hash.split("$", maxsplit=1)
    except ValueError:
        return False
    calculated = _hash_password(password, salt).split("$", maxsplit=1)[1]
    return hmac.compare_digest(calculated, expected)


def _public_user(user: dict[str, Any]) -> UserPublic:
    return UserPublic(
        id=str(user["id"]),
        username=str(user["username"]),
        is_admin=bool(user.get("is_admin")),
        is_active=bool(user.get("is_active", True)),
        created_at=str(user["created_at"]),
        updated_at=str(user["updated_at"]),
        last_login_at=user.get("last_login_at"),
    )


def _find_user(state: dict[str, list[dict[str, Any]]], username: str) -> dict[str, Any] | None:
    for user in state["users"]:
        if user.get("username") == username:
            return user
    return None


def users_exist() -> bool:
    with _lock:
        return bool(_read_state()["users"])


def login_or_create_admin(payload: LoginRequest) -> tuple[str, UserPublic, bool]:
    username = _normalize_username(payload.username)
    timestamp = _now().isoformat()

    with _lock:
        state = _read_state()
        setup_admin = False

        if not state["users"]:
            user = {
                "id": str(uuid4()),
                "username": username,
                "password_hash": _hash_password(payload.password),
                "is_admin": True,
                "is_active": True,
                "created_at": timestamp,
                "updated_at": timestamp,
                "last_login_at": timestamp,
            }
            state["users"].append(user)
            _write_state(state)
            setup_admin = True
        else:
            user = _find_user(state, username)
            if not user or not _verify_password(payload.password, str(user.get("password_hash", ""))):
                raise HTTPException(status_code=401, detail="Invalid username or password")
            if not user.get("is_active", True):
                raise HTTPException(status_code=403, detail="User access is disabled")
            user["last_login_at"] = timestamp
            user["updated_at"] = timestamp
            _write_state(state)

        public = _public_user(user)
        return create_token(public), public, setup_admin


def _sign(payload: str) -> str:
    secret = get_settings().auth_secret_key.encode("utf-8")
    return hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def create_token(user: UserPublic) -> str:
    expires_at = _now() + timedelta(hours=get_settings().auth_token_ttl_hours)
    payload = {
        "sub": user.id,
        "username": user.username,
        "is_admin": user.is_admin,
        "exp": int(expires_at.timestamp()),
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).decode("ascii")
    return f"{encoded}.{_sign(encoded)}"


def _decode_token(token: str) -> dict[str, Any]:
    try:
        encoded, signature = token.split(".", maxsplit=1)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc
    if not hmac.compare_digest(_sign(encoded), signature):
        raise HTTPException(status_code=401, detail="Invalid token")
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=401, detail="Invalid token") from exc
    if int(payload.get("exp", 0)) <= int(_now().timestamp()):
        raise HTTPException(status_code=401, detail="Session expired")
    return payload


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> UserPublic:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    payload = _decode_token(credentials.credentials)
    with _lock:
        state = _read_state()
        user = next((item for item in state["users"] if item.get("id") == payload.get("sub")), None)
    if not user:
        raise HTTPException(status_code=401, detail="User no longer exists")
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="User access is disabled")
    return _public_user(user)


def admin_user(user: UserPublic = Depends(current_user)) -> UserPublic:
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def list_users() -> list[UserPublic]:
    with _lock:
        return [_public_user(user) for user in _read_state()["users"]]


def create_user(payload: CreateUserRequest) -> UserPublic:
    username = _normalize_username(payload.username)
    timestamp = _now().isoformat()
    with _lock:
        state = _read_state()
        if _find_user(state, username):
            raise HTTPException(status_code=409, detail="Username already exists")
        user = {
            "id": str(uuid4()),
            "username": username,
            "password_hash": _hash_password(payload.password),
            "is_admin": False,
            "is_active": payload.is_active,
            "created_at": timestamp,
            "updated_at": timestamp,
            "last_login_at": None,
        }
        state["users"].append(user)
        _write_state(state)
        return _public_user(user)


def update_user(user_id: str, payload: UpdateUserRequest, acting_user: UserPublic) -> UserPublic:
    with _lock:
        state = _read_state()
        user = next((item for item in state["users"] if item.get("id") == user_id), None)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        next_is_active = user.get("is_active", True) if payload.is_active is None else payload.is_active

        if payload.is_admin is not None and payload.is_admin != bool(user.get("is_admin")):
            raise HTTPException(status_code=400, detail="Only the initial admin account can be an admin")
        if user.get("id") == acting_user.id and not next_is_active:
            raise HTTPException(status_code=400, detail="Admins cannot remove their own access")

        if payload.password:
            user["password_hash"] = _hash_password(payload.password)
        if payload.is_active is not None:
            user["is_active"] = payload.is_active
        user["updated_at"] = _now().isoformat()
        _write_state(state)
        return _public_user(user)


def delete_user(user_id: str, acting_user: UserPublic) -> None:
    with _lock:
        state = _read_state()
        user = next((item for item in state["users"] if item.get("id") == user_id), None)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if user.get("id") == acting_user.id:
            raise HTTPException(status_code=400, detail="Admins cannot delete themselves")
        state["users"] = [item for item in state["users"] if item.get("id") != user_id]
        _write_state(state)
