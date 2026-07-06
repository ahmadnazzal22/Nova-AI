import os
import time
import hmac
import hashlib
import base64
import json as jsonlib
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .database import get_session, UserRepository
from .logger import get_logger

logger = get_logger(__name__)

_SECRET = os.getenv("JWT_SECRET", "change-me-in-production-rag-system-2024")
_ACCESS_EXPIRE = int(os.getenv("JWT_ACCESS_EXPIRE_SECONDS", "3600"))
_REFRESH_EXPIRE = int(os.getenv("JWT_REFRESH_EXPIRE_SECONDS", "2592000"))

bearer_scheme = HTTPBearer(auto_error=False)


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _sign(payload: str, secret: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _make_token(payload: dict) -> str:
    header = _b64url_encode(jsonlib.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = _b64url_encode(jsonlib.dumps(payload).encode())
    signature = _sign(f"{header}.{body}", _SECRET)
    return f"{header}.{body}.{signature}"


def _parse_token(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, payload, signature = parts
        expected = _sign(f"{header}.{payload}", _SECRET)
        if not hmac.compare_digest(signature, expected):
            return None
        decoded = jsonlib.loads(_b64url_decode(payload))
        if decoded.get("exp", 0) < time.time():
            return None
        return decoded
    except Exception as e:
        logger.debug("Token decode failed: %s", e)
        return None


def create_access_token(user_id: int, username: str, role: str) -> str:
    return _make_token({
        "sub": user_id,
        "username": username,
        "role": role,
        "type": "access",
        "iat": int(time.time()),
        "exp": int(time.time()) + _ACCESS_EXPIRE,
    })


def create_refresh_token(user_id: int, username: str, role: str) -> str:
    return _make_token({
        "sub": user_id,
        "username": username,
        "role": role,
        "type": "refresh",
        "iat": int(time.time()),
        "exp": int(time.time()) + _REFRESH_EXPIRE,
    })


def create_token_pair(user_id: int, username: str, role: str) -> dict:
    return {
        "access_token": create_access_token(user_id, username, role),
        "refresh_token": create_refresh_token(user_id, username, role),
        "token_type": "bearer",
    }


def refresh_access_token(refresh_token: str) -> Optional[dict]:
    payload = _parse_token(refresh_token)
    if payload is None:
        return None
    if payload.get("type") != "refresh":
        return None
    return create_token_pair(payload["sub"], payload["username"], payload["role"])


def hash_password(password: str) -> str:
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    import bcrypt
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except Exception:
        return False


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> Optional[dict]:
    if credentials is None:
        return None
    payload = _parse_token(credentials.credentials)
    if payload is None:
        return None
    if payload.get("type") != "access":
        return None
    with get_session() as session:
        repo = UserRepository(session)
        user = repo.get_by_id(payload.get("sub", 0))
    if user is None or not user.is_active:
        return None
    return {"id": user.id, "username": user.username, "email": user.email, "role": user.role}


async def require_user(current_user: Optional[dict] = Depends(get_current_user)):
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return current_user


async def require_admin(current_user: dict = Depends(require_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user
