from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .jwt_handler import parse_token
from ..database import get_session, UserRepository

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)) -> Optional[dict]:
    if credentials is None:
        return None
    payload = parse_token(credentials.credentials)
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
