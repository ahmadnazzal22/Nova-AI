from fastapi import APIRouter, HTTPException, Depends

from ...schemas.schemas import RegisterRequest, LoginRequest, TokenResponse, RefreshRequest, UserProfile, UpdateProfileRequest, UpdateSettingsRequest
from ...auth.jwt_handler import create_token_pair, refresh_access_token, hash_password, verify_password
from ...auth.middleware import get_current_user, require_user, require_admin
from ...database import get_session, session_scope, UserRepository
from ...logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Auth"])


@router.post("/auth/register", response_model=TokenResponse)
def register(req: RegisterRequest):
    logger.info("Register attempt: %s / %s", req.username, req.email)
    with session_scope() as session:
        repo = UserRepository(session)
        if repo.get_by_username(req.username):
            logger.warning("Register failed: username %s already exists", req.username)
            raise HTTPException(400, "Username already exists")
        if repo.get_by_email(req.email):
            logger.warning("Register failed: email %s already exists", req.email)
            raise HTTPException(400, "Email already exists")
        user = repo.create(req.username, req.email, hash_password(req.password))
        if not user:
            raise HTTPException(500, "Failed to create user")
        uid, uname, urole = user.id, user.username, user.role
    tokens = create_token_pair(uid, uname, urole)
    logger.info("Register success: %s (id=%d)", req.username, uid)
    return TokenResponse(**tokens)


@router.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest):
    logger.info("Login attempt: %.80s", req.email)
    with session_scope() as session:
        repo = UserRepository(session)
        user = repo.get_by_email(req.email)
        if not user or not verify_password(req.password, user.password_hash):
            logger.warning("Login failed: %s (invalid credentials)", req.email)
            raise HTTPException(401, "Invalid email or password")
        if not user.is_active:
            logger.warning("Login failed: %s (deactivated)", req.email)
            raise HTTPException(403, "Account deactivated")
        repo.update_login(user.id)
        uid, uname, urole = user.id, user.username, user.role
    tokens = create_token_pair(uid, uname, urole)
    logger.info("Login success: %s (id=%d)", req.email, uid)
    return TokenResponse(**tokens)


@router.post("/auth/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest):
    tokens = refresh_access_token(req.refresh_token)
    if not tokens:
        raise HTTPException(401, "Invalid or expired refresh token")
    return TokenResponse(**tokens)


@router.get("/auth/me", response_model=UserProfile)
def auth_me(current_user: dict = Depends(require_user)):
    with get_session() as session:
        repo = UserRepository(session)
        user = repo.get_by_id(current_user["id"])
        if not user:
            raise HTTPException(404, "User not found")
        return UserProfile(
            id=user.id, username=user.username, email=user.email, role=user.role,
            last_login=str(user.last_login) if user.last_login else None,
            created_at=str(user.created_at) if user.created_at else None,
        )


@router.put("/auth/profile", response_model=UserProfile)
def update_profile(req: UpdateProfileRequest, current_user: dict = Depends(require_user)):
    with session_scope() as session:
        repo = UserRepository(session)
        updates = {}
        if req.username:
            existing = repo.get_by_username(req.username)
            if existing and existing.id != current_user["id"]:
                raise HTTPException(400, "Username taken")
            updates["username"] = req.username
        if req.email:
            existing = repo.get_by_email(req.email)
            if existing and existing.id != current_user["id"]:
                raise HTTPException(400, "Email taken")
            updates["email"] = req.email
        user = repo.update_profile(current_user["id"], **updates)
        if not user:
            raise HTTPException(404, "User not found")
        return UserProfile(
            id=user.id, username=user.username, email=user.email, role=user.role,
            last_login=str(user.last_login) if user.last_login else None,
            created_at=str(user.created_at) if user.created_at else None,
        )


@router.get("/auth/settings")
def get_settings(current_user: dict = Depends(require_user)):
    with get_session() as session:
        repo = UserRepository(session)
        settings = repo.get_settings(current_user["id"])
        if not settings:
            return {}
        return {
            "theme": settings.theme,
            "default_mode": settings.default_mode,
            "default_sources": settings.default_sources,
            "streaming_enabled": settings.streaming_enabled,
            "preferences": settings.preferences or {},
        }


@router.put("/auth/settings")
def update_settings(req: UpdateSettingsRequest, current_user: dict = Depends(require_user)):
    with session_scope() as session:
        repo = UserRepository(session)
        updates = {k: v for k, v in req.model_dump(exclude_none=True).items()}
        settings = repo.update_settings(current_user["id"], **updates)
    if not settings:
        raise HTTPException(404, "Settings not found")
    return {"status": "updated"}


@router.delete("/auth/account")
def delete_account(current_user: dict = Depends(require_user)):
    with session_scope() as session:
        repo = UserRepository(session)
        if not repo.deactivate(current_user["id"]):
            raise HTTPException(404, "User not found")
    return {"status": "account deactivated"}
