import os
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base, User, UserSettings, Conversation, Message, Feedback, UserMemory, KnowledgeBase, KBUserPermission
from .logger import get_logger

logger = get_logger(__name__)

DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag.db"))
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized: %s", DB_URL)


def get_session() -> Session:
    return SessionLocal()


@contextmanager
def session_scope():
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Repository: User ──────────────────────────────────────────────

class UserRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, username: str, email: str, password_hash: str, role: str = "user") -> User | None:
        try:
            user = User(username=username, email=email, password_hash=password_hash, role=role)
            self.session.add(user)
            self.session.flush()
            settings = UserSettings(user_id=user.id)
            self.session.add(settings)
            self.session.flush()
            logger.info("User created: %s (id=%d)", username, user.id)
            return user
        except Exception as e:
            logger.warning("User creation failed: %s", e)
            return None

    def get_by_id(self, user_id: int) -> User | None:
        return self.session.get(User, user_id)

    def get_by_username(self, username: str) -> User | None:
        return self.session.query(User).filter(User.username == username).first()

    def get_by_email(self, email: str) -> User | None:
        return self.session.query(User).filter(User.email == email).first()

    def list(self, page: int = 1, per_page: int = 20) -> tuple[list[User], int]:
        total = self.session.query(User).count()
        users = self.session.query(User).order_by(User.id.desc()).offset((page - 1) * per_page).limit(per_page).all()
        return users, total

    def update_login(self, user_id: int):
        user = self.get_by_id(user_id)
        if user:
            user.last_login = datetime.now(timezone.utc)
            self.session.flush()

    def update_profile(self, user_id: int, **kwargs) -> User | None:
        user = self.get_by_id(user_id)
        if not user:
            return None
        allowed = {"username", "email", "password_hash"}
        for k, v in kwargs.items():
            if k in allowed:
                setattr(user, k, v)
        self.session.flush()
        return user

    def deactivate(self, user_id: int) -> bool:
        user = self.get_by_id(user_id)
        if not user:
            return False
        user.is_active = False
        self.session.flush()
        return True

    def delete(self, user_id: int) -> bool:
        user = self.get_by_id(user_id)
        if not user:
            return False
        self.session.delete(user)
        self.session.flush()
        return True

    def count_active(self) -> int:
        return self.session.query(User).filter(User.is_active == True).count()

    def get_settings(self, user_id: int) -> UserSettings | None:
        return self.session.query(UserSettings).filter(UserSettings.user_id == user_id).first()

    def update_settings(self, user_id: int, **kwargs) -> UserSettings | None:
        settings = self.get_settings(user_id)
        if not settings:
            return None
        allowed = {"theme", "default_mode", "default_sources", "streaming_enabled", "preferences"}
        for k, v in kwargs.items():
            if k in allowed:
                setattr(settings, k, v)
        self.session.flush()
        return settings


# ── Repository: Conversation ──────────────────────────────────────

class ConversationRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, user_id: int, title: str = "New Chat") -> Conversation:
        conv = Conversation(user_id=user_id, title=title)
        self.session.add(conv)
        self.session.flush()
        return conv

    def get_by_id(self, conv_id: int) -> Conversation | None:
        return self.session.get(Conversation, conv_id)

    def list_by_user(self, user_id: int, limit: int = 50) -> list[Conversation]:
        return self.session.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(Conversation.updated_at.desc()).limit(limit).all()

    def rename(self, conv_id: int, title: str) -> Conversation | None:
        conv = self.get_by_id(conv_id)
        if conv:
            conv.title = title
            self.session.flush()
        return conv

    def delete(self, conv_id: int) -> bool:
        conv = self.get_by_id(conv_id)
        if not conv:
            return False
        self.session.delete(conv)
        self.session.flush()
        return True

    def count_by_user(self, user_id: int) -> int:
        return self.session.query(Conversation).filter(Conversation.user_id == user_id).count()


# ── Repository: Message ───────────────────────────────────────────

class MessageRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, conversation_id: int, role: str, content: str, **kwargs) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            sources=kwargs.get("sources", []),
            related_questions=kwargs.get("related_questions", []),
            live=kwargs.get("live", False),
            cached=kwargs.get("cached", False),
            model=kwargs.get("model", ""),
            response_time=kwargs.get("response_time", 0.0),
            token_count=kwargs.get("token_count", 0),
            error=kwargs.get("error", False),
        )
        self.session.add(msg)
        self.session.flush()
        conv = self.session.get(Conversation, conversation_id)
        if conv:
            conv.updated_at = datetime.now(timezone.utc)
        return msg

    def list_by_conversation(self, conversation_id: int) -> list[Message]:
        return self.session.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.id).all()

    def get_by_id(self, msg_id: int) -> Message | None:
        return self.session.get(Message, msg_id)


# ── Repository: Feedback ──────────────────────────────────────────

class FeedbackRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, message_id: int, user_id: int | None, rating: int, comment: str = "") -> Feedback | None:
        try:
            fb = Feedback(message_id=message_id, user_id=user_id, rating=rating, comment=comment)
            self.session.add(fb)
            self.session.flush()
            logger.info("Feedback: msg=%d rating=%d", message_id, rating)
            return fb
        except Exception as e:
            logger.warning("Feedback failed: %s", e)
            return None

    def get_by_message(self, message_id: int) -> Feedback | None:
        return self.session.query(Feedback).filter(Feedback.message_id == message_id).first()

    def get_stats(self) -> dict:
        from sqlalchemy import func
        stats = {}
        stats["total_feedback"] = self.session.query(func.count(Feedback.id)).scalar() or 0
        stats["avg_rating"] = round(self.session.query(func.avg(Feedback.rating)).scalar() or 0, 2)
        stats["rating_distribution"] = {
            str(i): self.session.query(func.count(Feedback.id)).filter(Feedback.rating == i).scalar() or 0
            for i in range(1, 6)
        }
        return stats


# ── Repository: System Stats ──────────────────────────────────────

class StatsRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_system_stats(self) -> dict:
        from sqlalchemy import func
        uc = UserRepository(self.session)
        total_users = self.session.query(func.count(User.id)).scalar() or 0
        active_users = uc.count_active()
        total_convs = self.session.query(func.count(Conversation.id)).scalar() or 0
        total_msgs = self.session.query(func.count(Message.id)).scalar() or 0
        total_fb = self.session.query(func.count(Feedback.id)).scalar() or 0
        avg_rating = round(self.session.query(func.avg(Feedback.rating)).scalar() or 0, 2)
        live_count = self.session.query(func.count(Message.id)).filter(Message.live == True).scalar() or 0
        cached_count = self.session.query(func.count(Message.id)).filter(Message.cached == True).scalar() or 0
        error_count = self.session.query(func.count(Message.id)).filter(Message.error == True).scalar() or 0
        avg_response = round(self.session.query(func.avg(Message.response_time)).scalar() or 0, 2)
        total_tokens = self.session.query(func.sum(Message.token_count)).scalar() or 0

        top_questions = self.session.query(
            Message.content, func.count(Message.id).label("cnt")
        ).filter(Message.role == "user").group_by(Message.content).order_by(text("cnt DESC")).limit(10).all()

        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_conversations": total_convs,
            "total_messages": total_msgs,
            "total_feedback": total_fb,
            "avg_rating": avg_rating,
            "live_queries": live_count,
            "cached_queries": cached_count,
            "error_count": error_count,
            "avg_response_time_s": avg_response,
            "total_tokens_used": total_tokens,
            "top_questions": [{"question": q[0], "count": q[1]} for q in top_questions],
        }

# ── Repository: Memory ────────────────────────────────────────────

class MemoryRepository:
    def __init__(self, session: Session):
        self.session = session

    def store(self, user_id: int, key: str, value: str, importance: float = 0.5) -> UserMemory:
        # Check for existing memory with same key — update if found
        existing = self.get_by_key(user_id, key)
        if existing:
            existing.value = value
            existing.importance = max(existing.importance, importance)
            existing.created_at = datetime.now(timezone.utc)
            self.session.flush()
            return existing
        memory = UserMemory(user_id=user_id, key=key, value=value, importance=importance)
        self.session.add(memory)
        self.session.flush()
        return memory

    def get_recent(self, user_id: int, limit: int = 10) -> list[UserMemory]:
        return self.session.query(UserMemory).filter(
            UserMemory.user_id == user_id
        ).order_by(UserMemory.importance.desc(), UserMemory.created_at.desc()).limit(limit).all()

    def search(self, user_id: int, keyword: str) -> list[UserMemory]:
        return self.session.query(UserMemory).filter(
            UserMemory.user_id == user_id,
            UserMemory.value.contains(keyword),
        ).all()

    def get_by_key(self, user_id: int, key: str) -> UserMemory | None:
        return self.session.query(UserMemory).filter(
            UserMemory.user_id == user_id,
            UserMemory.key == key,
        ).first()

    def delete(self, memory_id: int) -> bool:
        mem = self.session.get(UserMemory, memory_id)
        if not mem:
            return False
        self.session.delete(mem)
        self.session.flush()
        return True

    def delete_by_key(self, user_id: int, key: str) -> bool:
        mem = self.get_by_key(user_id, key)
        if not mem:
            return False
        self.session.delete(mem)
        self.session.flush()
        return True

    def count(self, user_id: int) -> int:
        return self.session.query(UserMemory).filter(UserMemory.user_id == user_id).count()

    def store_batch(self, user_id: int, memories: list[dict]) -> list[UserMemory]:
        stored = []
        for m in memories:
            key = str(m.get("key", ""))
            value = str(m.get("value", ""))
            importance = float(m.get("importance", 0.5))
            mem = self.store(user_id, key, value, importance)
            stored.append(mem)
        return stored

    def get_top(self, user_id: int, limit: int = 10) -> list[UserMemory]:
        """Return highest-importance + most-recent memories, with diversity."""
        all_memories = self.session.query(UserMemory).filter(
            UserMemory.user_id == user_id
        ).order_by(UserMemory.importance.desc(), UserMemory.created_at.desc()).all()
        # Dedup by key prefix — keep the highest importance per semantic group
        seen_keys: set[str] = set()
        result = []
        for m in all_memories:
            prefix = m.key.split("_")[0] if "_" in m.key else m.key
            if prefix not in seen_keys:
                seen_keys.add(prefix)
                result.append(m)
            if len(result) >= limit:
                break
        return result


class KnowledgeBaseRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, user_id: int, name: str, description: str = "", collection_name: str = "", is_public: bool = False) -> KnowledgeBase:
        import uuid
        kb = KnowledgeBase(
            user_id=user_id,
            name=name,
            description=description,
            collection_name=collection_name or f"kb_{uuid.uuid4().hex[:12]}",
            is_public=is_public,
        )
        self.session.add(kb)
        self.session.flush()
        return kb

    def get(self, kb_id: int) -> KnowledgeBase | None:
        return self.session.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()

    def list_for_user(self, user_id: int) -> list[KnowledgeBase]:
        return self.session.query(KnowledgeBase).filter(
            (KnowledgeBase.user_id == user_id) |
            (KnowledgeBase.is_public == True) |
            (KnowledgeBase.id.in_(
                self.session.query(KBUserPermission.kb_id).filter(
                    KBUserPermission.user_id == user_id,
                    KBUserPermission.permission.in_(["read", "write", "admin"]),
                )
            ))
        ).order_by(KnowledgeBase.updated_at.desc()).all()

    def list_owned(self, user_id: int) -> list[KnowledgeBase]:
        return self.session.query(KnowledgeBase).filter(
            KnowledgeBase.user_id == user_id
        ).order_by(KnowledgeBase.updated_at.desc()).all()

    def update(self, kb_id: int, user_id: int, **kwargs) -> KnowledgeBase | None:
        kb = self.get(kb_id)
        if not kb or kb.user_id != user_id:
            return None
        allowed = {"name", "description", "is_public"}
        for k, v in kwargs.items():
            if k in allowed:
                setattr(kb, k, v)
        self.session.flush()
        return kb

    def delete(self, kb_id: int, user_id: int) -> bool:
        kb = self.get(kb_id)
        if not kb or kb.user_id != user_id:
            return False
        self.session.delete(kb)
        self.session.flush()
        return True

    def add_permission(self, kb_id: int, user_id: int, permission: str = "read") -> KBUserPermission | None:
        exists = self.session.query(KBUserPermission).filter(
            KBUserPermission.kb_id == kb_id,
            KBUserPermission.user_id == user_id,
        ).first()
        if exists:
            exists.permission = permission
            self.session.flush()
            return exists
        perm = KBUserPermission(kb_id=kb_id, user_id=user_id, permission=permission)
        self.session.add(perm)
        self.session.flush()
        return perm

    def remove_permission(self, perm_id: int) -> bool:
        perm = self.session.query(KBUserPermission).filter(KBUserPermission.id == perm_id).first()
        if not perm:
            return False
        self.session.delete(perm)
        self.session.flush()
        return True

    def list_permissions(self, kb_id: int) -> list[KBUserPermission]:
        return self.session.query(KBUserPermission).filter(
            KBUserPermission.kb_id == kb_id
        ).all()

    def check_permission(self, kb_id: int, user_id: int, required: str = "read") -> bool:
        kb = self.get(kb_id)
        if not kb:
            return False
        if kb.user_id == user_id:
            return True
        if kb.is_public and required == "read":
            return True
        perm = self.session.query(KBUserPermission).filter(
            KBUserPermission.kb_id == kb_id,
            KBUserPermission.user_id == user_id,
        ).first()
        if not perm:
            return False
        levels = {"read": 0, "write": 1, "admin": 2}
        return levels.get(perm.permission, 0) >= levels.get(required, 0)
