"""Tests for the database layer — repositories."""
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from part2_rag.models import Base, User, UserSettings, Conversation, Message, Feedback
from part2_rag.database import UserRepository, ConversationRepository, MessageRepository, FeedbackRepository, StatsRepository
from part2_rag.auth import hash_password


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def user_repo(db_session):
    return UserRepository(db_session)


@pytest.fixture
def conv_repo(db_session):
    return ConversationRepository(db_session)


@pytest.fixture
def msg_repo(db_session):
    return MessageRepository(db_session)


@pytest.fixture
def fb_repo(db_session):
    return FeedbackRepository(db_session)


@pytest.fixture
def stats_repo(db_session):
    return StatsRepository(db_session)


@pytest.fixture
def test_user(user_repo):
    return user_repo.create("alice", "alice@test.com", hash_password("secret"))


class TestUserRepository:
    def test_create_user(self, user_repo):
        user = user_repo.create("bob", "bob@test.com", hash_password("pass"))
        assert user is not None
        assert user.username == "bob"
        assert user.email == "bob@test.com"
        assert user.role == "user"
        assert user.is_active is True

    def test_create_user_duplicate(self, user_repo, test_user):
        user = user_repo.create("alice", "alice2@test.com", hash_password("pass"))
        assert user is None

    def test_get_by_id(self, user_repo, test_user):
        user = user_repo.get_by_id(test_user.id)
        assert user is not None
        assert user.username == "alice"

    def test_get_by_username(self, user_repo, test_user):
        user = user_repo.get_by_username("alice")
        assert user is not None

    def test_get_by_email(self, user_repo, test_user):
        user = user_repo.get_by_email("alice@test.com")
        assert user is not None

    def test_list_users(self, user_repo, test_user):
        users, total = user_repo.list()
        assert total >= 1
        assert len(users) >= 1

    def test_update_login(self, user_repo, test_user):
        user_repo.update_login(test_user.id)
        user = user_repo.get_by_id(test_user.id)
        assert user.last_login is not None

    def test_update_profile(self, user_repo, test_user):
        user_repo.update_profile(test_user.id, email="alice_new@test.com")
        user = user_repo.get_by_id(test_user.id)
        assert user.email == "alice_new@test.com"

    def test_deactivate(self, user_repo, test_user):
        result = user_repo.deactivate(test_user.id)
        assert result is True
        user = user_repo.get_by_id(test_user.id)
        assert user.is_active is False

    def test_count_active(self, user_repo, test_user):
        assert user_repo.count_active() == 1
        user_repo.deactivate(test_user.id)
        assert user_repo.count_active() == 0

    def test_get_settings(self, user_repo, test_user):
        settings = user_repo.get_settings(test_user.id)
        assert settings is not None
        assert settings.theme == "light"

    def test_update_settings(self, user_repo, test_user):
        settings = user_repo.update_settings(test_user.id, theme="dark")
        assert settings is not None
        assert settings.theme == "dark"

    def test_delete_user(self, user_repo, test_user):
        result = user_repo.delete(test_user.id)
        assert result is True
        assert user_repo.get_by_id(test_user.id) is None


class TestConversationRepository:
    def test_create_conv(self, conv_repo, test_user):
        conv = conv_repo.create(test_user.id, title="Test Chat")
        assert conv is not None
        assert conv.title == "Test Chat"
        assert conv.user_id == test_user.id

    def test_list_by_user(self, conv_repo, test_user):
        conv_repo.create(test_user.id)
        conv_repo.create(test_user.id, title="Chat 2")
        convs = conv_repo.list_by_user(test_user.id)
        assert len(convs) == 2

    def test_rename(self, conv_repo, test_user):
        conv = conv_repo.create(test_user.id, title="Old")
        conv_repo.rename(conv.id, "Renamed")
        assert conv_repo.get_by_id(conv.id).title == "Renamed"

    def test_delete(self, conv_repo, test_user):
        conv = conv_repo.create(test_user.id)
        assert conv_repo.delete(conv.id) is True
        assert conv_repo.get_by_id(conv.id) is None

    def test_count_by_user(self, conv_repo, test_user):
        conv_repo.create(test_user.id)
        conv_repo.create(test_user.id)
        assert conv_repo.count_by_user(test_user.id) == 2


class TestMessageRepository:
    def test_add_message(self, msg_repo, conv_repo, test_user):
        conv = conv_repo.create(test_user.id)
        msg = msg_repo.add(conv.id, "user", "Hello")
        assert msg is not None
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_list_by_conversation(self, msg_repo, conv_repo, test_user):
        conv = conv_repo.create(test_user.id)
        msg_repo.add(conv.id, "user", "Q1")
        msg_repo.add(conv.id, "assistant", "A1")
        msgs = msg_repo.list_by_conversation(conv.id)
        assert len(msgs) == 2

    def test_add_message_with_kwargs(self, msg_repo, conv_repo, test_user):
        conv = conv_repo.create(test_user.id)
        msg = msg_repo.add(conv.id, "assistant", "Answer", live=True, cached=False,
                          sources=[{"title": "Test"}], response_time=1.5, token_count=100)
        assert msg.live is True
        assert msg.response_time == 1.5
        assert msg.token_count == 100
        assert msg.sources == [{"title": "Test"}]

    def test_get_by_id(self, msg_repo, conv_repo, test_user):
        conv = conv_repo.create(test_user.id)
        msg = msg_repo.add(conv.id, "user", "Hi")
        assert msg_repo.get_by_id(msg.id) is not None
        assert msg_repo.get_by_id(99999) is None


class TestFeedbackRepository:
    def test_add_feedback(self, fb_repo, msg_repo, conv_repo, test_user):
        conv = conv_repo.create(test_user.id)
        msg = msg_repo.add(conv.id, "assistant", "Answer")
        fb = fb_repo.add(msg.id, test_user.id, 5, "Great!")
        assert fb is not None
        assert fb.rating == 5

    def test_get_by_message(self, fb_repo, msg_repo, conv_repo, test_user):
        conv = conv_repo.create(test_user.id)
        msg = msg_repo.add(conv.id, "assistant", "A")
        fb_repo.add(msg.id, test_user.id, 4)
        fb = fb_repo.get_by_message(msg.id)
        assert fb is not None
        assert fb.rating == 4

    def test_get_stats(self, fb_repo, msg_repo, conv_repo, test_user, db_session):
        conv = conv_repo.create(test_user.id)
        for i in range(3):
            msg = msg_repo.add(conv.id, "assistant", f"A{i}")
            fb_repo.add(msg.id, test_user.id, i + 3)
        stats = fb_repo.get_stats()
        assert stats["total_feedback"] == 3
        assert 3.0 <= stats["avg_rating"] <= 5.0


class TestStatsRepository:
    def test_get_system_stats(self, stats_repo, user_repo, conv_repo, msg_repo, test_user):
        u = user_repo.create("bob", "bob@test.com", hash_password("pass"))
        conv = conv_repo.create(u.id)
        msg_repo.add(conv.id, "user", "Hello")
        msg_repo.add(conv.id, "assistant", "Hi there", live=True, response_time=0.5)

        stats = stats_repo.get_system_stats()
        assert stats["total_users"] >= 2
        assert stats["total_conversations"] >= 1
        assert stats["total_messages"] >= 2
        assert stats["live_queries"] >= 1
