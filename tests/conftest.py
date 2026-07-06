"""Shared test configuration: temp DB and secrets."""
import os, sys, tempfile

# Must be set before any other imports
_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.environ["DB_PATH"] = _db_path
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"
os.environ["JWT_SECRET"] = "test-secret-for-ci"
os.environ["JWT_ACCESS_EXPIRE_SECONDS"] = "3600"
os.environ["JWT_REFRESH_EXPIRE_SECONDS"] = "2592000"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pytest
from part2_rag.database import init_db, engine
from part2_rag.models import Base


@pytest.fixture(autouse=True)
def clean_db():
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)
