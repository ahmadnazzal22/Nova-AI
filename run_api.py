"""Run the API server with database initialization."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import uvicorn
from part2_rag.database import init_db

init_db()
print("Database initialized at", os.getenv("DB_PATH", "./rag.db"))

from part2_rag.api import app
uvicorn.run(app, host="0.0.0.0", port=8002)
