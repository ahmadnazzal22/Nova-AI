#!/usr/bin/env python3
"""Run the Celery worker for background tasks."""
from part2_rag.worker.celery_app import celery_app

if __name__ == "__main__":
    celery_app.start(argv=["worker", "--loglevel=info", "--concurrency=4"])
