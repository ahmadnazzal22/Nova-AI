import os
import io
import uuid
from typing import BinaryIO

from ..logger import get_logger

logger = get_logger(__name__)


class ObjectStore:
    def __init__(self, endpoint: str = "http://localhost:9000", access_key: str = "minioadmin", secret_key: str = "minioadmin", bucket: str = "rag-documents", region: str = "us-east-1"):
        self._client = None
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._region = region
        self._local_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "local_storage")
        self._enabled = False

    def _ensure_client(self):
        if self._client is None:
            try:
                from minio import Minio
                self._client = Minio(
                    self._endpoint.replace("http://", "").replace("https://", ""),
                    access_key=self._access_key,
                    secret_key=self._secret_key,
                    secure=self._endpoint.startswith("https"),
                )
                if not self._client.bucket_exists(self._bucket):
                    self._client.make_bucket(self._bucket)
                self._enabled = True
                logger.info("MinIO connected: %s/%s", self._endpoint, self._bucket)
            except Exception as e:
                logger.warning("MinIO unavailable, using local filesystem: %s", e)
                os.makedirs(self._local_dir, exist_ok=True)
                self._enabled = False

    def upload(self, data: bytes, filename: str, content_type: str = "application/octet-stream") -> str:
        self._ensure_client()
        key = f"{uuid.uuid4().hex[:8]}_{filename}"
        if self._enabled:
            try:
                self._client.put_object(
                    self._bucket, key, io.BytesIO(data), len(data),
                    content_type=content_type,
                )
                logger.info("Uploaded to MinIO: %s (%d bytes)", key, len(data))
                return key
            except Exception as e:
                logger.warning("MinIO upload failed: %s", e)
        local_path = os.path.join(self._local_dir, key)
        with open(local_path, "wb") as f:
            f.write(data)
        logger.info("Saved locally: %s (%d bytes)", local_path, len(data))
        return key

    def download(self, key: str) -> bytes | None:
        self._ensure_client()
        if self._enabled:
            try:
                response = self._client.get_object(self._bucket, key)
                data = response.read()
                response.close()
                return data
            except Exception as e:
                logger.warning("MinIO download failed: %s", e)
        local_path = os.path.join(self._local_dir, key)
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                return f.read()
        return None

    def delete(self, key: str) -> bool:
        self._ensure_client()
        if self._enabled:
            try:
                self._client.remove_object(self._bucket, key)
                return True
            except Exception:
                pass
        local_path = os.path.join(self._local_dir, key)
        if os.path.exists(local_path):
            os.remove(local_path)
            return True
        return False

    def list_files(self, prefix: str = "") -> list[str]:
        self._ensure_client()
        if self._enabled:
            try:
                objs = self._client.list_objects(self._bucket, prefix=prefix)
                return [o.object_name for o in objs]
            except Exception:
                pass
        if os.path.exists(self._local_dir):
            return [f for f in os.listdir(self._local_dir) if f.startswith(prefix)]
        return []

    def get_url(self, key: str) -> str:
        self._ensure_client()
        if self._enabled:
            try:
                return self._client.presigned_get_object(self._bucket, key)
            except Exception:
                pass
        return f"local://{key}"


_object_store: ObjectStore | None = None


def get_object_store() -> ObjectStore:
    global _object_store
    if _object_store is None:
        from ..config.settings import settings
        _object_store = ObjectStore(
            endpoint=settings.s3_endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            bucket=settings.s3_bucket,
            region=settings.s3_region,
        )
    return _object_store
