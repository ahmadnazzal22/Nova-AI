import os
import time
from typing import AsyncGenerator
from part1_transformer.inference_engine import InferenceEngine
from .logger import get_logger

logger = get_logger(__name__)


class InferenceServer:
    def __init__(self, checkpoint_path: str | None = None,
                 device: str = "cuda", use_fp16: bool = True,
                 max_batch_size: int = 8):
        path = checkpoint_path or os.getenv("TRANSFORMER_CHECKPOINT", "transformer_best.pth")
        if not os.path.exists(path):
            path = "transformer_checkpoint.pth"
        self.engine = InferenceEngine(path, device, use_fp16, max_batch_size)
        self.start_time: float | None = None

    async def start(self):
        self.start_time = time.time()
        self.engine.warmup()
        await self.engine.start()
        logger.info("InferenceServer ready")

    async def stop(self):
        await self.engine.stop()

    async def generate(self, prompt: str, max_tokens: int = 128,
                       temperature: float = 0.7) -> str:
        return await self.engine.generate(prompt, max_tokens, temperature)

    async def generate_stream(self, prompt: str, max_tokens: int = 128,
                              temperature: float = 0.7) -> AsyncGenerator[str, None]:
        async for token in self.engine.generate_stream(prompt, max_tokens, temperature):
            yield token

    def health(self) -> dict:
        m = self.engine.scheduler.metrics
        uptime = time.time() - (self.start_time or time.time())
        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info = {
                    "device": torch.cuda.get_device_name(0),
                    "memory_allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
                    "memory_reserved_gb": round(torch.cuda.memory_reserved(0) / 1e9, 2),
                    "utilization": None,
                }
            else:
                gpu_info = {"device": "cpu"}
        except Exception:
            gpu_info = {"device": "unknown"}

        return {
            "status": "ok",
            "uptime_seconds": round(uptime, 1),
            "gpu": gpu_info,
            "scheduler": {
                "total_requests": m.total_requests,
                "total_tokens": m.total_tokens,
                "tokens_per_sec": round(m.tokens_per_sec, 1),
                "avg_batch_size": round(m.avg_batch_size, 2),
                "active_requests": len(self.engine.scheduler.active),
            },
        }
