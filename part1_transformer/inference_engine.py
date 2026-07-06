import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable
import torch
from .config import TransformerConfig
from .tokenizer import WordTokenizer
from .transformer import Transformer
from .logger import get_logger

logger = get_logger(__name__)

_ALLOWED_SAFE_GLOBALS = [TransformerConfig, WordTokenizer]


@dataclass
class InferenceRequest:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    prompt: str = ""
    max_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False
    result_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


@dataclass
class SchedulerMetrics:
    total_requests: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    batch_count: int = 0
    avg_batch_size: float = 0.0

    @property
    def tokens_per_sec(self) -> float:
        return self.total_tokens / max(self.total_time, 1e-6)


class ModelRunner:
    def __init__(self, checkpoint_path: str, device: str = "cuda", use_fp16: bool = True):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32

        if not checkpoint_path:
            raise ValueError("checkpoint_path is required")

        logger.info("Loading checkpoint: %s | device=%s | fp16=%s", checkpoint_path, self.device, self.use_fp16)

        with torch.serialization.safe_globals(_ALLOWED_SAFE_GLOBALS):
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        self.config: TransformerConfig = checkpoint["config"]
        self.tokenizer: WordTokenizer = checkpoint["tokenizer"]

        self.model = Transformer(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        if self.use_fp16:
            self.model = self.model.half()

        self.sos_id = self.tokenizer.word2idx.get("<SOS>", 1)
        self.eos_id = self.tokenizer.word2idx.get("<EOS>", 2)
        self.pad_id = self.tokenizer.word2idx.get("<PAD>", 0)

        logger.info("ModelRunner ready | vocab=%d | d_model=%d | device=%s | dtype=%s",
                    self.config.vocab_size, self.config.d_model, self.device, self.dtype)

    def warmup(self, steps: int = 3):
        dummy = torch.randint(1, min(self.config.vocab_size, 50), (1, 8), device=self.device)
        for _ in range(steps):
            _ = self.model.encode(dummy)
        logger.info("Warmup complete (%d steps)", steps)

    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id)
        src_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        return self.model.encoder(input_ids, src_mask)

    @torch.no_grad()
    def generate_step(self, input_ids: torch.Tensor, encoder_output: torch.Tensor,
                      src_mask: torch.Tensor, kv_cache: dict, pos_offset: int) -> torch.Tensor:
        dec_output = self.model.decoder.forward_cached(
            input_ids[:, -1:], encoder_output, src_mask,
            kv_cache=kv_cache, pos_offset=pos_offset
        )
        return self.model.fc_out(dec_output[:, -1, :])


class _ActiveRequest:
    __slots__ = ("req", "encoder_output", "src_mask", "kv_cache", "generated", "pos", "finished")

    def __init__(self, req: InferenceRequest, runner: ModelRunner):
        self.req = req
        self.finished = False
        self.pos = 0
        self.generated: list[int] = [runner.sos_id]

        token_ids = runner.tokenizer.encode(req.prompt, runner.config.max_seq_len)
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=runner.device)
        attn_mask = (input_tensor != runner.pad_id)

        enc_out = runner.encode(input_tensor, attn_mask)
        src_mask_t = attn_mask.unsqueeze(1).unsqueeze(2)

        self.encoder_output = enc_out
        self.src_mask = src_mask_t
        self.kv_cache: dict = {}


class Scheduler:
    def __init__(self, runner: ModelRunner, max_batch_size: int = 8, batch_timeout: float = 0.05):
        self.runner = runner
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.active: dict[str, _ActiveRequest] = {}
        self.running = False
        self.metrics = SchedulerMetrics()

    async def add_request(self, req: InferenceRequest):
        await self.request_queue.put(req)

    async def run_scheduler(self):
        self.running = True
        logger.info("Scheduler started | max_batch=%d | timeout=%.3fs", self.max_batch_size, self.batch_timeout)

        while self.running or not self.request_queue.empty():
            new = await self._drain_queue()
            for req in new:
                self.active[req.id] = _ActiveRequest(req, self.runner)
                self.metrics.total_requests += 1

            if not self.active:
                await asyncio.sleep(0.001)
                continue

            batch = list(self.active.values())[:self.max_batch_size]
            t0 = time.perf_counter()
            await self._process_batch(batch)
            self.metrics.total_time += time.perf_counter() - t0
            self.metrics.batch_count += 1
            self.metrics.avg_batch_size += (len(batch) - self.metrics.avg_batch_size) / self.metrics.batch_count

            done = [rid for rid, ar in self.active.items() if ar.finished]
            for rid in done:
                del self.active[rid]

        logger.info("Scheduler stopped | req=%d | tok=%d | tok/s=%.1f",
                    self.metrics.total_requests, self.metrics.total_tokens, self.metrics.tokens_per_sec)

    def stop(self):
        self.running = False

    async def _drain_queue(self) -> list[InferenceRequest]:
        items: list[InferenceRequest] = []
        try:
            while len(items) < self.max_batch_size:
                req = await asyncio.wait_for(self.request_queue.get(), timeout=self.batch_timeout)
                items.append(req)
        except (asyncio.TimeoutError, StopAsyncIteration):
            pass
        return items

    async def _process_batch(self, batch: list[_ActiveRequest]):
        ready = [ar for ar in batch if not ar.finished]
        if not ready:
            return

        step_tokens = torch.tensor(
            [[ar.generated[-1]] for ar in ready],
            dtype=torch.long, device=self.runner.device
        )

        all_logits = []
        for i, ar in enumerate(ready):
            logits = self.runner.generate_step(
                step_tokens[i:i+1], ar.encoder_output, ar.src_mask, ar.kv_cache, ar.pos
            )
            all_logits.append(logits)

        logits = torch.cat(all_logits, dim=0)
        temps = torch.tensor([ar.req.temperature for ar in ready], device=self.runner.device)

        if (temps > 0).all():
            probs = torch.softmax(logits / temps.unsqueeze(1), dim=-1)
            next_tokens = torch.multinomial(probs, 1)
        else:
            next_tokens = logits.argmax(dim=-1, keepdim=True)

        for i, ar in enumerate(ready):
            tok = next_tokens[i, 0].item()
            ar.generated.append(tok)
            ar.pos += 1
            self.metrics.total_tokens += 1

            if ar.req.stream:
                text = self.runner.tokenizer.decode([tok])
                asyncio.ensure_future(ar.req.result_queue.put(("token", text)))

            is_eos = tok == self.runner.eos_id
            is_max_len = (len(ar.generated) - 1) >= ar.req.max_tokens
            if is_eos or is_max_len:
                ar.finished = True
                full_text = self.runner.tokenizer.decode(ar.generated[1:])
                asyncio.ensure_future(ar.req.result_queue.put(("done", full_text)))


class InferenceEngine:
    def __init__(self, checkpoint_path: str, device: str = "cuda",
                 use_fp16: bool = True, max_batch_size: int = 8):
        self.runner = ModelRunner(checkpoint_path, device, use_fp16)
        self.scheduler = Scheduler(self.runner, max_batch_size)
        self._scheduler_task: asyncio.Task | None = None

    async def start(self):
        self._scheduler_task = asyncio.create_task(self.scheduler.run_scheduler())
        logger.info("InferenceEngine started")

    async def stop(self):
        self.scheduler.stop()
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("InferenceEngine stopped")

    async def generate(self, prompt: str, max_tokens: int = 128,
                       temperature: float = 0.7) -> str:
        req = InferenceRequest(prompt=prompt, max_tokens=max_tokens,
                               temperature=temperature, stream=False)
        await self.scheduler.add_request(req)
        result = await req.result_queue.get()
        return result[1]

    async def generate_stream(self, prompt: str, max_tokens: int = 128,
                              temperature: float = 0.7) -> AsyncGenerator[str, None]:
        req = InferenceRequest(prompt=prompt, max_tokens=max_tokens,
                               temperature=temperature, stream=True)
        await self.scheduler.add_request(req)
        while True:
            msg = await req.result_queue.get()
            if msg[0] == "done":
                break
            yield msg[1]

    def warmup(self):
        self.runner.warmup()
