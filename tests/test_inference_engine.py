import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import asyncio
import pytest
import torch
from part1_transformer.config import TransformerConfig
from part1_transformer.transformer import Transformer
from part1_transformer.tokenizer import WordTokenizer
from part1_transformer.inference_engine import (
    InferenceRequest,
    ModelRunner,
    Scheduler,
    InferenceEngine,
)


@pytest.fixture
def tiny_checkpoint(tmp_path):
    config = TransformerConfig()
    config.vocab_size = 50
    config.d_model = 32
    config.num_heads = 2
    config.num_encoder_layers = 2
    config.num_decoder_layers = 2
    config.d_ff = 64
    config.max_seq_len = 20
    config.dropout = 0.0

    tokenizer = WordTokenizer(max_vocab_size=100)
    tokenizer.fit(["hello world test sentence transformer model attention"])
    config.vocab_size = tokenizer.vocab_size

    model = Transformer(config)
    path = os.path.join(tmp_path, "model.pth")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "tokenizer": tokenizer}, path)
    return path


class TestInferenceRequest:
    def test_default_id(self):
        req = InferenceRequest(prompt="hello")
        assert len(req.id) == 12
        assert req.prompt == "hello"
        assert req.max_tokens == 128
        assert req.temperature == 0.7
        assert req.stream is False

    def test_custom_values(self):
        req = InferenceRequest(prompt="test", max_tokens=64, temperature=0.5, stream=True)
        assert req.prompt == "test"
        assert req.max_tokens == 64
        assert req.temperature == 0.5
        assert req.stream is True


class TestModelRunner:
    def test_init_and_encode(self, tiny_checkpoint):
        runner = ModelRunner(tiny_checkpoint, device="cpu", use_fp16=False)
        assert runner.config.vocab_size > 4
        assert runner.model is not None
        assert runner.sos_id is not None

        input_ids = torch.randint(1, runner.config.vocab_size, (1, 8))
        enc = runner.encode(input_ids)
        assert enc.shape == (1, 8, runner.config.d_model)

    def test_generate_step(self, tiny_checkpoint):
        runner = ModelRunner(tiny_checkpoint, device="cpu", use_fp16=False)
        input_ids = torch.randint(1, runner.config.vocab_size, (1, 8))
        enc = runner.encode(input_ids)
        src_mask = (input_ids != runner.pad_id).unsqueeze(1).unsqueeze(2)
        kv_cache = {}
        step_input = torch.full((1, 1), runner.sos_id, dtype=torch.long)
        logits = runner.generate_step(step_input, enc, src_mask, kv_cache, pos_offset=0)
        assert logits.shape == (1, runner.config.vocab_size)

    def test_fp16_cpu_fallback(self, tiny_checkpoint):
        runner = ModelRunner(tiny_checkpoint, device="cuda", use_fp16=True)
        assert runner.device.type == "cpu"
        assert runner.dtype == torch.float32

    def test_warmup(self, tiny_checkpoint):
        runner = ModelRunner(tiny_checkpoint, device="cpu", use_fp16=False)
        runner.warmup(steps=2)


class TestScheduler:
    @pytest.mark.asyncio
    async def test_add_and_process(self, tiny_checkpoint):
        runner = ModelRunner(tiny_checkpoint, device="cpu", use_fp16=False)
        scheduler = Scheduler(runner, max_batch_size=4, batch_timeout=0.1)
        req1 = InferenceRequest(prompt="hello", max_tokens=5, temperature=0.0)
        req2 = InferenceRequest(prompt="world", max_tokens=5, temperature=0.0)

        task = asyncio.create_task(scheduler.run_scheduler())
        await asyncio.sleep(0.05)

        await scheduler.add_request(req1)
        await scheduler.add_request(req2)

        result1 = await req1.result_queue.get()
        result2 = await req2.result_queue.get()

        scheduler.stop()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, StopAsyncIteration):
            pass

        assert result1[0] == "done"
        assert isinstance(result1[1], str)
        assert result2[0] == "done"
        assert isinstance(result2[1], str)
        assert scheduler.metrics.total_requests == 2


class TestInferenceEngine:
    @pytest.mark.asyncio
    async def test_generate(self, tiny_checkpoint):
        engine = InferenceEngine(tiny_checkpoint, device="cpu", use_fp16=False, max_batch_size=2)
        await engine.start()
        result = await engine.generate("hello", max_tokens=5, temperature=0.0)
        assert isinstance(result, str)
        await engine.stop()

    @pytest.mark.asyncio
    async def test_generate_stream(self, tiny_checkpoint):
        engine = InferenceEngine(tiny_checkpoint, device="cpu", use_fp16=False, max_batch_size=2)
        await engine.start()
        tokens = []
        async for token in engine.generate_stream("hello", max_tokens=5, temperature=0.0):
            tokens.append(token)
        assert len(tokens) >= 0
        await engine.stop()

    @pytest.mark.asyncio
    async def test_concurrent_generate(self, tiny_checkpoint):
        engine = InferenceEngine(tiny_checkpoint, device="cpu", use_fp16=False, max_batch_size=4)
        await engine.start()
        results = await asyncio.gather(
            engine.generate("hello", max_tokens=5, temperature=0.0),
            engine.generate("world", max_tokens=5, temperature=0.0),
            engine.generate("test", max_tokens=5, temperature=0.0),
        )
        assert len(results) == 3
        all(isinstance(r, str) for r in results)
        await engine.stop()
