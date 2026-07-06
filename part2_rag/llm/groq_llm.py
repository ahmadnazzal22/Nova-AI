import os
from typing import Generator
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk


class GroqLLM(LLM):
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.3
    max_tokens: int = 1024

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from groq import Groq
        self._client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    @property
    def _llm_type(self) -> str:
        return "groq"

    @staticmethod
    def _split_prompt(prompt: str) -> tuple[str, str]:
        for marker in ("QUESTION:", "Question:"):
            idx = prompt.rfind(marker)
            if idx != -1:
                return prompt[:idx].strip(), prompt[idx:].strip()
        return "", prompt

    def _call(self, prompt: str, **kwargs) -> str:
        system, user = self._split_prompt(prompt)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return resp.choices[0].message.content or ""

    def _stream(self, prompt: str, **kwargs) -> Generator[GenerationChunk, None, None]:
        system, user = self._split_prompt(prompt)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else ""
            if delta:
                yield GenerationChunk(text=delta)

    @property
    def _identifying_params(self) -> dict:
        return {"model": self.model}
