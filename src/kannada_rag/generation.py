"""LLM providers behind one interface, under matched decoding settings.

Three providers are supported:

``ollama``
    Local inference. The deployment target of this system and the condition
    under which all headline numbers are reported.
``groq``
    A hosted endpoint, used only for the optional hosted-model comparison.
``echo``
    A deterministic stub that returns the retrieved context. It exists so that
    the retrieval and evaluation machinery can be tested end to end in CI
    without a GPU or a network call.

**Matched decoding.** Every provider receives the same temperature, top-p,
maximum token count and seed from :class:`~kannada_rag.config.GenerationConfig`.
A comparison between a local model and a hosted model is only interpretable if
the decoding parameters are held fixed; otherwise the hosted model's longer,
more elaborate outputs reflect its default sampling settings rather than any
property of the system under test. Where a provider cannot honour a setting
(several hosted APIs ignore ``seed``), the returned
:class:`GenerationOutput` records that fact in ``honoured_settings`` so the
limitation is reported rather than assumed away.

**Latency.** ``latency_s`` measures only the generation call. End-to-end query
latency, including embedding and retrieval, is measured by the pipeline. Local
GPU latency and hosted API latency are recorded in separate fields and are
never averaged together, because the hosted figure is dominated by network
round-trip time and queueing that have nothing to do with the model.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .config import GenerationConfig

logger = logging.getLogger(__name__)

__all__ = [
    "GenerationOutput",
    "LLMProvider",
    "OllamaProvider",
    "GroqProvider",
    "EchoProvider",
    "get_provider",
]


@dataclass
class GenerationOutput:
    """One generation, with the metadata needed to audit it later."""

    text: str
    model: str
    provider: str
    latency_s: float
    is_local: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    honoured_settings: dict[str, bool] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "latency_s": round(self.latency_s, 4),
            "is_local": self.is_local,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "honoured_settings": self.honoured_settings,
            "error": self.error,
        }


class LLMProvider(ABC):
    is_local: bool = False
    name: str = "abstract"

    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg

    @abstractmethod
    def _generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        ...

    def generate(self, prompt: str) -> GenerationOutput:
        start = time.perf_counter()
        error = None
        meta: dict[str, Any] = {}
        try:
            text, meta = self._generate(prompt)
        except Exception as exc:  # noqa: BLE001 - surfaced in the log, not swallowed
            logger.error("%s generation failed: %s", self.name, exc)
            text, error = "", str(exc)
        latency = time.perf_counter() - start

        return GenerationOutput(
            text=text.strip(),
            model=self.cfg.model,
            provider=self.name,
            latency_s=latency,
            is_local=self.is_local,
            prompt_tokens=meta.get("prompt_tokens"),
            completion_tokens=meta.get("completion_tokens"),
            honoured_settings=meta.get("honoured_settings", {}),
            error=error,
        )


class OllamaProvider(LLMProvider):
    """Local inference through the Ollama HTTP API."""

    is_local = True
    name = "ollama"

    def _generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        import requests

        options: dict[str, Any] = {
            "temperature": self.cfg.temperature,
            "num_predict": self.cfg.max_tokens,
            "top_p": self.cfg.top_p,
        }
        if self.cfg.seed is not None:
            options["seed"] = self.cfg.seed

        response = requests.post(
            f"{self.cfg.ollama_host}/api/generate",
            json={
                "model": self.cfg.model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            },
            timeout=self.cfg.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()

        return payload.get("response", ""), {
            "prompt_tokens": payload.get("prompt_eval_count"),
            "completion_tokens": payload.get("eval_count"),
            # Ollama honours all three; recorded explicitly so the claim is
            # auditable rather than assumed.
            "honoured_settings": {"temperature": True, "top_p": True, "seed": True},
        }


class GroqProvider(LLMProvider):
    """Hosted inference, used only for the optional hosted-model comparison."""

    is_local = False
    name = "groq"

    def _generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and fill it in. "
                "Never commit .env."
            )

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=self.cfg.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            top_p=self.cfg.top_p,
            seed=self.cfg.seed,
        )
        usage = getattr(completion, "usage", None)

        return completion.choices[0].message.content or "", {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            # Hosted endpoints treat `seed` as best-effort only.
            "honoured_settings": {"temperature": True, "top_p": True, "seed": False},
        }


class EchoProvider(LLMProvider):
    """Deterministic stub for tests. Returns the first context passage."""

    is_local = True
    name = "echo"

    def _generate(self, prompt: str) -> tuple[str, dict[str, Any]]:
        marker = "ಆಧಾರ ಪಠ್ಯ:"
        if marker in prompt:
            body = prompt.split(marker, 1)[1]
            first = body.strip().split("\n\n", 1)[0]
            return first, {"honoured_settings": {"temperature": True, "seed": True}}
        return "", {"honoured_settings": {"temperature": True, "seed": True}}


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "ollama": OllamaProvider,
    "groq": GroqProvider,
    "echo": EchoProvider,
}


def get_provider(cfg: GenerationConfig) -> LLMProvider:
    try:
        return _PROVIDERS[cfg.provider](cfg)
    except KeyError:
        raise ValueError(
            f"Unknown provider {cfg.provider!r}. Available: {sorted(_PROVIDERS)}"
        ) from None
