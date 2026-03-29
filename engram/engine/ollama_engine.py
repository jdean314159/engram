"""
Ollama Engine Implementation

Connects to local Ollama server (OpenAI-compatible API).
Supports layer offloading to CPU via OLLAMA_NUM_GPU / num_gpu option,
making large models (e.g. Qwen 32B) viable when VRAM is constrained.

Logprobs require Ollama >= 0.12.11.

Author: Jeffrey Dean
"""

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:  # pragma: no cover
    OpenAI = AsyncOpenAI = None
from typing import Optional, Union
from pydantic import BaseModel
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:  # pragma: no cover
    def retry(*args, **kwargs):
        def _decorator(fn):
            return fn
        return _decorator
    def stop_after_attempt(*args, **kwargs):
        return None
    def wait_exponential(*args, **kwargs):
        return None
    def retry_if_exception_type(*args, **kwargs):
        return None
import logging
from urllib.parse import urlparse
from urllib.request import urlopen
import json

from .base import LLMEngine, CompressionStrategy, LogprobResult, _count_tokens
from .openai_compat_mixin import OpenAICompatMixin

logger = logging.getLogger(__name__)


class OllamaEngine(OpenAICompatMixin, LLMEngine):
    """
    Local Ollama server connection.

    Key advantage over VLLMEngine: CPU layer offloading via num_gpu lets
    you run models larger than VRAM (e.g. Qwen 32B on 24GB with partial
    offload). Throughput on CPU layers is slower (~5-10 tok/s vs 40+),
    so prefer VLLMEngine when the model fits in VRAM.

    Logprobs + async streaming come from OpenAICompatMixin.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",          # Ollama ignores the key but OpenAI client requires one
        system_prompt: Optional[str] = None,
        compression_strategy: CompressionStrategy = CompressionStrategy.COMPRESS,
        max_retries: int = 1,
        timeout: int = 300,               # Longer default: CPU offload is slower
        max_context: int = 8192,
        num_gpu: Optional[int] = None,    # None = use all GPUs; 0 = CPU only; N = N layers on GPU
        thinking: bool = False,           # Enable/disable thinking mode (Qwen3/3.5 etc.)
    ):
        """
        Args:
            num_gpu: Controls GPU layer offloading.
                     None  → Ollama decides (default, all-GPU if it fits)
                     0     → CPU-only (slow but no VRAM used)
                     N     → N layers on GPU, rest on CPU
                     Passed as options.num_gpu in each request.
        """
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            compression_strategy=compression_strategy,
            max_retries=max_retries,
            timeout=timeout,
        )

        self.base_url = base_url
        self._max_context = max_context
        self._num_gpu = num_gpu
        self._thinking = thinking

        # Capability flags (best-effort detection)
        self.supports_logprobs: bool = True

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self.async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

        # Best-effort capability detection (avoid breaking older Ollama installs)
        try:
            self.supports_logprobs = self._detect_logprobs_support()
            if not self.supports_logprobs:
                logger.warning(
                    "Ollama server appears to be too old to support logprobs; "
                    "surprise filtering will be disabled or run in conservative mode."
                )
        except Exception as e:
            # If detection fails, keep optimistic default and let runtime handle it.
            logger.debug("Could not detect Ollama capabilities (%s); proceeding.", e)

    def _detect_logprobs_support(self) -> bool:
        """Return True if the connected Ollama supports logprobs.

        Ollama exposes a version endpoint at /api/version on the server root.
        Logprobs support is documented as >= 0.12.11.
        """
        parsed = urlparse(self.base_url)
        # base_url is usually http://host:11434/v1 → root is http://host:11434
        root = f"{parsed.scheme}://{parsed.netloc}"
        url = f"{root}/api/version"

        with urlopen(url, timeout=3) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        version = str(payload.get("version") or "").strip()
        if not version:
            return True

        def parse_semver(v: str) -> tuple[int, int, int]:
            # Accept forms like "0.12.11", "v0.12.11", "0.12.11-rc1".
            parts = v.lstrip("v").split(".")
            out: list[int] = []
            for p in parts[:3]:
                num = "".join(ch for ch in p if ch.isdigit())
                out.append(int(num) if num else 0)
            while len(out) < 3:
                out.append(0)
            return out[0], out[1], out[2]

        try:
            major, minor, patch = parse_semver(version)
        except Exception:
            return True

        return (major, minor, patch) >= (0, 12, 11)
    def ensure_model_pulled(self) -> bool:
        """Ensure the model is present on the Ollama host.

        Uses Ollama's native HTTP API (not the CLI) so it works on Windows/Linux
        and inside containers.

        Returns:
            True if the model is present (or was pulled successfully), False otherwise.
        """
        parsed = urlparse(self.base_url)
        root = f"{parsed.scheme}://{parsed.netloc}"

        # 1) Check presence
        show_url = f"{root}/api/show"
        try:
            from urllib.request import Request
            req = Request(
                show_url,
                data=json.dumps({"name": self.model_name}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=10) as resp:
                _ = json.loads(resp.read().decode("utf-8"))
            return True
        except Exception:
            pass

        # 2) Pull (best-effort)
        pull_url = f"{root}/api/pull"
        try:
            from urllib.request import Request
            req = Request(
                pull_url,
                data=json.dumps({"name": self.model_name, "stream": False}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=600) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if isinstance(payload, dict) and payload.get("error"):
                logger.warning("Ollama pull failed for %s: %s", self.model_name, payload.get("error"))
                return False
            return True
        except Exception as e:
            logger.warning("Could not pull model %s from Ollama (%s)", self.model_name, e)
            return False

    def _num_gpu_kwargs(self) -> dict:
        """Return extra_body options for num_gpu and thinking mode."""
        extra_body: dict = {}
        options: dict = {}
        if self._num_gpu is not None:
            options["num_gpu"] = self._num_gpu
        if options:
            extra_body["options"] = options
        if not self._thinking:
            extra_body["think"] = False
        return {"extra_body": extra_body} if extra_body else {}

    def _native_url(self) -> str:
            """Derive native Ollama API root from the OpenAI-compat base_url."""
            from urllib.parse import urlparse
            parsed = urlparse(self.base_url)
            # base_url is http://host:port/v1 — strip /v1 to get root
            return f"{parsed.scheme}://{parsed.netloc}"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[type[BaseModel]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs,
    ) -> Union[str, BaseModel]:
        """Generate with auto-compression and structured output."""

        sys_prompt = system_prompt or self.system_prompt
        prompt_tokens = self.count_tokens(prompt)
        context_limit = self.adaptive_context_limit or self.max_context_length

        if prompt_tokens > context_limit - max_tokens - 100:
            logger.warning("Prompt (%d tokens) exceeds limit, compressing...", prompt_tokens)
            prompt = self.compress_prompt(prompt, target_tokens=context_limit - max_tokens - 100)

        if response_format:
            from .utils.structured_output import StructuredOutputHandler
            schema_prompt = StructuredOutputHandler.create_schema_prompt(response_format, include_examples=True)
            prompt = f"{prompt}\n\n{schema_prompt}"

        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            raw_response = self._generate_with_retry(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if response_format:
                from .utils.structured_output import StructuredOutputHandler
                return StructuredOutputHandler.parse(raw_response, response_format)

            return raw_response

        except Exception as e:
            error_str = str(e).lower()
            if "out of memory" in error_str or ("cuda" in error_str and "memory" in error_str) or "oom" in error_str:
                if kwargs.pop("_oom_retried", False):
                    raise RuntimeError(f"OOM persists after recovery: {e}")
                logger.error("OOM detected! Attempting recovery...")
                if self.recover_from_oom():
                    logger.info("OOM recovery successful, retrying...")
                    return self.generate(
                        prompt, sys_prompt, response_format, temperature, max_tokens,
                        _oom_retried=True, **kwargs,
                    )
                raise RuntimeError(f"OOM recovery failed: {e}")
            raise

    def _generate_with_retry(self, messages: list, temperature: float, max_tokens: int) -> str:
        """Generate via native /api/chat endpoint to support think:false."""
        import urllib.request
        import json as _json

        url = f"{self._native_url()}/api/chat"

        payload: dict = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": self._thinking,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if self._num_gpu is not None:
            payload["options"]["num_gpu"] = self._num_gpu

        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = _json.loads(resp.read().decode("utf-8"))

        content = result.get("message", {}).get("content", "")
        if not content:
            # Fallback: thinking field (should be empty when think=False)
            content = result.get("message", {}).get("thinking", "") or ""
        return content

    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (cl100k_base) with len//4 fallback."""
        return _count_tokens(text)

    @property
    def max_context_length(self) -> int:
        return self._max_context
