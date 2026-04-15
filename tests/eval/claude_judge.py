"""
Engram Memory Eval — Claude Judge

Sends (retrieved context, expected fact, query) triples to Claude
and returns structured judgments.
"""
import asyncio
import aiohttp
import json
from dataclasses import dataclass
from typing import Optional

from fact_generator import Fact
from engram_probe import RetrievalResult

JUDGE_SYSTEM = """
You are a strict memory retrieval judge. Given a query, the expected fact,
and a list of retrieved context chunks, you evaluate whether the retrieval
succeeded. Output ONLY a JSON object. No commentary, no markdown fences.
"""

JUDGE_PROMPT = """
EXPECTED FACT:
{canonical}

EXPECTED SNIPPET (must appear conceptually in correct result):
{expected_snippet}

QUERY ISSUED:
{query}

QUERY TYPE: {query_type}
(If query_type is "decoy", the expected fact should NOT be retrieved — mark retrieved=false if it does appear.)

RETRIEVED CONTEXT CHUNKS:
{chunks}

Evaluate and return this JSON object:
{{
  "retrieved": true|false,
  "relevance": 0-5,
  "contaminated": true|false,
  "verbatim_match": true|false,
  "notes": "one sentence explanation"
}}

Definitions:
- retrieved: the expected fact (or its core claim) is present in the chunks
- relevance: 0=wrong topic, 5=exact match
- contaminated: a clearly wrong/contradicting fact was returned (bad retrieval)
- verbatim_match: the expected_snippet appears nearly verbatim in the chunks
- For decoy queries: retrieved should be FALSE if the target fact is absent (that's correct behavior)
"""


@dataclass
class JudgmentResult:
    fact_id: str
    trial: int
    query_type: str           # direct | paraphrase | decoy
    top_k: int
    retrieved: bool
    relevance: int            # 0-5
    contaminated: bool
    verbatim_match: bool
    notes: str
    judge_error: Optional[str] = None

    # Derived: for decoy queries, "correct" means NOT retrieved
    @property
    def correct(self) -> bool:
        if self.query_type == "decoy":
            return not self.retrieved
        return self.retrieved

    def as_dict(self) -> dict:
        return {
            "fact_id": self.fact_id,
            "trial": self.trial,
            "query_type": self.query_type,
            "top_k": self.top_k,
            "retrieved": self.retrieved,
            "relevance": self.relevance,
            "contaminated": self.contaminated,
            "verbatim_match": self.verbatim_match,
            "correct": self.correct,
            "notes": self.notes,
            "judge_error": self.judge_error,
        }


class ClaudeJudge:
    def __init__(self, model: str = "claude-sonnet-4-20250514", max_tokens: int = 512):
        self.model = model
        self.max_tokens = max_tokens
        self._session: Optional[aiohttp.ClientSession] = None
        # Serialise all judge calls: one at a time, 1.3s apart (46 RPM)
        self._rate_lock = asyncio.Lock()
        self._rate_interval = 1.3
        self._last_call_time = 0.0

    async def start(self):
        self._session = aiohttp.ClientSession()

    async def stop(self):
        if self._session:
            await self._session.close()

    async def judge(
        self,
        fact: Fact,
        retrieval: RetrievalResult,
        trial: int,
    ) -> JudgmentResult:
        chunks_text = "\n---\n".join(retrieval.retrieved_chunks) or "(no chunks returned)"
        prompt = JUDGE_PROMPT.format(
            canonical=fact.canonical,
            expected_snippet=fact.expected_snippet,
            query=retrieval.query,
            query_type=retrieval.query_type,
            chunks=chunks_text,
        )

        # Serialize all API calls through the lock; sleep inside ensures
        # the interval is enforced atomically — no two coroutines can race
        # past the sleep simultaneously.
        async with self._rate_lock:
            import time as _time
            now = _time.monotonic()
            wait = self._rate_interval - (now - self._last_call_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_call_time = _time.monotonic()
            raw = await self._call_claude(prompt)

        if raw is None:
            return JudgmentResult(
                fact_id=fact.id, trial=trial,
                query_type=retrieval.query_type, top_k=retrieval.top_k,
                retrieved=False, relevance=0, contaminated=False,
                verbatim_match=False, notes="",
                judge_error="Claude API call failed",
            )

        try:
            parsed = json.loads(raw.strip())
            return JudgmentResult(
                fact_id=fact.id,
                trial=trial,
                query_type=retrieval.query_type,
                top_k=retrieval.top_k,
                retrieved=bool(parsed.get("retrieved", False)),
                relevance=int(parsed.get("relevance", 0)),
                contaminated=bool(parsed.get("contaminated", False)),
                verbatim_match=bool(parsed.get("verbatim_match", False)),
                notes=str(parsed.get("notes", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return JudgmentResult(
                fact_id=fact.id, trial=trial,
                query_type=retrieval.query_type, top_k=retrieval.top_k,
                retrieved=False, relevance=0, contaminated=False,
                verbatim_match=False, notes="",
                judge_error=f"Parse error: {e} | raw: {raw[:200]}",
            )

    async def _call_claude(self, user_content: str) -> Optional[str]:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": JUDGE_SYSTEM,
            "messages": [{"role": "user", "content": user_content}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        try:
            async with self._session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                if "content" not in data:
                    print(f"[judge] API error: {data.get('error', data)}")
                    return None
                return data["content"][0]["text"]
        except Exception as e:
            print(f"[judge] Request failed: {e}")
            return None

    async def judge_batch(
        self,
        facts: list[Fact],
        retrievals: list[RetrievalResult],
        trial: int,
    ) -> list[JudgmentResult]:
        """Judge a batch of retrieval results concurrently."""
        tasks = [
            self.judge(fact, ret, trial)
            for fact, ret in zip(facts, retrievals)
        ]
        return await asyncio.gather(*tasks)


class OllamaJudge:
    """
    Drop-in replacement for ClaudeJudge using a local Ollama model.
    Same public interface: start(), stop(), judge(), judge_batch().
    """
    def __init__(
        self,
        model: str = "qwen3:32b",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 512,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self._session: Optional[aiohttp.ClientSession] = None
        # Serialize calls to avoid VRAM thrash from concurrent requests
        self._sem = asyncio.Semaphore(3)

    async def start(self):
        self._session = aiohttp.ClientSession()

    async def stop(self):
        if self._session:
            await self._session.close()

    async def judge(
        self,
        fact: Fact,
        retrieval: RetrievalResult,
        trial: int,
    ) -> JudgmentResult:
        chunks_text = "\n---\n".join(retrieval.retrieved_chunks) or "(no chunks returned)"
        prompt = JUDGE_PROMPT.format(
            canonical=fact.canonical,
            expected_snippet=fact.expected_snippet,
            query=retrieval.query,
            query_type=retrieval.query_type,
            chunks=chunks_text,
        )
        full_prompt = f"/no_think\n{JUDGE_SYSTEM.strip()}\n\n{prompt}"

        async with self._sem:
            raw = await self._call_ollama(full_prompt)

        if raw is None:
            return JudgmentResult(
                fact_id=fact.id, trial=trial,
                query_type=retrieval.query_type, top_k=retrieval.top_k,
                retrieved=False, relevance=0, contaminated=False,
                verbatim_match=False, notes="",
                judge_error="Ollama call failed",
            )

        # Strip <think>...</think> blocks (qwen3 thinking mode)
        if "<think>" in raw:
            raw = raw[raw.rfind("</think>") + 8:].strip()
        # Strip markdown fences
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("{"):
                    raw = part
                    break

        try:
            parsed = json.loads(raw.strip())
            return JudgmentResult(
                fact_id=fact.id,
                trial=trial,
                query_type=retrieval.query_type,
                top_k=retrieval.top_k,
                retrieved=bool(parsed.get("retrieved", False)),
                relevance=int(parsed.get("relevance", 0)),
                contaminated=bool(parsed.get("contaminated", False)),
                verbatim_match=bool(parsed.get("verbatim_match", False)),
                notes=str(parsed.get("notes", "")),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return JudgmentResult(
                fact_id=fact.id, trial=trial,
                query_type=retrieval.query_type, top_k=retrieval.top_k,
                retrieved=False, relevance=0, contaminated=False,
                verbatim_match=False, notes="",
                judge_error=f"Parse error: {e} | raw: {raw[:200]}",
            )

    async def _call_ollama(self, prompt: str) -> Optional[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,          # disable qwen3 thinking mode
            "options": {
                "temperature": 0.1,
                "num_predict": 256,  # response only; thinking disabled so this is enough
            },
        }
        try:
            async with self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=45),
            ) as resp:
                data = await resp.json()
                raw = data.get("response", "")
                if not raw:
                    print(f"[judge/ollama] Empty response: {data}")
                return raw or None
        except asyncio.TimeoutError:
            print("[judge/ollama] Timeout — restarting Ollama runner")
            import subprocess
            subprocess.run(["sudo", "systemctl", "restart", "ollama"], check=False)
            import asyncio as _asyncio
            await _asyncio.sleep(10)
            return None
        except Exception as e:
            print(f"[judge/ollama] Request failed: {e}")
            return None

    async def judge_batch(
        self,
        facts: list[Fact],
        retrievals: list[RetrievalResult],
        trial: int,
    ) -> list[JudgmentResult]:
        tasks = [
            self.judge(fact, ret, trial)
            for fact, ret in zip(facts, retrievals)
        ]
        return await asyncio.gather(*tasks)
