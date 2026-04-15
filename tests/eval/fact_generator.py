"""
Engram Memory Eval — Fact Corpus Generator

Uses Claude to generate a labeled, structured fact corpus with:
  - canonical fact statements
  - direct retrieval queries
  - paraphrase queries
  - adversarial decoy queries
  - expected retrieval snippet
  - category + salience label
"""
import json
import random
import asyncio
import aiohttp
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

CATEGORIES = [
    "person",        # biographical facts about fictional people
    "event",         # past events with dates/locations
    "preference",    # likes/dislikes, habits
    "skill",         # abilities, expertise levels
    "contradiction", # facts designed to be overwritten by later contradictions
    "distractor",    # topically similar but retrievally irrelevant
]

GENERATION_PROMPT = """
Generate a synthetic fact corpus for testing an LLM memory system. 
Output ONLY a JSON array, no markdown, no commentary.

Generate {n} facts for category: "{category}"

Each fact object must have exactly these fields:
{{
  "id": "unique string like '{category}_001'",
  "category": "{category}",
  "salience": "high|medium|low",
  "canonical": "A single declarative sentence stating the fact.",
  "contradiction": "A sentence that directly contradicts the canonical fact (used to test fact updating).",
  "direct_query": "A question whose answer requires this specific fact.",
  "paraphrase_query": "Same question, differently worded.",
  "decoy_query": "A superficially similar question that should NOT retrieve this fact.",
  "expected_snippet": "A 5-15 word phrase that should appear in a correct retrieval result."
}}

Rules:
- Facts must be self-contained (no references to prior context).
- People should be fictional but realistic-sounding.
- Events should have specific dates/locations.
- High-salience facts are emotionally significant or frequently referenced.
- Decoy queries must NOT share named entities (person names, place names, org names)
  with the canonical fact. The decoy should be topically adjacent but reference
  a DIFFERENT named entity entirely — not a slight variation of the same name.
  BAD: canonical mentions "Grand Oak Restaurant", decoy asks about "Golden Oak Restaurant".
  GOOD: canonical mentions "Grand Oak Restaurant", decoy asks about "La Piazza Bistro".
- Decoy queries should share vocabulary but target genuinely different facts.
- Do NOT output markdown fences. Output raw JSON array only.
"""


@dataclass
class Fact:
    id: str
    category: str
    salience: str           # high | medium | low
    canonical: str          # the ground-truth fact
    contradiction: str      # contradicting version (for trial 4)
    direct_query: str
    paraphrase_query: str
    decoy_query: str
    expected_snippet: str
    injected_count: int = 0     # updated by trial runner
    last_injected_trial: Optional[int] = None

    def as_dict(self) -> dict:
        return asdict(self)


class FactCorpus:
    def __init__(self, facts: list[Fact]):
        self.facts = facts
        self._by_id = {f.id: f for f in facts}
        self._by_category = {}
        for f in facts:
            self._by_category.setdefault(f.category, []).append(f)

    def subset(self, label: str) -> list[Fact]:
        """Return subset by logical name used in schedule."""
        if label == "all":
            return self.facts
        elif label == "high_salience":
            return [f for f in self.facts if f.salience == "high"]
        elif label == "contradictions":
            # Return contradiction versions of high-salience facts
            return [f for f in self.facts if f.salience == "high"]
        elif label == "none":
            return []
        else:
            return self._by_category.get(label, [])

    def by_id(self, fid: str) -> Fact:
        return self._by_id[fid]

    def save(self, path: str):
        Path(path).write_text(json.dumps([f.as_dict() for f in self.facts], indent=2))

    @classmethod
    def load(cls, path: str) -> "FactCorpus":
        raw = json.loads(Path(path).read_text())
        facts = [Fact(**r) for r in raw]
        return cls(facts)

    def __len__(self):
        return len(self.facts)


async def _generate_category(
    session: aiohttp.ClientSession,
    category: str,
    n: int,
    model: str,
    retries: int = 3,
) -> list[dict]:
    prompt = GENERATION_PROMPT.format(category=category, n=n)
    payload = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }

    for attempt in range(retries):
        try:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers=headers,
            ) as resp:
                data = await resp.json()
                if "content" not in data:
                    raise RuntimeError(
                        f"API error (HTTP {resp.status}): {data.get('error', data)}"
                    )
                text = data["content"][0]["text"].strip()
                # Strip accidental markdown fences
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return json.loads(text.strip())
        except (json.JSONDecodeError, aiohttp.ClientError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Failed to generate {category} after {retries} attempts: {e}")
            await asyncio.sleep(2 ** attempt)
        except RuntimeError:
            raise  # don't retry API auth/format errors
    return []


async def generate_corpus(
    n_per_category: int = 10,
    model: str = "claude-sonnet-4-20250514",
    cache_path: Optional[str] = None,
) -> FactCorpus:
    """
    Generate full fact corpus. Uses cache_path if it exists.
    """
    if cache_path and Path(cache_path).exists():
        print(f"[corpus] Loading cached corpus from {cache_path}")
        return FactCorpus.load(cache_path)

    print(f"[corpus] Generating {n_per_category} facts × {len(CATEGORIES)} categories via Claude...")
    all_facts = []

    async with aiohttp.ClientSession() as session:
        tasks = [
            _generate_category(session, cat, n_per_category, model)
            for cat in CATEGORIES
        ]
        results = await asyncio.gather(*tasks)

    for cat, raw_facts in zip(CATEGORIES, results):
        for rf in raw_facts:
            try:
                all_facts.append(Fact(**{k: rf[k] for k in Fact.__dataclass_fields__ 
                                         if k not in ("injected_count", "last_injected_trial")}))
            except (KeyError, TypeError) as e:
                print(f"[corpus] Warning: malformed fact in {cat}: {e} — skipping")

    corpus = FactCorpus(all_facts)
    if cache_path:
        corpus.save(cache_path)
        print(f"[corpus] Saved {len(corpus)} facts to {cache_path}")

    return corpus
