import tempfile
import unittest
from pathlib import Path


class _DummyEngine:
    """Minimal engine stub implementing the pieces ProjectMemory uses."""

    def __init__(self, max_context_length: int = 256):
        self.model_name = "dummy"
        self.system_prompt = "You are a test engine."
        self._max_context_length = max_context_length
        self.is_cloud = False

    @property
    def max_context_length(self) -> int:
        return self._max_context_length

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def compress_prompt(self, prompt: str, target_tokens: int) -> str:
        # Roughly keep target_tokens*4 chars
        keep = max(32, target_tokens * 4)
        return prompt[:keep]


class TestPressureValve(unittest.TestCase):
    def test_build_prompt_compresses_retrieved_memory(self):
        from engram.project_memory import ProjectMemory

        with tempfile.TemporaryDirectory() as td:
            pm = ProjectMemory(
                project_id="p1",
                project_type="general",
                base_dir=Path(td),
                llm_engine=_DummyEngine(max_context_length=256),
            )

            # Stuff working memory with content to force overflow
            for i in range(50):
                pm.working.add("user", f"message {i} " + ("x" * 50))

            out = pm.build_prompt(
                user_message="hello",
                max_prompt_tokens=256,
                reserve_output_tokens=64,
            )
            self.assertTrue(out["compressed"])
            # best-effort token count should be within usable cap
            self.assertLessEqual(int(out["prompt_tokens"]), 192)


if __name__ == "__main__":
    unittest.main()
