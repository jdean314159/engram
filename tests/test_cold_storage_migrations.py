import tempfile
import unittest
from pathlib import Path


class TestColdStorageMigrations(unittest.TestCase):
    def test_dedup_insert_or_ignore(self):
        from engram.memory.cold_storage import ColdStorage

        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "cold.db"
            cold = ColdStorage(db_path=db_path)

            cold.archive(
                [
                    {"project_id": "p1", "session_id": "s1", "text": "hello world", "metadata": {"a": 1}},
                    {"project_id": "p1", "session_id": "s1", "text": "hello world", "metadata": {"a": 2}},
                ]
            )
            stats = cold.get_stats()
            # dedup is project-scoped, so duplicates should be ignored
            self.assertEqual(int(stats.get("total_rows", 0)), 1)


if __name__ == "__main__":
    unittest.main()
