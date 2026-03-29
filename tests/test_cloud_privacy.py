import unittest


class TestCloudPrivacy(unittest.TestCase):
    def test_sanitize_query_only_strips_memory(self):
        from engram.engine.utils.privacy import sanitize_prompt_for_cloud

        prompt = (
            "System\n\n"
            "----- BEGIN RETRIEVED MEMORY -----\n"
            "[WORKING]\nsecret\n"
            "----- END RETRIEVED MEMORY -----\n\n"
            "User: hi\nAssistant:"
        )
        out = sanitize_prompt_for_cloud(prompt, policy="query_only")
        self.assertNotIn("secret", out)
        self.assertIn("User: hi", out)

    def test_sanitize_query_plus_summary_replaces_memory(self):
        from engram.engine.utils.privacy import sanitize_prompt_for_cloud

        prompt = (
            "----- BEGIN RETRIEVED MEMORY -----\n"
            "[WORKING]\nsecret\n"
            "----- END RETRIEVED MEMORY -----\n\n"
            "User: hi\nAssistant:"
        )
        out = sanitize_prompt_for_cloud(prompt, policy="query_plus_summary")
        self.assertIn("Retrieved memory summary", out)
        self.assertNotIn("BEGIN RETRIEVED MEMORY", out)
        self.assertIn("User: hi", out)


if __name__ == "__main__":
    unittest.main()
