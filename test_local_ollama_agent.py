import unittest
from unittest.mock import patch

from local_ollama_agent import ToolRunner, normalize_model_name, parse_response


class AgentTests(unittest.TestCase):
    """Guard critical parsing, workspace, and command-approval behavior."""
    def test_normalizes_user_model_spelling(self):
        self.assertEqual(normalize_model_name("qwen2.5coder7b"), "qwen2.5-coder:7b")

    def test_parses_tool_json(self):
        tool, final = parse_response('{"tool":"list_dir","args":{"path":"."}}')
        self.assertEqual(tool, {"tool": "list_dir", "args": {"path": "."}})
        self.assertIsNone(final)

    def test_parses_final_json(self):
        tool, final = parse_response('{"final":"Bitti."}')
        self.assertIsNone(tool)
        self.assertEqual(final, "Bitti.")

    def test_parses_fenced_tool_json_with_nested_arguments(self):
        tool, final = parse_response(
            '```json\n{"tool":"write_file","args":{"path":"a.txt","content":"{}"}}\n```'
        )
        self.assertEqual(
            tool,
            {"tool": "write_file", "args": {"path": "a.txt", "content": "{}"}},
        )
        self.assertIsNone(final)

    def test_rejects_workspace_escape(self):
        with self.assertRaises(ValueError):
            ToolRunner(".")._inside_workspace("..\\outside.txt")

    @patch("builtins.input", return_value="hayır")
    @patch("local_ollama_agent.subprocess.run")
    def test_denied_shell_command_is_not_run(self, run, _input):
        result = ToolRunner(".").run_shell("echo test")
        self.assertIn("onay vermedi", result)
        run.assert_not_called()

    @patch("builtins.input", return_value="evet")
    @patch("local_ollama_agent.subprocess.run")
    def test_shell_command_requires_confirmation(self, run, _input):
        run.return_value.stdout = "ok"
        run.return_value.stderr = ""
        run.return_value.returncode = 0
        result = ToolRunner(".").run_shell("echo test")
        self.assertIn("ok", result)
        run.assert_called_once()

    @patch("builtins.input", return_value="hayır")
    @patch("local_ollama_agent.GitHubTools.publish_project")
    def test_github_publish_requires_confirmation(self, publish, _input):
        result = ToolRunner(".").publish_github_project("example", private=False)
        self.assertIn("onay vermedi", result)
        publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
