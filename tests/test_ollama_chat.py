"""Exercise the Ollama HTTP protocol and terminal flow without downloading a model."""

from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import unittest
from unittest import mock

import ollama_chat
from ollama_chat import OllamaClient, OllamaError


ROOT = Path(__file__).resolve().parents[1]


def ndjson(*records):
    return b"".join(json.dumps(record).encode() + b"\n" for record in records)


def reply(text="Hello!"):
    return ndjson(
        {"message": {"role": "assistant", "content": text}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True},
    )


@contextmanager
def ollama_server(models=None, responses=None, tags_status=200, tags_body=None, pulls=None):
    """Run a real local HTTP endpoint and retain requests for assertions."""
    state = {
        "models": models if models is not None else ["alpha:latest", "beta:latest"],
        "responses": deque(responses or []),
        "pulls": deque(pulls or []),
        "requests": [],
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def send_body(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state["requests"].append(("GET", self.path, None))
            if self.path != "/api/tags":
                self.send_body(404, b'{"error":"unknown path"}')
                return
            body = tags_body
            if body is None:
                body = json.dumps({"models": [{"name": name} for name in state["models"]]}).encode()
            self.send_body(tags_status, body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(("POST", self.path, body))
            if self.path == "/api/pull":
                status, response = state["pulls"].popleft() if state["pulls"] else (200, ndjson({"status": "success"}))
                if status == 200 and b'"success"' in response and body["model"] not in state["models"]:
                    state["models"].append(body["model"])
                self.send_body(status, response, "application/x-ndjson")
                return
            if self.path != "/api/chat":
                self.send_body(404, b'{"error":"unknown path"}')
                return
            status, response = state["responses"].popleft() if state["responses"] else (200, reply())
            self.send_body(status, response, "application/x-ndjson")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    state["host"] = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class ClientTests(unittest.TestCase):
    def test_malformed_hosts_raise_application_errors(self):
        for host in ("http://[", "http://localhost:bad", "http://localhost:65536", "ftp://localhost"):
            with self.subTest(host=host):
                with self.assertRaises(OllamaError):
                    OllamaClient(host)

    def test_lists_models_from_tags_endpoint(self):
        with ollama_server() as server:
            self.assertEqual(OllamaClient(server["host"]).list_models(), server["models"])
            self.assertEqual(server["requests"], [("GET", "/api/tags", None)])

    def test_empty_model_list(self):
        with ollama_server(models=[]) as server:
            self.assertEqual(OllamaClient(server["host"]).list_models(), [])

    def test_streams_content_and_sends_model_and_history(self):
        response = ndjson(
            {"message": {"content": "Hello "}, "done": False},
            {"message": {"content": "world!"}, "done": False},
            {"message": {"content": ""}, "done": True},
        )
        history = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]
        with ollama_server(responses=[(200, response)]) as server:
            chunks = list(OllamaClient(server["host"]).chat("alpha:latest", history))
            self.assertEqual("".join(chunks), "Hello world!")
            method, path, body = server["requests"][0]
            self.assertEqual((method, path), ("POST", "/api/chat"))
            self.assertEqual(body["model"], "alpha:latest")
            self.assertEqual(body["messages"], history)
            self.assertIs(body["stream"], True)

    def test_content_on_final_record_is_not_lost(self):
        response = ndjson({"message": {"content": "Final text"}, "done": True})
        with ollama_server(responses=[(200, response)]) as server:
            self.assertEqual("".join(OllamaClient(server["host"]).chat("alpha", [])), "Final text")

    def test_reasoning_chunks_do_not_replace_the_answer(self):
        response = ndjson(
            {"message": {"thinking": "Considering the question", "content": ""}, "done": False},
            {"message": {"content": "The answer"}, "done": False},
            {"message": {"content": ""}, "done": True},
        )
        with ollama_server(responses=[(200, response)]) as server:
            self.assertEqual("".join(OllamaClient(server["host"]).chat("alpha", [])), "The answer")

    def test_chat_reports_http_and_protocol_errors(self):
        cases = [
            (404, b'{"error":"model missing; pull it first"}'),
            (500, b"server failed"),
            (200, b"not json\n"),
            (200, ndjson({"error": "runner failed"})),
            (200, ndjson({"message": {"content": "partial"}, "done": False})),
            (200, b""),
            (200, ndjson(["unexpected array"])),
            (200, ndjson({"message": {"content": 3}, "done": True})),
        ]
        for status, response in cases:
            with self.subTest(status=status, response=response):
                with ollama_server(responses=[(status, response)]) as server:
                    with self.assertRaises(OllamaError):
                        list(OllamaClient(server["host"]).chat("alpha", []))

    def test_listing_reports_http_and_protocol_errors(self):
        for status, body in [(503, b'{"error":"unavailable"}'), (200, b"invalid"), (200, b"[]")]:
            with self.subTest(status=status, body=body):
                with ollama_server(tags_status=status, tags_body=body) as server:
                    with self.assertRaises(OllamaError):
                        OllamaClient(server["host"]).list_models()

    def test_connection_errors_are_wrapped(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        host = f"http://127.0.0.1:{server.server_port}"
        server.server_close()
        with self.assertRaises(OllamaError):
            OllamaClient(host, timeout=0.1).list_models()
        with self.assertRaises(OllamaError):
            list(OllamaClient(host, timeout=0.1).chat("alpha", []))

    def test_pull_streams_progress_and_sends_model(self):
        progress = {"status": "downloading", "completed": 5, "total": 10}
        with ollama_server(pulls=[(200, ndjson(progress, {"status": "success"}))]) as server:
            records = list(OllamaClient(server["host"]).pull_model("alpha:latest"))
            self.assertEqual(records, [progress, {"status": "success"}])
            method, path, body = server["requests"][0]
            self.assertEqual((method, path), ("POST", "/api/pull"))
            self.assertEqual(body["model"], "alpha:latest")
            self.assertIs(body["stream"], True)

    def test_pull_errors_and_incomplete_download_raise(self):
        for status, body in [
            (404, b'{"error":"model not found"}'),
            (200, ndjson({"error": "disk full"})),
            (200, ndjson({"status": "downloading"})),
            (200, b"not json\n"),
        ]:
            with self.subTest(status=status, body=body):
                with ollama_server(pulls=[(status, body)]) as server:
                    with self.assertRaises(OllamaError):
                        list(OllamaClient(server["host"]).pull_model("alpha:latest"))


class TerminalTests(unittest.TestCase):
    def run_cli(self, server, text, *args):
        result = subprocess.run(
            [sys.executable, str(ROOT / "ollama_chat.py"), "--host", server["host"], *args],
            input=text,
            text=True,
            capture_output=True,
            cwd=ROOT,
            timeout=10,
        )
        return result

    def test_two_turn_chat_keeps_history_and_system_prompt(self):
        with ollama_server(responses=[(200, reply("First answer")), (200, reply("Second answer"))]) as server:
            result = self.run_cli(server, "Hello\nAgain\n/quit\n", "--model", "alpha:latest", "--system", "Be brief.")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("First answer", result.stdout)
            self.assertIn("Second answer", result.stdout)
            chats = [body for method, _, body in server["requests"] if method == "POST"]
            self.assertEqual(len(chats), 2)
            self.assertEqual(chats[1]["messages"], [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Again"},
            ])

    def test_clear_resets_history_and_preserves_system_prompt(self):
        with ollama_server() as server:
            result = self.run_cli(server, "Before\n/clear\nAfter\n/exit\n", "--model", "alpha:latest", "--system", "Be brief.")
            self.assertEqual(result.returncode, 0, result.stderr)
            chats = [body for method, _, body in server["requests"] if method == "POST"]
            self.assertEqual(len(chats), 2)
            self.assertEqual(chats[1]["messages"], [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "After"},
            ])

    def test_failed_turn_is_not_added_to_next_request(self):
        with ollama_server(responses=[(200, ndjson({"error": "runner failed"})), (200, reply("Recovered"))]) as server:
            result = self.run_cli(server, "Failed prompt\nRetry\n/quit\n", "--model", "alpha:latest")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Recovered", result.stdout)
            chats = [body for method, _, body in server["requests"] if method == "POST"]
            self.assertEqual(len(chats), 2)
            self.assertEqual(chats[1]["messages"], [{"role": "user", "content": "Retry"}])

    def test_cancelled_chat_does_not_commit_partial_turn(self):
        calls = []

        def stream(model, messages, context_length):
            calls.append((model, [dict(message) for message in messages], context_length))
            if len(calls) == 1:
                yield "Partial answer"
                raise KeyboardInterrupt
            yield "Recovered answer"

        stdout = io.StringIO()
        with mock.patch("builtins.input", side_effect=["Interrupted prompt", "Next prompt", "/quit"]), \
                mock.patch.object(OllamaClient, "list_models", return_value=["alpha:latest"]), \
                mock.patch.object(OllamaClient, "chat", side_effect=stream), \
                redirect_stdout(stdout), redirect_stderr(io.StringIO()):
            result = ollama_chat.main(["--model", "alpha:latest"])
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1], [{"role": "user", "content": "Next prompt"}])
        self.assertIn("Partial answer", stdout.getvalue())
        self.assertIn("Recovered answer", stdout.getvalue())

    def test_cancelled_pull_option_returns_interrupted_status(self):
        with mock.patch.object(OllamaClient, "pull_model", side_effect=KeyboardInterrupt), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = ollama_chat.main(["--pull", "alpha:latest"])
        self.assertEqual(result, 130)

    def test_malformed_host_reports_error_without_traceback(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "ollama_chat.py"), "--host", "http://["],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=10,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_switching_models_starts_a_new_conversation(self):
        with ollama_server() as server:
            result = self.run_cli(server, "Before\n/model beta:latest\nAfter\n/quit\n", "--model", "alpha:latest")
            self.assertEqual(result.returncode, 0, result.stderr)
            chats = [body for method, path, body in server["requests"] if path == "/api/chat"]
            self.assertEqual(len(chats), 2)
            self.assertEqual(chats[1]["model"], "beta:latest")
            self.assertEqual(chats[1]["messages"], [{"role": "user", "content": "After"}])

    def test_context_length_option_reaches_ollama(self):
        with ollama_server() as server:
            result = self.run_cli(server, "Hi\n/quit\n", "--model", "alpha:latest", "--context-length", "2048")
            self.assertEqual(result.returncode, 0, result.stderr)
            chats = [body for method, path, body in server["requests"] if path == "/api/chat"]
            self.assertEqual(len(chats), 1)
            self.assertEqual(chats[0]["options"]["num_ctx"], 2048)

    def test_eof_exits_cleanly(self):
        with ollama_server() as server:
            result = self.run_cli(server, "", "--model", "alpha:latest")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(any(method == "POST" for method, _, _ in server["requests"]))

    def test_no_installed_models_shows_download_guidance(self):
        with ollama_server(models=[]) as server:
            result = self.run_cli(server, "/quit\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("pull", result.stdout + result.stderr)

    def test_offline_catalog_does_not_require_ollama(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "ollama_chat.py"), "--host", "http://127.0.0.1:1", "--catalog"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("24", result.stdout)

    def test_pull_option_downloads_and_exits(self):
        with ollama_server(models=[]) as server:
            result = self.run_cli(server, "", "--pull", "alpha:latest")
            self.assertEqual(result.returncode, 0, result.stderr)
            pulls = [body for method, path, body in server["requests"] if path == "/api/pull"]
            self.assertEqual(len(pulls), 1)
            self.assertEqual(pulls[0]["model"], "alpha:latest")

    def test_download_from_empty_install_then_chat(self):
        with ollama_server(models=[]) as server:
            result = self.run_cli(server, "/pull alpha:latest\n/model alpha:latest\nHello\n/quit\n")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Hello!", result.stdout)
            chats = [body for method, path, body in server["requests"] if path == "/api/chat"]
            self.assertEqual(len(chats), 1)
            self.assertEqual(chats[0]["model"], "alpha:latest")


if __name__ == "__main__":
    unittest.main()
