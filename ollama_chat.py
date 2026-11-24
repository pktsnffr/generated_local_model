#!/usr/bin/env python3
"""A dependency-free terminal client for Ollama (Python 3.10+)."""

import argparse
from contextlib import contextmanager
import http.client
import json
import os
import sys
from urllib import error, parse, request


# Approximate default-quantization weight/download sizes in decimal GB.
# These leave some room on a 24 GiB GPU; context and runtime also consume VRAM.
MODEL_CATALOG = (
    ("llama3.1:8b", 4.9, "General chat"),
    ("qwen3:8b", 5.2, "Chat and reasoning"),
    ("gemma3:12b", 8.1, "Chat and vision"),
    ("qwen3:14b", 9.3, "Chat and reasoning"),
    ("gemma3:27b", 17.0, "Larger chat and vision model"),
    ("qwen3:30b", 19.0, "Mixture-of-experts reasoning"),
    ("qwen2.5-coder:32b", 20.0, "Code generation"),
)


class OllamaError(Exception):
    """A connection, API, or response-format error."""


def _json_object(data):
    try:
        value = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise OllamaError("Ollama returned invalid JSON.") from exc
    if not isinstance(value, dict):
        raise OllamaError("Ollama returned an invalid response object.")
    if value.get("error"):
        raise OllamaError(str(value["error"]))
    return value


class OllamaClient:
    def __init__(self, host="http://127.0.0.1:11434", timeout=120.0):
        if "://" not in host:
            host = "http://" + host
        try:
            url = parse.urlsplit(host)
            # Accessing port validates malformed and out-of-range port numbers.
            url.port
        except ValueError as exc:
            raise OllamaError(f"Invalid Ollama host: {exc}") from exc
        if url.scheme not in ("http", "https") or not url.hostname:
            raise OllamaError("The Ollama host must be an HTTP or HTTPS address.")
        if url.query or url.fragment or url.username or url.password:
            raise OllamaError("The Ollama host must not contain credentials, a query, or a fragment.")
        self.host = host.rstrip("/")
        self.timeout = timeout

    @contextmanager
    def _response(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.host + path, data=data,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                yield response
        except error.HTTPError as exc:
            with exc:
                try:
                    body = json.loads(exc.read(65536))
                    detail = body.get("error", exc.reason) if isinstance(body, dict) else exc.reason
                except (ValueError, UnicodeDecodeError, OSError):
                    detail = exc.reason
            raise OllamaError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
        except (error.URLError, OSError, http.client.HTTPException) as exc:
            raise OllamaError(
                f"Cannot communicate with Ollama at {self.host}: {exc}. "
                "Start Ollama with 'ollama serve' and check --host. "
                "For slow models or downloads, increase --timeout."
            ) from exc
        except ValueError as exc:
            raise OllamaError(f"Invalid Ollama request: {exc}") from exc

    def list_models(self):
        with self._response("/api/tags") as response:
            result = _json_object(response.read())
        models = result.get("models")
        if not isinstance(models, list) or any(
            not isinstance(model, dict) or not isinstance(model.get("name"), str)
            for model in models
        ):
            raise OllamaError("Ollama returned an invalid model list.")
        return sorted({model["name"] for model in models})

    def chat(self, model, messages, context_length=4096):
        payload = {"model": model, "messages": messages, "stream": True,
                   "options": {"num_ctx": context_length}}
        with self._response("/api/chat", payload) as response:
            for line in response:
                if not line.strip():
                    continue
                event = _json_object(line)
                message = event.get("message", {})
                if not isinstance(message, dict) or not isinstance(message.get("content", ""), str):
                    raise OllamaError("Ollama returned an invalid chat message.")
                if message.get("content"):
                    yield message["content"]
                if event.get("done") is True:
                    return
        raise OllamaError("Ollama closed the chat stream before it finished. Please retry.")

    def pull_model(self, model):
        with self._response("/api/pull", {"model": model, "stream": True}) as response:
            for line in response:
                if not line.strip():
                    continue
                event = _json_object(line)
                yield event
                if event.get("status") == "success":
                    return
        raise OllamaError("Ollama closed the download stream before it finished. Retry to resume.")


def show_catalog():
    print("\nModels expected to fit a 24 GB GPU at the default quantization:")
    print(f"{'Model':<24} {'Weights':>8}  Use")
    for name, size, description in MODEL_CATALOG:
        print(f"{name:<24} {size:>5.1f} GB  {description}")
    print("\nSizes are approximate, not total VRAM usage. Context, vision inputs, and")
    print("other GPU processes need extra memory. This is a curated list, not live search.")
    print("Use /pull <model> to download; /models lists installed models.")
    print("Ollama may offload to RAM if VRAM is insufficient; check 'ollama ps'.\n")


def download_model(client, model):
    if model not in {entry[0] for entry in MODEL_CATALOG}:
        print("This model is outside the 24 GB catalog; its memory requirements are unknown.")
    print(f"Downloading {model} through Ollama (Ctrl+C cancels waiting).")
    previous = None
    for event in client.pull_model(model):
        status = str(event.get("status", "Downloading"))
        progress_key = (status, event.get("digest"))
        total, completed = event.get("total"), event.get("completed")
        if isinstance(total, (int, float)) and total > 0 and isinstance(completed, (int, float)):
            percent = min(100, max(0, int(completed * 100 / total)))
            progress_key += (percent,)
            status += f" {percent:3d}% ({completed / 1e9:.2f}/{total / 1e9:.2f} GB)"
        if progress_key != previous:
            print(status, flush=True)
            previous = progress_key
    print(f"Ready: {model}. Use /model {model} to chat.")


HELP = """Commands:
  /help           Show this help
  /catalog        Suggested downloadable models for a 24 GB GPU
  /pull <model>   Download a model (retry the same command to resume)
  /models         List installed models
  /model [name]   Switch model, clearing conversation history
  /clear          Clear conversation history, keeping the system prompt
  /quit or /exit  Exit
Type a message and press Enter to chat. Ctrl+C cancels a reply; Ctrl+D exits.
"""


def resolve_model(name, models):
    if name in models:
        return name
    if name + ":latest" in models:
        return name + ":latest"
    raise OllamaError(f"Model '{name}' is not installed. Download it with /pull {name}.")


def choose_model(models):
    if not models:
        print("No models installed. Use /catalog, then /pull <model> and /model <model>.")
        return None
    if len(models) == 1:
        return models[0]
    for index, name in enumerate(models, 1):
        print(f"  {index}. {name}")
    while True:
        choice = input("Choose model number or name [1]: ").strip() or "1"
        if choice.isdigit() and 1 <= int(choice) <= len(models):
            return models[int(choice) - 1]
        try:
            return resolve_model(choice, models)
        except OllamaError as exc:
            print(exc)


def positive_number(value):
    number = float(value)
    if not 0 < number < float("inf"):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
                        help="Ollama server URL (default: OLLAMA_HOST or http://127.0.0.1:11434)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--model", help="Start chatting with an installed model")
    action.add_argument("--catalog", action="store_true", help="Show 24 GB model suggestions and exit")
    action.add_argument("--pull", metavar="MODEL", help="Download a model and exit")
    parser.add_argument("--system", default="", help="Optional system prompt")
    parser.add_argument("--timeout", type=positive_number, default=120.0,
                        help="Network inactivity timeout in seconds (default: 120)")
    parser.add_argument("--context-length", type=positive_int, default=4096,
                        help="Context window in tokens; larger values use more VRAM (default: 4096)")
    args = parser.parse_args(argv)
    if args.catalog:
        show_catalog()
        return 0
    try:
        client = OllamaClient(args.host, args.timeout)
        if args.pull:
            download_model(client, args.pull)
            return 0
        models = client.list_models()
        print("Ollama terminal chat | /help for commands | /catalog for 24 GB models")
        model = resolve_model(args.model, models) if args.model else choose_model(models)
    except OllamaError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled waiting. Ollama may still be working.")
        return 130
    except EOFError:
        print("\nGoodbye.")
        return 0

    initial = [{"role": "system", "content": args.system}] if args.system else []
    messages = initial.copy()
    if model:
        print(f"Using {model} (context: {args.context_length} tokens).")
    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        if not prompt:
            continue
        if prompt.startswith("/"):
            parts = prompt.split(maxsplit=1)
            command, argument = parts[0].lower(), parts[1].strip() if len(parts) > 1 else ""
            try:
                if command in ("/quit", "/exit"):
                    print("Goodbye.")
                    return 0
                elif command == "/help":
                    print(HELP)
                elif command == "/catalog":
                    show_catalog()
                elif command == "/models":
                    models = client.list_models()
                    print("\n".join(f"{'*' if name == model else ' '} {name}" for name in models)
                          or "No models installed. Use /catalog and /pull <model>.")
                elif command == "/pull":
                    if not argument:
                        print("Usage: /pull <model> (see /catalog)")
                    else:
                        download_model(client, argument)
                elif command == "/model":
                    models = client.list_models()
                    selected = resolve_model(argument, models) if argument else choose_model(models)
                    if selected:
                        model, messages = selected, initial.copy()
                        print(f"Using {model}. Conversation cleared.")
                elif command == "/clear":
                    messages = initial.copy()
                    print("Conversation cleared.")
                else:
                    print("Unknown command. Type /help for commands.")
            except OllamaError as exc:
                print(f"Error: {exc}", file=sys.stderr)
            except KeyboardInterrupt:
                print("\nCancelled waiting. Ollama may still be working; retry /pull to resume.")
            except EOFError:
                print("\nGoodbye.")
                return 0
            continue
        if not model:
            print("Select an installed model with /model, or download one with /pull <model>.")
            continue
        user_message = {"role": "user", "content": prompt}
        chunks = []
        print(f"{model}> ", end="", flush=True)
        try:
            for chunk in client.chat(model, messages + [user_message], args.context_length):
                print(chunk, end="", flush=True)
                chunks.append(chunk)
        except KeyboardInterrupt:
            print("\nReply cancelled; this turn was not added to history.")
        except OllamaError as exc:
            print(f"\nError: {exc}\nThis turn was not added to history.", file=sys.stderr)
        else:
            print()
            messages.extend([user_message, {"role": "assistant", "content": "".join(chunks)}])


if __name__ == "__main__":
    sys.exit(main())
