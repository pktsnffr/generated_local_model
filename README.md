# Ollama terminal chat

A simple terminal client for local Ollama models: streaming chat, model downloads,
and conversation history. Requires **Python 3.10+** and
[Ollama](https://ollama.com/download); no pip packages are needed.

## Run

1. Install Python and Ollama, then get the project:

```sh
git clone https://github.com/pktsnffr/generated_local_model.git
cd generated_local_model
```

2. Start the Ollama app, or keep this command running in a separate terminal:

```sh
ollama serve
```

3. In the project directory, choose a model, download it, and start chatting:

```sh
python3 ollama_chat.py --catalog             # Suggestions for 24 GB VRAM
python3 ollama_chat.py --pull qwen3:14b       # One-time download
python3 ollama_chat.py --model qwen3:14b      # Start here on later runs
```

Run `python3 ollama_chat.py` without options to choose an installed model.
Downloads need Internet access and free disk space on the Ollama server, which
stores the weights outside this repository.

## During chat

| Command | Action |
| --- | --- |
| `/catalog` | Show suggested models for 24 GB VRAM |
| `/pull <model>` | Download with progress; repeat to resume |
| `/models` | List installed models |
| `/model [name]` | Select a model and reset the conversation |
| `/clear` | Reset the conversation, keeping the system prompt |
| `/help` | Show all commands |
| `/quit` or `/exit` | Exit |

Enter one message per line. Ctrl+C cancels a reply; at the prompt, Ctrl+C or
Ctrl+D exits. History lasts only for the session. Text input only.

## Memory and troubleshooting

The curated catalog lists default quantized weights of roughly **5–20 GB**.
Context and runtime need extra VRAM, so fit on a 24 GB GPU is an estimate.
The default context is **4096 tokens**. Check placement with `ollama ps`; choose
an 8B or 14B model if Ollama offloads to RAM. See the
[model library](https://ollama.com/library) for current tags and sizes.

- Connection error: start Ollama and check `--host`.
- Missing model: run `/pull <model>`, then `/model <model>`.
- Slow load/download: increase `--timeout 300`; repeat `/pull` after interruption.

```sh
python3 ollama_chat.py --help
python3 ollama_chat.py --model qwen3:14b --system "Answer concisely."
python3 ollama_chat.py --host http://127.0.0.1:11434 --context-length 4096 --timeout 300
```

The host defaults to `OLLAMA_HOST` or `http://127.0.0.1:11434`. In a cloud
workspace, localhost means the cloud machine. Use a trusted, securely connected
host to reach Ollama elsewhere; prompts are sent to that server.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

Tests use a mock API; no Ollama server, model download, or GPU is required.
