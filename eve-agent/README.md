# EVE

Gemini-powered terminal agent with shell execution, btop-style neon look.

> ⚠️ **Security warning:** EVE can execute shell commands **without asking
> for confirmation**, guarded only by a small blacklist of catastrophic
> patterns (`rm -rf /`, `mkfs`, fork bombs, etc — see `shell_exec.py`).
> Anything not on that blacklist runs immediately, including `sudo` commands.
> This is a deliberate design choice for personal, trusted use — **do not**
> run this against untrusted input, expose it to other users, or use it on a
> machine where an unexpected command could cause real damage.

## Install

```bash
pip install textual httpx --break-system-packages
```

(or, better, in a venv: `python -m venv venv && source venv/bin/activate && pip install textual httpx`)

## Usage

```bash
cd eve-agent
python app.py
```

On launch, the input shows a `/help` hint listing every command. First time,
save your **Gemini** API key (get one at
[aistudio.google.com/apikey](https://aistudio.google.com/apikey)):
```
/key YOUR_GEMINI_API_KEY_HERE
```
Stored in `~/.config/eve/gemini.key` (permissions 600) — you won't need to repeat this.

> **Supported model:** EVE's Online mode is hardcoded to `gemini-3.5-flash-lite`
> (see `MODEL` in `gemini.py`) — no other Gemini model is currently supported.
> If you want a different one, edit that constant.

Other commands:
- `/help` — shows the command reference again
- `/new` — clears the chat
- `/settings` — opens the System prompt / User prompt screen (also `Ctrl+S`)
- `/localmodel <path>` — sets the local `.gguf` model path (see "Local mode" below)
- `/localserver <path>` — sets the `llama-server` binary/path (default: `llama-server` on PATH)
- `/quit` — exit
- `Ctrl+N` — new chat (shortcut)
- `Ctrl+C` — exit

## Default system prompt

Out of the box, EVE is configured as an agent with full control over the
terminal of a Linux distribution — it knows the conventional paths
(`~/.config`, `~/.local/bin`, `/etc`, `/var/log`, etc.) and is told to check
`/etc/os-release` before picking a package manager. You can fully replace
this from `/settings` if you want a different persona or scope.

## Local mode (llama.cpp)

The **Local**/**Online** buttons in the header switch backends. Local mode
runs entirely offline through your own `llama.cpp` build's `llama-server`
(its built-in OpenAI-compatible HTTP server) — EVE launches it as a
subprocess and talks to it over `http://127.0.0.1:8080`, streaming replies
and tool calls exactly like it does with Gemini.

You need `llama.cpp` built on your own machine — EVE does not bundle or
install it for you. From scratch:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON      # drop -DGGML_CUDA=ON if you have no NVIDIA GPU
cmake --build build --target llama-server --config Release -j
```
The resulting binary lands at `build/bin/llama-server`. (Swap
`-DGGML_CUDA=ON` for `-DGGML_METAL=ON` on Apple Silicon, or omit GPU flags
entirely for a CPU-only build — see `llama.cpp`'s own README for other
backends, like ROCm or Vulkan.)

You'll also need a **local model in GGUF format** — e.g. downloaded from
Hugging Face (search for a `GGUF` quant of the model you want, such as a
Qwen3 build). Tool calling (shell execution) needs a chat template that
supports it; Qwen3-family models support it out of the box.

Once you have both:

1. Point EVE at your `llama-server` binary (skip this if it's already on
   your `PATH`):
   ```
   /localserver /path/to/llama.cpp/build/bin/llama-server
   ```
2. Point EVE at your `.gguf` model file:
   ```
   /localmodel /home/you/models/Qwen3.5-4B-IQ4_XS.gguf
   ```
3. Click **Local** in the header (or the mode will already be Local if you
   left it there last time). The first message after switching starts
   `llama-server` (this can take a bit while it loads the model into
   memory/VRAM) and reuses that same process for the rest of the session.

EVE always starts `llama-server` with `--jinja`, which is required for
tool-calling chat templates to work.

Notes:
- GPU offload defaults to `-ngl 999` (as many layers as fit); context size
  defaults to `8192`. Both are read from `~/.config/eve/config.json`
  (`local_ngl`, `local_ctx`) if you want to hand-edit them — no in-app
  control for these yet.
- If a `llama-server` is already running on the configured port (e.g. you
  started one yourself for testing), EVE just uses it as-is instead of
  launching a second one.
- Switching back to Online doesn't kill the local server; it keeps running
  in the background so switching back to Local is instant. It's stopped
  when you quit EVE (only if EVE started it).

## Settings (System prompt / User prompt)

`Ctrl+S` or `/settings` opens a screen with two text areas:

- **System prompt** — fully replaces EVE's base instructions (defaults to
  the Linux terminal agent persona described above).
- **User prompt** — free text about you (who you are, your setup, preferences) that
  gets appended to the end of the system prompt automatically on every message.

Saved to `~/.config/eve/config.json` and loaded automatically the next time
you open the app — no need to set them again.

## Command execution

When you ask for something that needs shell access, EVE runs the command
**with no confirmation prompt**. There's only a minimal blacklist of
catastrophic/irreversible patterns (`sudo`, `rm -rf /`, `mkfs`, `dd` to disks,
fork bombs, `wipefs`, etc. — see `shell_exec.py`, `CATASTROPHIC_PATTERNS`).
Everything else runs freely, as requested.

## Markdown rendering

EVE's replies, the executed-command bubbles, and results render as real
Markdown (via Textual's `Markdown` widget) — headings, bullet lists, and
fenced code blocks with a distinct highlighted background. Your own messages
stay as plain text.

## Retractable panel on Hyprland

To use it as a floating panel that appears/disappears on a keybind, with no
layer-shell or Quickshell involved — launch it in a kitty window (or your
terminal) with Hyprland rules:

```ini
# hyprland.conf
windowrulev2 = float, class:^(kitty-eve)$
windowrulev2 = size 500 700, class:^(kitty-eve)$
windowrulev2 = move 100%-520 40, class:^(kitty-eve)$  # adjust position
windowrulev2 = workspace special:eve, class:^(kitty-eve)$

bind = SUPER, N, togglespecialworkspace, eve
exec-once = kitty --class kitty-eve -e python /path/to/eve-agent/app.py
```

`togglespecialworkspace` gives you the native "retractable" show/hide effect
from Hyprland, zero dependency on Quickshell/GlobalStates, and if something
breaks inside the TUI it can't take down your whole graphical session like
the earlier attempts did.

## Files

- `app.py` — the Textual app (UI, function-calling loop, chat-style bubbles)
- `config.py` — API key + system/user prompt + local-model settings storage
- `gemini.py` — Gemini client with SSE streaming
- `local_llm.py` — local backend: launches `llama-server` and streams from its OpenAI-compatible API
- `shell_exec.py` — command execution + blacklist
- `eve.tcss` — styles, "Ciudad Sion" palette (pink/cyan on black)
