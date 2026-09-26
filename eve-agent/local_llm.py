"""
Local model backend, via llama.cpp's `llama-server` (its built-in
OpenAI-compatible HTTP server). We don't talk to llama.cpp directly —
we spawn `llama-server` as a subprocess and speak `/v1/chat/completions`
to it over HTTP, streaming.

The yielded event shape from stream_reply() intentionally mirrors
gemini.stream_reply() exactly — ("text_delta"/"function_call"/"done"/
"error", payload) — so app.py can swap backends without branching on
anything but which stream_reply to call.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from typing import AsyncIterator, Optional

import httpx

from gemini import ChatSession

SHELL_TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "run_shell_command",
        "description": "Executes a shell command in the user's terminal and returns stdout/stderr.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The full command to run, exactly as you'd type it in bash.",
                }
            },
            "required": ["command"],
        },
    },
}


class LocalServerError(Exception):
    pass


# Module-level handle to a llama-server we launched ourselves, so we can
# reuse it across turns instead of respawning per message, and can shut
# it down cleanly on exit / mode switch.
_proc: Optional[subprocess.Popen] = None
_proc_model_path: Optional[str] = None


def base_url(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def is_server_running(url: str, timeout: float = 1.0) -> bool:
    try:
        r = httpx.get(f"{url}/health", timeout=timeout)
        return r.status_code == 200
    except httpx.RequestError:
        return False


def ensure_server(
    model_path: str,
    server_bin: str,
    port: int,
    ctx_size: int,
    ngl: int,
    startup_timeout: float = 120.0,
) -> None:
    """Makes sure a llama-server answering at base_url(port) is up and
    serving `model_path`. Blocking — call via asyncio.to_thread from the
    UI. Reuses an already-running server on that port as-is: if you (or a
    previous run) already have llama-server up on that port, we trust it
    rather than trying to detect/replace its model."""
    global _proc, _proc_model_path
    url = base_url(port)

    if is_server_running(url):
        return

    if not model_path:
        raise LocalServerError(
            "No local model configured. Set one with /localmodel /path/to/model.gguf"
        )

    resolved_bin = server_bin if os_path_is_abs_or_exists(server_bin) else shutil.which(server_bin)
    if not resolved_bin:
        raise LocalServerError(
            f"'{server_bin}' not found on PATH. Point EVE at your llama.cpp build with "
            "/localserver /path/to/llama.cpp/build/bin/llama-server (build it there first if needed)."
        )

    if _proc is not None and _proc.poll() is None:
        # Our own previous attempt is alive but not answering — kill and retry clean.
        _proc.kill()
        _proc.wait()

    try:
        _proc = subprocess.Popen(
            [
                resolved_bin,
                "-m", model_path,
                "--port", str(port),
                "--host", "127.0.0.1",
                "-c", str(ctx_size),
                "-ngl", str(ngl),
                "--jinja",  # needed for tool-calling chat templates (Qwen3 etc.)
                "--reasoning", "off",  # Qwen3 thinking disabled by default
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        raise LocalServerError(f"Failed to launch llama-server: {e}") from e
    _proc_model_path = model_path

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if _proc.poll() is not None:
            raise LocalServerError(
                f"llama-server exited immediately (code {_proc.returncode}). "
                "Check the model path/quant and that this llama.cpp build supports it."
            )
        if is_server_running(url, timeout=1.0):
            return
        time.sleep(0.5)

    raise LocalServerError("Timed out waiting for llama-server to become ready (model load can be slow — try again).")


def os_path_is_abs_or_exists(path: str) -> bool:
    import os
    return os.path.isabs(path) or os.path.exists(path)


def stop_server() -> None:
    global _proc, _proc_model_path
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None
    _proc_model_path = None


def current_model_path() -> Optional[str]:
    return _proc_model_path


def _to_openai_messages(session: ChatSession, system_prompt: str) -> list[dict]:
    """Converts our Gemini-shaped ChatSession into OpenAI chat format.
    Gemini's Message objects don't carry tool_call ids, so we synthesize
    sequential call_N ids in emission order — the matching function-result
    message always immediately follows its function_call message, so the
    same counter value pairs them correctly."""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    call_counter = 0
    for m in session.messages:
        if m.role == "user":
            messages.append({"role": "user", "content": m.text})
        elif m.role == "model":
            msg: dict = {"role": "assistant", "content": m.text or None}
            if m.function_call:
                call_counter += 1
                msg["tool_calls"] = [
                    {
                        "id": f"call_{call_counter}",
                        "type": "function",
                        "function": {
                            "name": m.function_call["name"],
                            "arguments": json.dumps(m.function_call["args"], ensure_ascii=False),
                        },
                    }
                ]
            messages.append(msg)
        elif m.role == "function":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": f"call_{call_counter}",
                    "content": json.dumps(m.function_response["result"], ensure_ascii=False),
                }
            )
    return messages


async def stream_reply(
    session: ChatSession, system_prompt: str, port: int
) -> AsyncIterator[tuple[str, dict]]:
    """Same event shape as gemini.stream_reply:
      ("text_delta", {"text": "...", "thought_signature": None})
      ("function_call", {"name":.., "args":.., "thought_signature": None})
      ("done", {})
      ("error", {"message": "..."})
    """
    url = f"{base_url(port)}/v1/chat/completions"
    body = {
        "model": "local",
        "messages": _to_openai_messages(session, system_prompt),
        "tools": [SHELL_TOOL_OPENAI],
        "stream": True,
        "temperature": 0.7,
    }

    # Tool-call fragments stream in piecewise (name once, arguments in
    # chunks) keyed by index — accumulate until finish_reason fires.
    tool_calls: dict[int, dict] = {}

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            async with client.stream("POST", url, json=body) as resp:
                if resp.status_code != 200:
                    raw = await resp.aread()
                    yield (
                        "error",
                        {"message": f"HTTP {resp.status_code}: {raw.decode(errors='replace')[:500]}"},
                    )
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    if delta.get("content"):
                        yield ("text_delta", {"text": delta["content"], "thought_signature": None})

                    for tc in delta.get("tool_calls", []) or []:
                        idx = tc.get("index", 0)
                        slot = tool_calls.setdefault(idx, {"name": "", "arguments": ""})
                        fn = tc.get("function", {}) or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]

                    if choices[0].get("finish_reason") == "tool_calls" and tool_calls:
                        for slot in tool_calls.values():
                            try:
                                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
                            except json.JSONDecodeError:
                                args = {}
                            yield (
                                "function_call",
                                {"name": slot["name"], "args": args, "thought_signature": None},
                            )
                        tool_calls.clear()

        yield ("done", {})
    except httpx.RequestError as e:
        yield ("error", {"message": f"Local server connection error: {e}"})
