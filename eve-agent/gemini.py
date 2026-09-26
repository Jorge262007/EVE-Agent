"""
Minimal Gemini client with SSE streaming + function calling
for shell command execution. No heavy dependencies: httpx + json.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

import httpx

from config import load_api_key, save_api_key  # re-exported for convenience

MODEL = "gemini-3.5-flash-lite"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"

SHELL_TOOL = {
    "function_declarations": [
        {
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
        }
    ]
}

def load_api_key_legacy() -> Optional[str]:
    """Kept for backwards-compat import paths; use config.load_api_key instead."""
    return load_api_key()


@dataclass
class Message:
    role: str  # "user" | "model" | "function"
    text: str = ""
    function_call: Optional[dict] = None  # {"name": ..., "args": {...}, "thought_signature": "..."}
    function_response: Optional[dict] = None
    thought_signature: Optional[str] = None  # signature attached to the text part, if any


@dataclass
class ChatSession:
    messages: list[Message] = field(default_factory=list)

    def to_api_contents(self) -> list[dict]:
        contents = []
        for m in self.messages:
            if m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.text}]})
            elif m.role == "model":
                parts = []
                if m.text:
                    text_part = {"text": m.text}
                    if m.thought_signature:
                        text_part["thought_signature"] = m.thought_signature
                    parts.append(text_part)
                if m.function_call:
                    fc = {
                        "function_call": {
                            "name": m.function_call["name"],
                            "args": m.function_call["args"],
                        }
                    }
                    sig = m.function_call.get("thought_signature")
                    if sig:
                        fc["thought_signature"] = sig
                    parts.append(fc)
                contents.append({"role": "model", "parts": parts})
            elif m.role == "function":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "function_response": {
                                    "name": m.function_response["name"],
                                    "response": {"result": m.function_response["result"]},
                                }
                            }
                        ],
                    }
                )
        return contents


class GeminiError(Exception):
    pass


async def stream_reply(
    session: ChatSession, api_key: str, system_prompt: str
) -> AsyncIterator[tuple[str, dict]]:
    """
    Yields tuples of (kind, payload):
      ("text_delta", {"text": "..."})       -> streamed text chunk
      ("function_call", {"name":..,"args":..}) -> model wants to call a tool
      ("done", {})                           -> stream finished cleanly
      ("error", {"message": "..."})          -> something went wrong
    """
    url = f"{API_BASE}/models/{MODEL}:streamGenerateContent?alt=sse&key={api_key}"
    body = {
        "contents": session.to_api_contents(),
        "tools": [SHELL_TOOL],
        "system_instruction": {"parts": [{"text": system_prompt}]},
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
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
                    if not data:
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    candidates = chunk.get("candidates") or []
                    if not candidates:
                        continue
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        sig = part.get("thoughtSignature") or part.get("thought_signature")
                        if "text" in part:
                            yield ("text_delta", {"text": part["text"], "thought_signature": sig})
                        if "functionCall" in part:
                            fc = part["functionCall"]
                            yield (
                                "function_call",
                                {
                                    "name": fc.get("name", ""),
                                    "args": fc.get("args", {}),
                                    "thought_signature": sig,
                                },
                            )
        yield ("done", {})
    except httpx.RequestError as e:
        yield ("error", {"message": f"Network error: {e}"})
