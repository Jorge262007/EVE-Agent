"""
Shell command execution with no confirmation prompt, guarded only by a
minimal blacklist of catastrophic, irreversible patterns. Everything
else runs freely.
"""
from __future__ import annotations

import asyncio
import re

# Only the worst of the worst: irreversible data or system loss.
# NOTE: sudo is intentionally NOT blocked — the user explicitly wants
# elevated commands (e.g. `sudo pacman -S ...`) to run without a manual
# confirmation step. Everything below still applies even with sudo in
# front of it (e.g. `sudo rm -rf /` is still caught by the rm pattern).
CATASTROPHIC_PATTERNS = [
    r"\brm\s+(-\w*\s+)*-[rf]\w*\s+/\s*($|[;&|])",   # rm -rf / (o -fr, con espacios)
    r"\brm\s+(-\w*\s+)*-[rf]\w*\s+/\*",              # rm -rf /*
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\s+.*\bof=/dev/sd[a-z]\b",
    r"\bdd\s+.*\bof=/dev/nvme\d+n\d+\b",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",     # fork bomb :(){ :|:& };:
    r"\bwipefs\b",
    r"\bshred\b.*/dev/",
    r">\s*/dev/sd[a-z]\b",
    r"\bparted\b.*\b(rm|mklabel)\b",
    r"\bmdadm\b.*--zero-superblock",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in CATASTROPHIC_PATTERNS]


def is_blacklisted(command: str) -> str | None:
    """Returns the matching pattern description if blocked, else None."""
    for pattern in _COMPILED:
        if pattern.search(command):
            return pattern.pattern
    return None


async def run_command(command: str, timeout: float = 30.0) -> tuple[int, str]:
    """Runs a shell command, returns (exit_code, combined_output)."""
    blocked = is_blacklisted(command)
    if blocked:
        return (126, f"[blocked] Command matches catastrophic pattern: {blocked}")

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return (124, f"[timeout tras {timeout}s]")
        output = stdout.decode(errors="replace")
        return (proc.returncode or 0, output[-4000:])  # cap output size
    except Exception as e:
        return (1, f"[error running command] {e}")
