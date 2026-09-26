"""
Simple config storage: API key (separate 600-perm file) plus
config.json holding the custom system prompt / user prompt.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "eve"
KEY_FILE = CONFIG_DIR / "gemini.key"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_SYSTEM_PROMPT = (
    "You are EVE, a direct and concise agent with full control over the "
    "terminal of a Linux distribution. "
    "You have access to run_shell_command to execute shell commands "
    "whenever the user asks or whenever it's the fastest way to answer. "
    "Commands run automatically without asking the user for confirmation, "
    "so be careful, but don't hesitate to use them when it makes sense. "
    "Keep replies short and to the point.\n\n"
    "Reference paths and conventions for this machine:\n"
    "- Home directory: `~` (`$HOME`)\n"
    "- User config files: `~/.config/`\n"
    "- User-local binaries/scripts: `~/.local/bin/`\n"
    "- User-local installed apps/libraries: `~/.local/share/`\n"
    "- Temporary files: `/tmp/`\n"
    "- System-wide config: `/etc/`\n"
    "- System-wide binaries: `/usr/bin/` and `/usr/local/bin/`\n"
    "- Logs: `/var/log/` (use `journalctl` on systemd distros)\n"
    "- Installed package management: detect the distro first (check "
    "`/etc/os-release`) and use the matching tool — `pacman` (Arch), "
    "`apt` (Debian/Ubuntu), `dnf` (Fedora), etc.\n"
    "Prefer these conventional locations when creating, editing, or "
    "looking for files, but always verify with a command (e.g. `ls`, "
    "`find`, `which`) rather than assuming a path exists.\n\n"
    "Your replies are rendered as real Markdown, so always format using it "
    "instead of writing plain text — the same way you'd see Markdown used "
    "in any modern chat assistant:\n"
    "- Wrap all code, commands, file paths, and output in fenced code "
    "blocks with a language tag, e.g. ```bash\\nls -la\\n```. Use ```text "
    "for plain output with no language.\n"
    "- Use `backticks` for short inline code, filenames, or commands "
    "mentioned mid-sentence.\n"
    "- Use **bold** for key terms or warnings, and *italics* for emphasis.\n"
    "- Use `-` bullet lists or `1.` numbered lists for steps or options.\n"
    "- Use `#`/`##` headings only for longer, multi-section answers — skip "
    "them for short replies.\n"
    "Never describe formatting in words (e.g. don't say 'in bold' or 'in a "
    "code block') — just use the actual Markdown syntax so it renders."
)
DEFAULT_USER_PROMPT = (
    "The user is a Linux user comfortable with the terminal. "
    "Assume a standard, up-to-date Linux desktop setup unless told "
    "otherwise, and ask before making assumptions about their specific "
    "distro, shell, or desktop environment if a command depends on it."
)

# Matrix rain tuning — all persisted alongside the prompts.
DEFAULT_MATRIX_ENABLED = True   # whether the background effect is on
DEFAULT_MATRIX_EFFECT = "dna"  # "matrix" (digital rain) or "dna" (rotating helix)
DEFAULT_MATRIX_DENSITY = 0.55   # fraction of eligible columns raining at once (0.1-1.0)
DEFAULT_MATRIX_WIDTH = 1        # 1 = mostly thin lines, 2 = mostly thick lines (1-2)
DEFAULT_MATRIX_MIN_LEN = 4      # shortest streak length, in rows
DEFAULT_MATRIX_MAX_LEN_DIV = 3  # longest streak length = rows // this (smaller = longer)
DEFAULT_DNA_SPEED = 0.12        # helix rotation speed, radians per tick (0.02-0.4)
DEFAULT_DNA_WIDTH = 1           # strand thickness in columns (1-3)
DEFAULT_MODEL_MODE = "online"   # "online" (Gemini API) or "local" (local model)

# Local model (llama.cpp llama-server) settings.
DEFAULT_LOCAL_MODEL_PATH = ""          # e.g. /home/user/Modelos/Qwen3.5-4B-IQ4_XS.gguf
DEFAULT_LOCAL_SERVER_BIN = "llama-server"  # name on PATH, or full path to the binary
DEFAULT_LOCAL_PORT = 8080
DEFAULT_LOCAL_CTX = 8192
DEFAULT_LOCAL_NGL = 999                # GPU layers to offload; 999 = "all of them"


def load_api_key() -> str | None:
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key:
        return env_key
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip() or None
    return None


def save_api_key(key: str) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key.strip())
    KEY_FILE.chmod(0o600)


def load_settings() -> dict:
    settings = {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "user_prompt": DEFAULT_USER_PROMPT,
        "matrix_enabled": DEFAULT_MATRIX_ENABLED,
        "matrix_effect": DEFAULT_MATRIX_EFFECT,
        "matrix_density": DEFAULT_MATRIX_DENSITY,
        "matrix_width": DEFAULT_MATRIX_WIDTH,
        "matrix_min_len": DEFAULT_MATRIX_MIN_LEN,
        "matrix_max_len_div": DEFAULT_MATRIX_MAX_LEN_DIV,
        "dna_speed": DEFAULT_DNA_SPEED,
        "dna_width": DEFAULT_DNA_WIDTH,
        "model_mode": DEFAULT_MODEL_MODE,
        "local_model_path": DEFAULT_LOCAL_MODEL_PATH,
        "local_server_bin": DEFAULT_LOCAL_SERVER_BIN,
        "local_port": DEFAULT_LOCAL_PORT,
        "local_ctx": DEFAULT_LOCAL_CTX,
        "local_ngl": DEFAULT_LOCAL_NGL,
    }
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            settings.update({k: v for k, v in data.items() if k in settings})
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save_model_mode(model_mode: str) -> None:
    """Standalone setter for the online/local toggle, kept separate from
    save_settings() so the header button can flip modes instantly without
    needing to round-trip every other settings field."""
    save_local_config(model_mode=model_mode)


def _read_config_raw() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_local_config(**fields) -> None:
    """Standalone setter for local-model fields (local_model_path,
    local_server_bin, local_port, local_ctx, local_ngl), same pattern as
    save_model_mode: read-modify-write so it doesn't disturb prompts or
    matrix settings. Pass only the keys you want to change."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = _read_config_raw()
    data.update(fields)
    CONFIG_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def save_settings(
    system_prompt: str,
    user_prompt: str,
    matrix_enabled: bool = DEFAULT_MATRIX_ENABLED,
    matrix_effect: str = DEFAULT_MATRIX_EFFECT,
    matrix_density: float = DEFAULT_MATRIX_DENSITY,
    matrix_width: int = DEFAULT_MATRIX_WIDTH,
    matrix_min_len: int = DEFAULT_MATRIX_MIN_LEN,
    matrix_max_len_div: int = DEFAULT_MATRIX_MAX_LEN_DIV,
    dna_speed: float = DEFAULT_DNA_SPEED,
    dna_width: int = DEFAULT_DNA_WIDTH,
) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Preserve model_mode and local_* fields, since they're set
    # independently (header toggle / /localmodel etc.) and aren't
    # parameters of this function.
    existing = _read_config_raw()
    preserved = {
        "model_mode": existing.get("model_mode", DEFAULT_MODEL_MODE),
        "local_model_path": existing.get("local_model_path", DEFAULT_LOCAL_MODEL_PATH),
        "local_server_bin": existing.get("local_server_bin", DEFAULT_LOCAL_SERVER_BIN),
        "local_port": existing.get("local_port", DEFAULT_LOCAL_PORT),
        "local_ctx": existing.get("local_ctx", DEFAULT_LOCAL_CTX),
        "local_ngl": existing.get("local_ngl", DEFAULT_LOCAL_NGL),
    }
    CONFIG_FILE.write_text(
        json.dumps(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "matrix_enabled": matrix_enabled,
                "matrix_effect": matrix_effect,
                "matrix_density": matrix_density,
                "matrix_width": matrix_width,
                "matrix_min_len": matrix_min_len,
                "matrix_max_len_div": matrix_max_len_div,
                "dna_speed": dna_speed,
                "dna_width": dna_width,
                **preserved,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
