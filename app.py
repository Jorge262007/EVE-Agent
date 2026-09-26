"""
EVE — terminal agent powered by Gemini + shell execution
with no confirmation prompt (minimal blacklist). btop-style neon look.

Usage:
    python app.py
    /help                       -> shows all available commands (hinted in the input placeholder)
    /key YOUR_GEMINI_API_KEY    -> saves the key (only needed once)
    /localmodel /path/to.gguf   -> sets the local model path (for Local mode)
    /settings                   -> System/User prompt screen
    /new                        -> clears the chat
    /quit  or Ctrl+C            -> exit
"""
from __future__ import annotations

import asyncio
import math
import random
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.strip import Strip
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Collapsible, Input, Markdown, Static, TextArea
from textual.screen import ModalScreen
from rich.segment import Segment
from rich.style import Style

from config import (
    DEFAULT_DNA_SPEED,
    DEFAULT_DNA_WIDTH,
    DEFAULT_MATRIX_DENSITY,
    DEFAULT_MATRIX_EFFECT,
    DEFAULT_MATRIX_ENABLED,
    DEFAULT_MATRIX_MAX_LEN_DIV,
    DEFAULT_MATRIX_MIN_LEN,
    DEFAULT_MATRIX_WIDTH,
    load_api_key,
    load_settings,
    save_api_key,
    save_local_config,
    save_model_mode,
    save_settings,
)
from gemini import ChatSession, Message, stream_reply
import local_llm
from shell_exec import run_command


# ======================================================================
# Settings screen
# ======================================================================

MATRIX_CHARS = "ｦｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ0123456789"


# Column spacing: only every _COL_STEP-th column position is even eligible
# to rain, independent of the density setting (which picks a fraction of
# those eligible slots).
_COL_STEP = 3

_STYLE_HEAD = Style(color="#eafcff", bold=True)
_STYLE_BODY = Style(color="#3ee6ff")
_STYLE_BLANK = Style()
_BLANK_SEGMENT = Segment(" ", _STYLE_BLANK)


class _Streak:
    """One falling column of characters. A streak is just a head position
    plus a fixed-length list of glyphs generated once at spawn time —
    advancing it each tick is a single float add, and drawing it is a
    single slice, not a per-row object walk. 'width' makes each streak
    occupy 1-2 adjacent columns so the falling line reads as thicker,
    without touching how long (how many rows) it falls."""

    __slots__ = ("col", "head", "length", "chars", "speed", "width")

    def __init__(self, col: int, rows: int, min_len: int, max_len_div: int, width_bias: int) -> None:
        self.col = col
        self.length = _rand_length(rows, min_len, max_len_div)
        self.chars = [random.choice(MATRIX_CHARS) for _ in range(self.length)]
        self.speed = random.uniform(0.9, 1.6)
        self.width = _rand_width(width_bias)
        # Start above the screen by a random amount so streaks don't all
        # enter in lockstep.
        self.head = random.uniform(-rows, 0)

    def respawn(self, rows: int, min_len: int, max_len_div: int, width_bias: int) -> None:
        self.length = _rand_length(rows, min_len, max_len_div)
        self.chars = [random.choice(MATRIX_CHARS) for _ in range(self.length)]
        self.speed = random.uniform(0.9, 1.6)
        self.width = _rand_width(width_bias)
        self.head = random.uniform(-6, 0)


def _rand_length(rows: int, min_len: int, max_len_div: int) -> int:
    max_len = max(min_len + 2, rows // max(max_len_div, 1))
    return random.randint(min_len, max_len)


def _rand_width(width_bias: int) -> int:
    # width_bias 1 -> mostly 1-wide with occasional 2-wide.
    # width_bias 2 -> mostly 2-wide with occasional 3-wide.
    if width_bias <= 1:
        return random.choice((1, 1, 1, 2))
    return random.choice((2, 2, 2, 3))


class MatrixRain(Widget):
    """Lightweight digital rain: a small, fixed set of independent falling
    streaks (head position + a short precomputed glyph list per streak)
    rather than a per-cell node/eraser simulation. Only a fraction of
    columns are ever active, keeping the number of visible falling lines
    low on purpose.

    Why this is a Widget with render_line() instead of a Static updated via
    Content.from_markup(): the earlier version paid for a full markup-string
    parse ("[style]char[/style]" tokenizing) on every tick just to produce
    colored text, then handed that to Static.update(), which rebuilds the
    widget's render cache and invalidates layout on every call. curses-based
    unimatrix never pays any of that — it writes a glyph straight to a
    screen cell. render_line() is the Textual equivalent: Textual asks for
    a row only when it's about to paint it, and we hand back a Strip of
    Segments (Rich's own primitive, no text parsing involved) built directly
    from the streak data for that row. refresh() then just marks the widget
    dirty instead of rebuilding a cache eagerly. Each tick's actual work is
    still O(active streaks), not O(rows*cols) — only a fraction of columns
    are ever active by design (_ACTIVE_FRACTION), keeping the streak count,
    and therefore the segment count per line, small. No painted background —
    only the falling glyphs render; everything else is left untouched so the
    terminal's own background shows through."""

    def __init__(
        self,
        enabled: bool = DEFAULT_MATRIX_ENABLED,
        density: float = DEFAULT_MATRIX_DENSITY,
        width_bias: int = DEFAULT_MATRIX_WIDTH,
        min_len: int = DEFAULT_MATRIX_MIN_LEN,
        max_len_div: int = DEFAULT_MATRIX_MAX_LEN_DIV,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._cols = 0
        self._rows = 0
        self._streaks: list[_Streak] = []
        self._timer = None
        self._blank_strip = Strip.blank(0)
        self.enabled = enabled
        self.density = density
        self.width_bias = width_bias
        self.min_len = min_len
        self.max_len_div = max_len_div

    def set_enabled(self, enabled: bool) -> None:
        """Fully stop or start the effect. Disabling stops all ticking,
        refresh(), and render_line() calls for this widget — so an off
        rain strip costs exactly nothing, not just "less". Each widget
        still owns whether it's enabled; the actual timer driving ticks
        lives on the App so both strips share a single interval instead
        of paying for two independent timers/compositor passes."""
        if enabled == self.enabled:
            return
        self.enabled = enabled
        if enabled and self._cols and self._rows:
            self._spawn_streaks()
        elif not enabled:
            self._streaks = []
        self.refresh()

    def apply_settings(self, density: float, width_bias: int, min_len: int, max_len_div: int) -> None:
        """Update tuning live and re-spawn streaks so the new density/width/
        length take effect immediately, without waiting for a resize."""
        self.density = density
        self.width_bias = width_bias
        self.min_len = min_len
        self.max_len_div = max_len_div
        if self.enabled and self._cols and self._rows:
            self._spawn_streaks()

    def on_resize(self, event) -> None:
        self._cols = max(event.size.width, 1)
        self._rows = max(event.size.height, 1)
        if self.enabled:
            self._spawn_streaks()

    def _spawn_streaks(self) -> None:
        eligible = list(range(0, self._cols, _COL_STEP))
        n_active = max(1, int(len(eligible) * self.density))
        chosen = random.sample(eligible, min(n_active, len(eligible)))
        # Columns never change after spawn, so sorting once here means
        # render_line() never needs to sort per-row, per-tick.
        chosen.sort()
        self._streaks = [
            _Streak(col, self._rows, self.min_len, self.max_len_div, self.width_bias)
            for col in chosen
        ]
        self._blank_strip = Strip([Segment(" " * self._cols, _STYLE_BLANK)], self._cols)

    def _tick(self) -> None:
        if not self.enabled or not self._cols or not self._rows:
            return

        rows = self._rows
        for streak in self._streaks:
            streak.head += streak.speed
            # Once the whole streak has fallen off the bottom, respawn it
            # from just above the top — one object reused forever instead
            # of spawning/expiring separate node objects.
            if streak.head - streak.length > rows:
                streak.respawn(rows, self.min_len, self.max_len_div, self.width_bias)

        # Just mark dirty; Textual pulls rows back out via render_line()
        # only for what actually needs painting, instead of us eagerly
        # rebuilding a whole renderable up front like Content.from_markup
        # forced us to.
        self.refresh()

    def render_line(self, y: int) -> Strip:
        cols = self._cols
        if not cols or not self.enabled:
            return Strip.blank(cols)

        # Collect (col, char, style, width) hits for this one row directly
        # from each streak's short glyph list — no full-grid scan, no
        # per-tick allocation of a rows-long list of lists. Streaks are
        # kept sorted by column since spawn, so hits come out already
        # sorted — no per-row, per-tick sort() call needed.
        hits: list[tuple[int, str, Style, int]] = []
        for streak in self._streaks:
            offset = int(streak.head) - y
            if 0 <= offset < streak.length:
                style = _STYLE_HEAD if offset == 0 else _STYLE_BODY
                hits.append((streak.col, streak.chars[offset], style, streak.width))

        if not hits:
            return self._blank_strip

        segments: list[Segment] = []
        pos = 0
        for col, ch, style, width in hits:
            if col < pos:
                continue  # overlapped by a wider streak just drawn
            if col > pos:
                segments.append(Segment(" " * (col - pos), _STYLE_BLANK))
            span = min(width, cols - col)
            segments.append(Segment(ch * span, style))
            pos = col + span
        if pos < cols:
            segments.append(Segment(" " * (cols - pos), _STYLE_BLANK))
        return Strip(segments, cols)


_DNA_STYLE_A = Style(color="#5ec8ff", bold=True)   # front strand, brighter blue
_DNA_STYLE_B = Style(color="#1f5f8f")              # back strand, dimmer blue (depth cue)
_DNA_STYLE_RUNG = Style(color="#123a57")           # base-pair rungs connecting strands


class DnaHelix(Widget):
    """Vertical double-helix effect: two strands run top-to-bottom along
    the strip, each strand's horizontal position oscillating side to side
    as a sine wave of the row index, 180 degrees out of phase from each
    other — the classic double-helix silhouette, rotating as it descends.
    A rung (base pair) is drawn on rows where the strands cross.

    Cost profile mirrors MatrixRain on purpose: per-row strand columns are
    precomputed once per _tick() into a flat list indexed by row, and
    render_line() only ever indexes into that precomputed list — no trig,
    no per-row math, so painting is still O(rows) per tick and O(1) per
    render_line() call."""

    def __init__(
        self,
        enabled: bool = DEFAULT_MATRIX_ENABLED,
        speed: float = DEFAULT_DNA_SPEED,
        width: int = DEFAULT_DNA_WIDTH,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._cols = 0
        self._rows = 0
        self.enabled = enabled
        self.speed = speed
        self.width = width
        self._phase = 0.0
        # Precomputed per-row state, refreshed each tick: for every row,
        # which column each strand sits in, and whether that row is a rung.
        self._row_a: list[int] = []       # strand A column per row
        self._row_b: list[int] = []       # strand B column per row
        self._row_rung: list[bool] = []   # whether strands cross on this row
        self._blank_strip = Strip.blank(0)

    def set_enabled(self, enabled: bool) -> None:
        if enabled == self.enabled:
            return
        self.enabled = enabled
        if not enabled:
            self._row_a = []
            self._row_b = []
            self._row_rung = []
        self.refresh()

    def on_resize(self, event) -> None:
        self._cols = max(event.size.width, 1)
        self._rows = max(event.size.height, 1)
        self._blank_strip = Strip([Segment(" " * self._cols, _STYLE_BLANK)], self._cols)
        if self.enabled:
            self._recompute()

    def _recompute(self) -> None:
        """One pass over rows per tick: derive each strand's horizontal
        column from a sine wave in the current phase, spaced so roughly
        one full turn of the helix is visible across the strip's height
        at once. The strand oscillates between the strip's left and right
        edges so the helix fills the whole narrow column."""
        cols = self._cols
        rows = self._rows
        if not cols or not rows:
            return
        # Amplitude: how far the strand swings from center. Keep a small
        # margin so the strand never touches the very edge column.
        center = (cols - 1) / 2
        amplitude = max(0.0, center - 0.5)
        turn_height = max(rows, 6)
        row_a = [0] * rows
        row_b = [0] * rows
        rung = [False] * rows
        for y in range(rows):
            angle = self._phase + (y / turn_height) * 2 * math.pi
            a = math.sin(angle)
            b = math.sin(angle + math.pi)
            col_a = round(center + a * amplitude)
            col_b = round(center + b * amplitude)
            row_a[y] = col_a
            row_b[y] = col_b
            # A rung (base pair) is drawn where the two strands nearly
            # cross — i.e. where they're at roughly the same column —
            # which happens twice per turn, just like real base pairs.
            rung[y] = abs(col_a - col_b) <= 1
        self._row_a = row_a
        self._row_b = row_b
        self._row_rung = rung

    def apply_settings(self, speed: float, width: int) -> None:
        """Update rotation speed and strand thickness live; no re-spawn
        needed since state is fully recomputed from phase every tick."""
        self.speed = speed
        self.width = width

    def _tick(self) -> None:
        if not self.enabled or not self._cols or not self._rows:
            return
        self._phase += self.speed  # rotation speed
        if self._phase > 2 * math.pi:
            self._phase -= 2 * math.pi
        self._recompute()
        self.refresh()

    def render_line(self, y: int) -> Strip:
        cols = self._cols
        if not cols or not self._rows or not self.enabled or y >= len(self._row_a):
            return Strip.blank(cols)

        col_a = self._row_a[y]
        col_b = self._row_b[y]
        is_rung = self._row_rung[y]

        w = max(1, self.width)
        half_lo = (w - 1) // 2
        half_hi = w // 2
        a_lo, a_hi = col_a - half_lo, col_a + half_hi
        b_lo, b_hi = col_b - half_lo, col_b + half_hi

        segments: list[Segment] = []
        pos = 0
        # Rung rows draw a connecting line between the two strand blocks
        # in addition to the strand blocks themselves, so base pairs read
        # as actual crossbars rather than just two coincident dots.
        lo, hi = (a_hi, b_lo) if col_a <= col_b else (b_hi, a_lo)
        for x in range(cols):
            if a_lo <= x <= a_hi:
                ch, style = "█", _DNA_STYLE_A
            elif b_lo <= x <= b_hi:
                ch, style = "█", _DNA_STYLE_B
            elif is_rung and lo < x < hi:
                ch, style = "─", _DNA_STYLE_RUNG
            else:
                continue
            if x > pos:
                segments.append(Segment(" " * (x - pos), _STYLE_BLANK))
            segments.append(Segment(ch, style))
            pos = x + 1
        if not segments:
            return self._blank_strip
        if pos < cols:
            segments.append(Segment(" " * (cols - pos), _STYLE_BLANK))
        return Strip(segments, cols)


class SettingsScreen(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss_screen", "Close")]

    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        matrix_enabled: bool,
        matrix_effect: str,
        matrix_density: float,
        matrix_width: int,
        matrix_min_len: int,
        matrix_max_len_div: int,
        dna_speed: float,
        dna_width: int,
    ) -> None:
        super().__init__()
        self._system_prompt = system_prompt
        self._user_prompt = user_prompt
        self._matrix_enabled = matrix_enabled
        self._matrix_effect = matrix_effect
        self._matrix_density = matrix_density
        self._matrix_width = matrix_width
        self._matrix_min_len = matrix_min_len
        self._matrix_max_len_div = matrix_max_len_div
        self._dna_speed = dna_speed
        self._dna_width = dna_width

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-panel") as panel:
            panel.border_title = "settings"
            yield Static("System prompt", classes="settings-label")
            yield TextArea(self._system_prompt, id="system-prompt-input")
            yield Static("User prompt (info about you, optional)", classes="settings-label")
            yield TextArea(self._user_prompt, id="user-prompt-input")

            yield Static("Background animation", classes="settings-label")
            yield Checkbox(
                "Enable background animation (turn it off if it's running slow)",
                value=self._matrix_enabled,
                id="matrix-enabled-checkbox",
            )

            with Collapsible(title="Matrix", collapsed=True, id="matrix-collapsible"):
                yield Checkbox(
                    "Use Matrix as the active effect",
                    value=self._matrix_effect == "matrix",
                    id="effect-matrix-checkbox",
                )
                with Horizontal(id="matrix-settings-row"):
                    with Vertical(classes="matrix-field"):
                        yield Static("Density (0.1-1.0)", classes="settings-sublabel")
                        yield Input(
                            value=str(self._matrix_density),
                            id="matrix-density-input",
                            type="number",
                        )
                    with Vertical(classes="matrix-field"):
                        yield Static("Thickness (1-2)", classes="settings-sublabel")
                        yield Input(
                            value=str(self._matrix_width),
                            id="matrix-width-input",
                            type="integer",
                        )
                    with Vertical(classes="matrix-field"):
                        yield Static("Min length", classes="settings-sublabel")
                        yield Input(
                            value=str(self._matrix_min_len),
                            id="matrix-minlen-input",
                            type="integer",
                        )
                    with Vertical(classes="matrix-field"):
                        yield Static("Max length (÷ height)", classes="settings-sublabel")
                        yield Input(
                            value=str(self._matrix_max_len_div),
                            id="matrix-maxlendiv-input",
                            type="integer",
                        )

            with Collapsible(title="DNA", collapsed=True, id="dna-collapsible"):
                yield Checkbox(
                    "Use DNA as the active effect",
                    value=self._matrix_effect == "dna",
                    id="effect-dna-checkbox",
                )
                with Horizontal(id="dna-settings-row"):
                    with Vertical(classes="matrix-field"):
                        yield Static("Rotation speed (0.02-0.4)", classes="settings-sublabel")
                        yield Input(
                            value=str(self._dna_speed),
                            id="dna-speed-input",
                            type="number",
                        )
                    with Vertical(classes="matrix-field"):
                        yield Static("Thickness (1-3)", classes="settings-sublabel")
                        yield Input(
                            value=str(self._dna_width),
                            id="dna-width-input",
                            type="integer",
                        )

            with Horizontal(id="settings-buttons"):
                yield Button("Save", id="save-settings")
                yield Button("Cancel", id="cancel-settings")

    def on_mount(self) -> None:
        self.query_one("#system-prompt-input", TextArea).focus()

    def action_dismiss_screen(self) -> None:
        self.dismiss(None)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        # The two "use this effect" checkboxes act like a radio group:
        # checking one unchecks the other, and expands/collapses the
        # matching panel so the active effect's settings are the ones
        # left open, VS Code explorer-style.
        if event.checkbox.id == "effect-matrix-checkbox" and event.value:
            self.query_one("#effect-dna-checkbox", Checkbox).value = False
            self.query_one("#matrix-collapsible", Collapsible).collapsed = False
            self.query_one("#dna-collapsible", Collapsible).collapsed = True
        elif event.checkbox.id == "effect-dna-checkbox" and event.value:
            self.query_one("#effect-matrix-checkbox", Checkbox).value = False
            self.query_one("#dna-collapsible", Collapsible).collapsed = False
            self.query_one("#matrix-collapsible", Collapsible).collapsed = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-settings":
            system_prompt = self.query_one("#system-prompt-input", TextArea).text
            user_prompt = self.query_one("#user-prompt-input", TextArea).text
            enabled = self.query_one("#matrix-enabled-checkbox", Checkbox).value
            effect = "dna" if self.query_one("#effect-dna-checkbox", Checkbox).value else "matrix"

            density = _clamp_float(
                self.query_one("#matrix-density-input", Input).value, 0.1, 1.0, DEFAULT_MATRIX_DENSITY
            )
            width = _clamp_int(
                self.query_one("#matrix-width-input", Input).value, 1, 2, DEFAULT_MATRIX_WIDTH
            )
            min_len = _clamp_int(
                self.query_one("#matrix-minlen-input", Input).value, 1, 50, DEFAULT_MATRIX_MIN_LEN
            )
            max_len_div = _clamp_int(
                self.query_one("#matrix-maxlendiv-input", Input).value, 1, 20, DEFAULT_MATRIX_MAX_LEN_DIV
            )
            dna_speed = _clamp_float(
                self.query_one("#dna-speed-input", Input).value, 0.02, 0.4, DEFAULT_DNA_SPEED
            )
            dna_width = _clamp_int(
                self.query_one("#dna-width-input", Input).value, 1, 3, DEFAULT_DNA_WIDTH
            )

            save_settings(
                system_prompt, user_prompt, enabled, effect, density, width, min_len, max_len_div,
                dna_speed, dna_width,
            )
            app = self.app
            assert isinstance(app, EveApp)
            app.system_prompt = system_prompt
            app.user_prompt = user_prompt
            app.matrix_enabled = enabled
            app.matrix_effect = effect
            app.matrix_density = density
            app.matrix_width = width
            app.matrix_min_len = min_len
            app.matrix_max_len_div = max_len_div
            app.dna_speed = dna_speed
            app.dna_width = dna_width
            app.apply_matrix_settings()
            self.dismiss(None)
        else:
            self.dismiss(None)


def _clamp_float(raw: str, lo: float, hi: float, default: float) -> float:
    try:
        return min(hi, max(lo, float(raw)))
    except (TypeError, ValueError):
        return default


def _clamp_int(raw: str, lo: int, hi: int, default: int) -> int:
    try:
        return min(hi, max(lo, int(float(raw))))
    except (TypeError, ValueError):
        return default


# ======================================================================
# Main app
# ======================================================================

class EveApp(App):
    CSS_PATH = "eve.tcss"
    TITLE = "EVE"
    ENABLE_COMMAND_PALETTE = False  # no themes/palette that actually do anything here
    BINDINGS = [
        ("ctrl+n", "new_chat", "New chat"),
        ("ctrl+s", "open_settings", "Settings"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.session = ChatSession()
        self.api_key: str | None = load_api_key()
        settings = load_settings()
        self.system_prompt: str = settings["system_prompt"]
        self.user_prompt: str = settings["user_prompt"]
        self.matrix_enabled: bool = settings["matrix_enabled"]
        self.matrix_effect: str = settings["matrix_effect"]
        self.matrix_density: float = settings["matrix_density"]
        self.matrix_width: int = settings["matrix_width"]
        self.matrix_min_len: int = settings["matrix_min_len"]
        self.matrix_max_len_div: int = settings["matrix_max_len_div"]
        self.dna_speed: float = settings["dna_speed"]
        self.dna_width: int = settings["dna_width"]
        self.model_mode: str = settings["model_mode"]  # "online" or "local"
        self.local_model_path: str = settings["local_model_path"]
        self.local_server_bin: str = settings["local_server_bin"]
        self.local_port: int = settings["local_port"]
        self.local_ctx: int = settings["local_ctx"]
        self.local_ngl: int = settings["local_ngl"]
        self.is_loading = False

    def _make_effect_pair(self, side: str) -> tuple[MatrixRain, DnaHelix]:
        """Build both effect widgets for one side (left/right). Both are
        always mounted; only the one matching self.matrix_effect is ever
        enabled/visible, and switching effects in Settings just flips
        which of the pair is on — no dynamic mount/unmount needed."""
        is_matrix = self.matrix_effect == "matrix"
        rain = MatrixRain(
            id=f"matrix-{side}",
            classes="matrix-strip",
            enabled=self.matrix_enabled and is_matrix,
            density=self.matrix_density,
            width_bias=self.matrix_width,
            min_len=self.matrix_min_len,
            max_len_div=self.matrix_max_len_div,
        )
        dna = DnaHelix(
            id=f"dna-{side}",
            classes="matrix-strip",
            enabled=self.matrix_enabled and not is_matrix,
            speed=self.dna_speed,
            width=self.dna_width,
        )
        if is_matrix:
            dna.display = False
        else:
            rain.display = False
        return rain, dna

    def compose(self) -> ComposeResult:
        left_rain, left_dna = self._make_effect_pair("left")
        right_rain, right_dna = self._make_effect_pair("right")
        with Horizontal(id="root"):
            yield left_rain
            yield left_dna
            with Vertical(id="main-column"):
                with Horizontal(id="header") as header:
                    header.border_title = "menu"
                    yield Static(" ⬤ EVE " + self._header_status_text(), id="header-text")
                    yield Static("", classes="spacer")
                    yield Button(
                        "Local",
                        id="mode-local-btn",
                        classes="mode-segment active" if self.model_mode == "local" else "mode-segment",
                    )
                    yield Button(
                        "Online",
                        id="mode-cloud-btn",
                        classes="mode-segment active" if self.model_mode != "local" else "mode-segment",
                    )
                    yield Button("New chat", id="new-chat-btn", classes="header-btn")
                    yield Button("Settings", id="settings-btn", classes="header-btn")
                with Vertical(id="chat-panel") as panel:
                    panel.border_title = "chat"
                    yield VerticalScroll(id="chat-log")
                with Vertical(id="input-panel") as ipanel:
                    ipanel.border_title = "message"
                    with Horizontal(id="input-row"):
                        yield Input(placeholder="/help", id="input-field")
                        yield Button("Send", id="send-btn", variant="primary")
                yield Static("", id="statusbar")
            yield right_rain
            yield right_dna

    def on_mount(self) -> None:
        self.query_one("#input-field", Input).focus()
        self._set_status("")
        # Single shared timer driving both matrix strips, instead of each
        # MatrixRain owning its own interval — halves the number of timer
        # callbacks and refresh()/compositor passes per tick regardless of
        # how many strips are enabled.
        self.set_interval(0.09, self._tick_matrix)

    def _tick_matrix(self) -> None:
        self.query_one("#matrix-left", MatrixRain)._tick()
        self.query_one("#matrix-right", MatrixRain)._tick()
        self.query_one("#dna-left", DnaHelix)._tick()
        self.query_one("#dna-right", DnaHelix)._tick()

    def _header_status_text(self) -> str:
        if self.model_mode == "local":
            if self.local_model_path:
                return f"· local: {Path(self.local_model_path).name}"
            return "· NO LOCAL MODEL — use /localmodel <path to .gguf>"
        return "· ready" if self.api_key else " · NO GEMINI API KEY — use /key <your_gemini_api_key>"

    def _refresh_header(self) -> None:
        self.query_one("#header-text", Static).update(" ⬤ EVE " + self._header_status_text())

    def _set_mode(self, mode: str) -> None:
        self.model_mode = mode
        save_model_mode(mode)
        self.query_one("#mode-local-btn", Button).set_class(mode == "local", "active")
        self.query_one("#mode-cloud-btn", Button).set_class(mode != "local", "active")
        self._refresh_header()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-chat-btn":
            self.action_new_chat()
        elif event.button.id == "settings-btn":
            self.action_open_settings()
        elif event.button.id == "send-btn":
            self.action_submit_input()
        elif event.button.id == "mode-local-btn":
            self._set_mode("local")
        elif event.button.id == "mode-cloud-btn":
            self._set_mode("online")

    def action_submit_input(self) -> None:
        input_field = self.query_one("#input-field", Input)
        self.call_later(self.on_input_submitted, Input.Submitted(input_field, input_field.value))

    # ------------------------------------------------------------------
    # Effective system prompt (system + optional user info block)
    # ------------------------------------------------------------------

    def _effective_system_prompt(self) -> str:
        if self.user_prompt.strip():
            return f"{self.system_prompt}\n\nInformation about the user:\n{self.user_prompt}"
        return self.system_prompt

    # ------------------------------------------------------------------
    # UI helpers — chat-style bubbles (left/right, variable width).
    # Assistant/system/error bubbles render as Markdown (headings, code
    # blocks with syntax highlight, lists, etc). User bubbles stay plain
    # Static since they're just what you typed.
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self.query_one("#statusbar", Static).update(text)

    async def _append_turn_box(self, caption: str, side: str):
        """Creates the single outer box for a whole assistant turn: one
        bordered container ('turn-box') that holds, in order, any tool-call
        collapsibles and the final Markdown reply — all with no border of
        their own, so the whole turn reads as one box instead of a stack of
        separately-bordered pieces. Returns (col, box, reply_widget)."""
        log = self.query_one("#chat-log", VerticalScroll)

        col = Vertical(classes="bubble-col")
        caption_widget = Static(caption, classes="caption")
        spacer = Static("", classes="spacer")
        row = Horizontal(classes="msg-row")

        box = Vertical(classes="turn-box")
        reply = Markdown("", classes="turn-reply")

        await log.mount(row)
        if side == "right":
            await row.mount(spacer)
            await row.mount(col)
        else:
            await row.mount(col)
            await row.mount(spacer)

        await col.mount(caption_widget)
        await col.mount(box)
        await box.mount(reply)

        log.scroll_end(animate=False)
        return col, box, reply

    async def _append_bubble(self, caption: str, text: str, css_class: str, side: str, markdown: bool = False):
        """side: 'left' or 'right'. Returns the content widget for streaming updates."""
        _row, _col, content = await self._append_bubble_full(caption, text, css_class, side, markdown=markdown)
        return content

    async def _append_bubble_full(self, caption: str, text: str, css_class: str, side: str, markdown: bool = False):
        """Same as _append_bubble but also returns the row/col containers.
        Used for simple one-off bubbles (user/system/error messages) that
        don't need the multi-piece turn-box layout."""
        log = self.query_one("#chat-log", VerticalScroll)

        content: Static | Markdown
        if markdown:
            content = Markdown(text, classes=css_class)
        else:
            content = Static(text, classes=css_class, markup=True)

        col = Vertical(classes="bubble-col")
        caption_widget = Static(caption, classes="caption")
        spacer = Static("", classes="spacer")

        row = Horizontal(classes="msg-row")
        await log.mount(row)

        if side == "right":
            await row.mount(spacer)
            await row.mount(col)
        else:
            await row.mount(col)
            await row.mount(spacer)

        await col.mount(caption_widget)
        await col.mount(content)

        log.scroll_end(animate=False)
        return row, col, content

    async def _update_markdown(self, widget: Markdown, text: str) -> None:
        await widget.update(text)
        self.query_one("#chat-log", VerticalScroll).scroll_end(animate=False)

    async def _append_tool_block(self, box: Vertical, command: str) -> Collapsible:
        """Mounts a collapsed 'Terminal' block for a tool call INTO the
        turn's box, above the final reply. The title never shows the
        command — only 'Terminal' — the command itself (and its output)
        are revealed inside when the user expands it, VS Code style.
        Starts collapsed and empty; fill it in with _finish_tool_block once
        the command finishes running."""
        collapsible = Collapsible(title="Terminal", collapsed=True, classes="tool-collapsible")
        await box.mount(collapsible)
        self.query_one("#chat-log", VerticalScroll).scroll_end(animate=False)
        return collapsible

    async def _finish_tool_block(self, collapsible: Collapsible, command: str, exit_code: int, output: str) -> None:
        """Fills the collapsible with the command that ran and its output,
        both hidden until the user expands it. Only mentions the exit code
        when it's non-zero — a clean run doesn't need to announce
        '(exit 0)' every single time."""
        result = f"```\n{output}\n```" if output else "*(no output)*"
        if exit_code != 0:
            result = f"**exit {exit_code}**\n\n{result}"
        body = f"```bash\n{command}\n```\n{result}"
        contents = collapsible.query_one(Collapsible.Contents)
        await contents.mount(Markdown(body, classes="tool-output"))

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        # The "/help" placeholder is only meant to catch a new user's eye
        # once per chat — after the first message it switches to a plain
        # placeholder until /new brings the hint back.
        if event.input.placeholder == "/help":
            event.input.placeholder = "Type a message…"

        if text.startswith("/"):
            await self._handle_command(text)
            return

        if self.model_mode == "online" and not self.api_key:
            await self._append_bubble("system", "No API key configured. Use /key <your_api_key>.", "msg-error", "left")
            return
        if self.model_mode == "local" and not self.local_model_path:
            await self._append_bubble(
                "system", "No local model configured. Use /localmodel <path to .gguf>.", "msg-error", "left"
            )
            return

        if self.is_loading:
            return

        self.session.messages.append(Message(role="user", text=text))
        await self._append_bubble("you", text, "msg-user", "right")
        await self._run_turn()

    async def _handle_command(self, text: str) -> None:
        parts = text.split(maxsplit=2)
        cmd = parts[0].lower()

        if cmd in ("/help", "/?"):
            help_text = (
                "**Available commands**\n\n"
                "- `/key <your_gemini_api_key>` — save your Gemini API key "
                "(needed for Online mode; get one at "
                "[aistudio.google.com/apikey](https://aistudio.google.com/apikey))\n"
                "- `/localmodel <path>` — set the path to a local `.gguf` model "
                "(needed for Local mode)\n"
                "- `/localserver <path>` — set the `llama-server` binary/path "
                "(default: `llama-server` on PATH)\n"
                "- `/settings` — open the System prompt / User prompt screen "
                "(also `Ctrl+S`)\n"
                "- `/new` — clear the chat (also `Ctrl+N`)\n"
                "- `/help` — show this message again\n"
                "- `/quit` — exit (also `Ctrl+C`)\n\n"
                "Switch between **Online** (Gemini) and **Local** (your own "
                "model) with the buttons in the header."
            )
            await self._append_bubble("system", help_text, "msg-system", "left", markdown=True)

        elif cmd == "/key" and len(parts) >= 2:
            key = parts[1]
            save_api_key(key)
            self.api_key = key
            await self._append_bubble("system", "API key saved.", "msg-system", "left")
            self._refresh_header()

        elif cmd == "/localmodel" and len(parts) >= 2:
            path = parts[1]
            self.local_model_path = path
            save_local_config(local_model_path=path)
            await self._append_bubble("system", f"Local model set to `{path}`.", "msg-system", "left", markdown=True)
            self._refresh_header()

        elif cmd == "/localserver" and len(parts) >= 2:
            path = parts[1]
            self.local_server_bin = path
            save_local_config(local_server_bin=path)
            await self._append_bubble("system", f"llama-server binary set to `{path}`.", "msg-system", "left", markdown=True)

        elif cmd == "/settings":
            self.action_open_settings()

        elif cmd == "/new":
            self.session = ChatSession()
            log = self.query_one("#chat-log", VerticalScroll)
            await log.remove_children()
            self.query_one("#input-field", Input).placeholder = "/help"
            await self._append_bubble("system", "Chat cleared.", "msg-system", "left")

        elif cmd in ("/quit", "/exit"):
            self.exit()

        else:
            await self._append_bubble("system", f"Unknown command: {cmd}", "msg-error", "left")

    def action_open_settings(self) -> None:
        self.push_screen(
            SettingsScreen(
                self.system_prompt,
                self.user_prompt,
                self.matrix_enabled,
                self.matrix_effect,
                self.matrix_density,
                self.matrix_width,
                self.matrix_min_len,
                self.matrix_max_len_div,
                self.dna_speed,
                self.dna_width,
            )
        )

    def apply_matrix_settings(self) -> None:
        """Push the current enabled/effect/density/width/length settings
        into both effect widgets on both sides immediately, so the person
        sees the change without resizing the terminal or restarting the
        app. Only the widget matching self.matrix_effect ends up enabled
        and visible; the other is disabled and hidden."""
        is_matrix = self.matrix_effect == "matrix"
        for side in ("left", "right"):
            rain = self.query_one(f"#matrix-{side}", MatrixRain)
            dna = self.query_one(f"#dna-{side}", DnaHelix)

            rain.set_enabled(self.matrix_enabled and is_matrix)
            rain.apply_settings(
                self.matrix_density, self.matrix_width, self.matrix_min_len, self.matrix_max_len_div
            )
            rain.display = is_matrix

            dna.set_enabled(self.matrix_enabled and not is_matrix)
            dna.apply_settings(self.dna_speed, self.dna_width)
            dna.display = not is_matrix

    # ------------------------------------------------------------------
    # Gemini turn (streaming + function calling loop)
    # ------------------------------------------------------------------

    def _stream_backend(self):
        """Returns the async generator for the currently selected backend,
        with the same (kind, payload) event shape either way."""
        if self.model_mode == "local":
            return local_llm.stream_reply(self.session, self._effective_system_prompt(), self.local_port)
        return stream_reply(self.session, self.api_key, self._effective_system_prompt())

    async def _run_turn(self) -> None:
        self.is_loading = True

        if self.model_mode == "local":
            self._set_status(f"● starting local model ({Path(self.local_model_path).name})…")
            try:
                await asyncio.to_thread(
                    local_llm.ensure_server,
                    self.local_model_path,
                    self.local_server_bin,
                    self.local_port,
                    self.local_ctx,
                    self.local_ngl,
                )
            except local_llm.LocalServerError as e:
                await self._append_bubble("error", str(e), "msg-error", "left")
                self.is_loading = False
                self._set_status("")
                return

        self._set_status("● thinking…")

        # One box for the whole turn: any tool calls the model makes along
        # the way get mounted as collapsibles inside this same container,
        # in order, and the final text reply is the last thing in it —
        # a single message instead of one bubble per model round-trip.
        col, box, bubble = await self._append_turn_box("eve", "left")

        while True:
            model_msg = Message(role="model", text="")
            got_function_call: dict | None = None
            had_error = False

            async for kind, payload in self._stream_backend():
                if kind == "text_delta":
                    model_msg.text += payload["text"]
                    if payload.get("thought_signature"):
                        model_msg.thought_signature = payload["thought_signature"]
                    await self._update_markdown(bubble, model_msg.text)
                elif kind == "function_call":
                    got_function_call = payload
                elif kind == "error":
                    had_error = True
                    if not model_msg.text:
                        await self._update_markdown(bubble, "*(no response)*")
                    await self._append_bubble("error", payload["message"], "msg-error", "left")
                elif kind == "done":
                    pass

            if had_error:
                break

            if got_function_call:
                model_msg.function_call = {
                    "name": got_function_call["name"],
                    "args": got_function_call["args"],
                    "thought_signature": got_function_call.get("thought_signature"),
                }
                self.session.messages.append(model_msg)

                command = got_function_call["args"].get("command", "")
                tool_block = await self._append_tool_block(box, command)
                self._set_status(f"● running: {command}")

                exit_code, output = await run_command(command)
                await self._finish_tool_block(tool_block, command, exit_code, output)

                self.session.messages.append(
                    Message(
                        role="function",
                        function_response={
                            "name": got_function_call["name"],
                            "result": {"exit_code": exit_code, "output": output},
                        },
                    )
                )

                # Keep the reply widget as the last child of the box, below
                # every collapsible mounted so far, regardless of how many
                # commands this turn ends up running.
                box.move_child(bubble, after=tool_block)
                continue
            else:
                self.session.messages.append(model_msg)
                break

        self.is_loading = False
        self._set_status("")

    def action_new_chat(self) -> None:
        self.call_later(self._handle_command, "/new")

    def on_unmount(self) -> None:
        # Only stops a llama-server we spawned ourselves; leaves an
        # externally-started one (e.g. run by hand for debugging) alone.
        local_llm.stop_server()


if __name__ == "__main__":
    EveApp().run()
