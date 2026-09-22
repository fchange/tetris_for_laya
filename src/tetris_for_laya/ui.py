"""Rich renderables for the terminal game screen.

This module deliberately contains no input or refresh loop.  ``render_game``
returns a regular Rich renderable, so the CLI can print it once or hand it to
``rich.live.Live``.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich import box
from rich.align import Align
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .game import BOARD_HEIGHT, BOARD_WIDTH, GameSnapshot, Piece, piece_cells

BOARD_PANEL_WIDTH = BOARD_WIDTH * 2 + 2
SCREEN_HEIGHT = BOARD_HEIGHT + 2
SIDE_PANEL_WIDTH = 22
DECISION_PANEL_WIDTH = 30

PIECE_COLORS = {
    "I": "bright_cyan",
    "O": "bright_yellow",
    "T": "bright_magenta",
    "S": "bright_green",
    "Z": "bright_red",
    "J": "bright_blue",
    "L": "white",
}

_EMPTY_CELL = " ."
_EMPTY_STYLE = "grey19"
_GHOST_CELL = "◤◢"


@dataclass(frozen=True, slots=True)
class CandidateView:
    """One policy candidate prepared for display."""

    label: str
    probability: float


@dataclass(frozen=True, slots=True)
class DecisionView:
    """Policy telemetry consumed by the UI without depending on policy code."""

    candidates: tuple[CandidateView, ...] = ()
    proposed: str | None = None
    executed: str | None = None
    shield_applied: bool = False
    inference_ms: float | None = None
    step: int = 0
    policy_name: str = "LAYA"


def _solid_cell(kind: str) -> Text:
    return Text("  ", style=f"on {PIECE_COLORS[kind]}")


def _board_text(snapshot: GameSnapshot) -> Text:
    active = {
        (x, y)
        for x, y in piece_cells(snapshot.current)
        if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT
    }
    ghost_piece = Piece(
        snapshot.current.kind,
        snapshot.current.rotation,
        snapshot.current.x,
        snapshot.ghost_y,
    )
    ghost = {
        (x, y)
        for x, y in piece_cells(ghost_piece)
        if 0 <= x < BOARD_WIDTH and 0 <= y < BOARD_HEIGHT
    }

    rendered = Text()
    for y, row in enumerate(snapshot.board):
        for x, settled_kind in enumerate(row):
            if (x, y) in active:
                rendered.append_text(_solid_cell(snapshot.current.kind))
            elif settled_kind is not None:
                rendered.append_text(_solid_cell(settled_kind))
            elif (x, y) in ghost:
                rendered.append(
                    _GHOST_CELL,
                    style=f"dim {PIECE_COLORS[snapshot.current.kind]}",
                )
            else:
                rendered.append(_EMPTY_CELL, style=_EMPTY_STYLE)
        if y < BOARD_HEIGHT - 1:
            rendered.append("\n")
    return rendered


def render_board(snapshot: GameSnapshot) -> Panel:
    """Render the fixed 10x20 playfield as a 22x22 heavy-bordered panel."""

    return Panel(
        _board_text(snapshot),
        title="[bold white] TETRIS [/bold white]",
        subtitle="[bold bright_red] GAME OVER [/bold bright_red]" if snapshot.game_over else None,
        box=box.HEAVY,
        border_style="grey50",
        padding=(0, 0),
        width=BOARD_PANEL_WIDTH,
        height=SCREEN_HEIGHT,
    )


def _stats_panel(snapshot: GameSnapshot) -> Panel:
    stats = Text(justify="center")
    values = (
        ("SCORE", f"{snapshot.score:,}"),
        ("LEVEL", str(snapshot.level)),
        ("LINES", str(snapshot.lines)),
        ("PIECES", str(snapshot.pieces)),
    )
    for index, (label, value) in enumerate(values):
        if index:
            stats.append("\n\n")
        stats.append(f"{label}\n", style="dim white")
        stats.append(value, style="bold bright_white")

    status = "GAME OVER" if snapshot.game_over else "PLAYING"
    status_style = "bold bright_red" if snapshot.game_over else "bold bright_green"
    stats.append("\n\nSTATUS\n", style="dim white")
    stats.append(status, style=status_style)
    return Panel(
        Align.center(stats, vertical="middle"),
        title="[bold] STATS [/bold]",
        box=box.HEAVY,
        border_style="grey50",
        padding=(0, 1),
        width=SIDE_PANEL_WIDTH,
        height=SCREEN_HEIGHT,
    )


def _next_piece(kind: str) -> Text:
    cells = piece_cells(Piece(kind, 0, 0, 0))
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    normalized = {(x - min_x, y - min_y) for x, y in cells}

    preview = Text()
    for y in range(4):
        preview.append(" " * 10)
        for x in range(4):
            if (x, y) in normalized:
                preview.append_text(_solid_cell(kind))
            else:
                preview.append("  ")
        if y < 3:
            preview.append("\n")
    return preview


def _next_panel(kind: str) -> Panel:
    return Panel(
        _next_piece(kind),
        title="[bold] NEXT [/bold]",
        box=box.HEAVY,
        border_style="grey50",
        padding=(0, 0),
        width=DECISION_PANEL_WIDTH,
        height=7,
    )


def _decision_panel(decision: DecisionView | None) -> Panel:
    if decision is None:
        help_text = Text()
        for key, action in (
            ("← →", "move"),
            ("↑", "rotate"),
            ("↓", "soft drop"),
            ("space", "hard drop"),
            ("q", "quit"),
        ):
            help_text.append(f"{key:<7}", style="bold bright_cyan")
            help_text.append(f"{action}\n", style="white")
        return Panel(
            Align.center(help_text, vertical="middle"),
            title="[bold] HELP [/bold]",
            box=box.HEAVY,
            border_style="grey50",
            padding=(0, 1),
            width=DECISION_PANEL_WIDTH,
            height=SCREEN_HEIGHT - 7,
        )

    details = Text()
    details.append("ACTIONS\n", style="dim white")
    visible_candidates = decision.candidates[:5]
    for candidate in visible_candidates:
        marker = "▸" if candidate.label == decision.proposed else " "
        label = candidate.label[:16]
        details.append(f"{marker} {label:<16}", style="bright_cyan" if marker.strip() else "white")
        details.append(f" {candidate.probability:>5.1%}\n", style="bold white")
    if len(decision.candidates) > len(visible_candidates):
        details.append(
            f"  +{len(decision.candidates) - len(visible_candidates)} more\n", style="dim"
        )

    details.append("\nPROPOSED  ", style="dim white")
    details.append(decision.proposed or "—", style="bold bright_cyan")
    details.append("\nEXECUTED  ", style="dim white")
    details.append(decision.executed or "—", style="bold bright_green")
    details.append("\nSHIELD    ", style="dim white")
    details.append(
        "APPLIED" if decision.shield_applied else "CLEAR",
        style="bold bright_yellow" if decision.shield_applied else "bright_green",
    )
    details.append("\nINFERENCE ", style="dim white")
    inference = "—" if decision.inference_ms is None else f"{decision.inference_ms:.1f} ms"
    details.append(inference, style="white")
    details.append("\nSTEP      ", style="dim white")
    details.append(str(decision.step), style="white")

    return Panel(
        details,
        title=f"[bold bright_magenta] {decision.policy_name} [/bold bright_magenta]",
        box=box.HEAVY,
        border_style="grey50",
        padding=(0, 1),
        width=DECISION_PANEL_WIDTH,
        height=SCREEN_HEIGHT - 7,
    )


def render_game(
    snapshot: GameSnapshot,
    decision: DecisionView | None = None,
) -> RenderableType:
    """Build the centered three-column game screen."""

    layout = Table.grid(padding=(0, 1))
    layout.add_column(width=SIDE_PANEL_WIDTH)
    layout.add_column(width=BOARD_PANEL_WIDTH)
    layout.add_column(width=DECISION_PANEL_WIDTH)
    layout.add_row(
        _stats_panel(snapshot),
        render_board(snapshot),
        Group(_next_panel(snapshot.next_kind), _decision_panel(decision)),
    )
    return Align.center(layout)
