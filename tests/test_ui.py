from rich.console import Console

from tetris_for_laya.game import BOARD_HEIGHT, BOARD_WIDTH, GameSnapshot, Piece
from tetris_for_laya.ui import (
    BOARD_PANEL_WIDTH,
    SCREEN_HEIGHT,
    CandidateView,
    DecisionView,
    render_board,
    render_game,
)


def _snapshot(
    *,
    board: tuple[tuple[str | None, ...], ...] | None = None,
    current: Piece = Piece("T", 0, 3, 0),
    ghost_y: int = 18,
    game_over: bool = False,
) -> GameSnapshot:
    return GameSnapshot(
        board=board or tuple(tuple(None for _ in range(BOARD_WIDTH)) for _ in range(BOARD_HEIGHT)),
        current=current,
        ghost_y=ghost_y,
        next_kind="I",
        score=12_340,
        lines=12,
        level=1,
        game_over=game_over,
        pieces=37,
    )


def _render(renderable: object, *, width: int = 100, color: bool = False) -> str:
    console = Console(
        width=width,
        height=40,
        record=True,
        force_terminal=color,
        color_system="truecolor" if color else None,
    )
    console.print(renderable)
    return console.export_text(styles=color)


def test_board_has_exact_fixed_dimensions_and_heavy_border() -> None:
    output = _render(render_board(_snapshot()), width=BOARD_PANEL_WIDTH)
    lines = output.splitlines()

    assert len(lines) == SCREEN_HEIGHT
    assert all(len(line) == BOARD_PANEL_WIDTH for line in lines)
    assert lines[0].startswith("┏") and lines[0].endswith("┓")
    assert lines[-1] == "┗" + "━" * (BOARD_PANEL_WIDTH - 2) + "┛"
    assert all(line.startswith("┃") and line.endswith("┃") for line in lines[1:-1])


def test_board_renders_dark_cells_piece_colors_and_ghost_glyphs() -> None:
    board = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
    board[-1][0] = "Z"
    output = _render(
        render_board(_snapshot(board=tuple(tuple(row) for row in board))),
        width=BOARD_PANEL_WIDTH,
        color=True,
    )

    assert " ." in output
    assert "◤◢" in output
    assert "[101m" in output  # settled Z uses a bright-red background
    assert "[105m" in output  # active T uses a bright-magenta background
    assert "[2;95m◤◢" in output  # ghost is dim magenta


def test_game_screen_contains_stats_next_help_and_decision_telemetry() -> None:
    help_output = _render(render_game(_snapshot()))
    assert all(
        text in help_output
        for text in ("STATS", "12,340", "LEVEL", "LINES", "PIECES", "NEXT", "HELP")
    )

    decision = DecisionView(
        candidates=(
            CandidateView("LEFT", 0.625),
            CandidateView("RIGHT", 0.375),
            CandidateView("ROTATE", 0),
            CandidateView("DOWN", 0),
            CandidateView("WAIT", 0),
        ),
        proposed="LEFT",
        executed="RIGHT",
        shield_applied=True,
        inference_ms=18.25,
        step=42,
    )
    decision_output = _render(render_game(_snapshot(), decision))

    assert all(
        text in decision_output
        for text in (
            "LAYA",
            "ACTIONS",
            "ROTATE",
            "DOWN",
            "WAIT",
            "62.5%",
            "37.5%",
            "PROPOSED",
            "EXECUTED",
            "APPLIED",
            "18.2 ms",
            "STEP",
            "42",
        )
    )
    assert len(decision_output.splitlines()) == SCREEN_HEIGHT
    assert "+1 more" not in decision_output


def test_game_over_is_visible_in_stats_and_board() -> None:
    output = _render(render_game(_snapshot(game_over=True)))
    assert output.count("GAME OVER") == 2
