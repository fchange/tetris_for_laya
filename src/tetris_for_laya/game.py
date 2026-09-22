"""Deterministic, UI-independent Tetris rules."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TypeAlias

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
WIDTH = BOARD_WIDTH
HEIGHT = BOARD_HEIGHT

Cell: TypeAlias = str | None
Board: TypeAlias = tuple[tuple[Cell, ...], ...]

# Coordinates use the conventional 4x4 tetromino boxes. Keeping all four
# rotation states makes snapshots straightforward for renderers to interpret.
TETROMINOES: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    "I": (
        ((0, 1), (1, 1), (2, 1), (3, 1)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((1, 0), (1, 1), (1, 2), (1, 3)),
    ),
    "J": (
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (0, 2), (1, 2)),
    ),
    "L": (
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ),
    "O": (
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (2, 1)),
    ),
    "S": (
        ((1, 0), (2, 0), (0, 1), (1, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((1, 1), (2, 1), (0, 2), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "T": (
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (1, 2)),
        ((1, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "Z": (
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((2, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 0), (0, 1), (1, 1), (0, 2)),
    ),
}

_LINE_SCORES = (0, 40, 100, 300, 1200)
_KICKS = ((0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0), (0, -1))


@dataclass(frozen=True, slots=True)
class Piece:
    kind: str
    rotation: int
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Landing:
    rotation: int
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class PlacementOption:
    id: str
    landing: Landing
    lines: int
    holes: int
    max_height: int
    aggregate_height: int
    bumpiness: int
    top_out: bool


@dataclass(frozen=True, slots=True)
class GameSnapshot:
    board: Board
    current: Piece
    ghost_y: int
    next_kind: str
    score: int
    lines: int
    level: int
    game_over: bool
    pieces: int


def piece_cells(piece: Piece) -> tuple[tuple[int, int], ...]:
    """Return the absolute board coordinates occupied by ``piece``."""

    return tuple(
        (piece.x + cell_x, piece.y + cell_y)
        for cell_x, cell_y in TETROMINOES[piece.kind][piece.rotation % 4]
    )


class TetrisGame:
    """A deterministic Tetris game suitable for a TUI or an agent policy."""

    def __init__(self, seed: int | None = None, start_level: int = 0) -> None:
        self._random = random.Random(seed)
        self._bag: list[str] = []
        self._board: list[list[Cell]] = [
            [None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)
        ]
        self.score = 0
        self.lines = 0
        self.start_level = start_level
        self.level = start_level
        self.game_over = False
        self.pieces = 0
        self.current = Piece(self._take_kind(), 0, 3, -1)
        self.next_kind = self._take_kind()

    def _take_kind(self) -> str:
        if not self._bag:
            self._bag = list("IJLOSTZ")
            self._random.shuffle(self._bag)
        return self._bag.pop()

    def _valid(self, piece: Piece) -> bool:
        return self._valid_on(piece, self._board)

    @staticmethod
    def _valid_on(piece: Piece, board: list[list[Cell]]) -> bool:
        for x, y in piece_cells(piece):
            if x < 0 or x >= BOARD_WIDTH or y >= BOARD_HEIGHT:
                return False
            if y >= 0 and board[y][x] is not None:
                return False
        return True

    def _translated(self, dx: int = 0, dy: int = 0) -> Piece:
        return Piece(
            self.current.kind,
            self.current.rotation,
            self.current.x + dx,
            self.current.y + dy,
        )

    def _move(self, dx: int = 0, dy: int = 0) -> bool:
        if self.game_over:
            return False
        candidate = self._translated(dx, dy)
        if not self._valid(candidate):
            return False
        self.current = candidate
        return True

    def move_left(self) -> bool:
        return self._move(dx=-1)

    def move_right(self) -> bool:
        return self._move(dx=1)

    def rotate(self) -> bool:
        """Rotate clockwise, trying a small deterministic set of wall kicks."""

        if self.game_over:
            return False
        rotation = (self.current.rotation + 1) % 4
        for dx, dy in _KICKS:
            candidate = Piece(
                self.current.kind,
                rotation,
                self.current.x + dx,
                self.current.y + dy,
            )
            if self._valid(candidate):
                self.current = candidate
                return True
        return False

    def _ghost_y(self, piece: Piece | None = None) -> int:
        candidate = piece or self.current
        while self._valid(Piece(candidate.kind, candidate.rotation, candidate.x, candidate.y + 1)):
            candidate = Piece(candidate.kind, candidate.rotation, candidate.x, candidate.y + 1)
        return candidate.y

    def soft_drop(self) -> bool:
        """Move down one row, or lock if blocked; manual movement scores one point."""

        if self._move(dy=1):
            self.score += 1
            return True
        if not self.game_over:
            self._lock_current()
        return False

    def tick(self) -> bool:
        """Advance gravity one row, locking when the piece can no longer fall."""

        if self._move(dy=1):
            return True
        if not self.game_over:
            self._lock_current()
        return False

    def hard_drop(self) -> int:
        if self.game_over:
            return 0
        destination = self._ghost_y()
        distance = destination - self.current.y
        self.current = Piece(self.current.kind, self.current.rotation, self.current.x, destination)
        self.score += distance * 2
        self._lock_current()
        return distance

    def _lock_current(self) -> int:
        top_out = False
        for x, y in piece_cells(self.current):
            if y < 0:
                top_out = True
            else:
                self._board[y][x] = self.current.kind

        cleared = self._clear_lines()
        self.score += _LINE_SCORES[cleared] * (self.level + 1)
        self.lines += cleared
        self.level = self.start_level + self.lines // 10
        self.pieces += 1

        if top_out:
            self.game_over = True
            return cleared

        self.current = Piece(self.next_kind, 0, 3, -1)
        self.next_kind = self._take_kind()
        if not self._valid(self.current):
            self.game_over = True
        return cleared

    def _clear_lines(self) -> int:
        remaining = [row for row in self._board if any(cell is None for cell in row)]
        cleared = BOARD_HEIGHT - len(remaining)
        self._board = [
            *([None for _ in range(BOARD_WIDTH)] for _ in range(cleared)),
            *remaining,
        ]
        return cleared

    def legal_landings(self) -> tuple[Landing, ...]:
        """Enumerate unique rotations that can be dropped vertically from above.

        This deliberately does not search paths involving sideways movement below
        the stack. Every result can be executed as a single rotate/column/drop
        choice, which keeps an agent's action space honest and reproducible.
        """

        if self.game_over:
            return ()

        landings: list[Landing] = []
        seen_shapes: set[tuple[tuple[int, int], ...]] = set()
        for rotation, shape in enumerate(TETROMINOES[self.current.kind]):
            min_x = min(x for x, _ in shape)
            max_x = max(x for x, _ in shape)
            min_y = min(y for _, y in shape)
            signature = tuple(sorted((x - min_x, y - min_y) for x, y in shape))
            if signature in seen_shapes:
                continue
            seen_shapes.add(signature)

            for x in range(-min_x, BOARD_WIDTH - max_x):
                y = -max(cell_y for _, cell_y in shape) - 1
                candidate = Piece(self.current.kind, rotation, x, y)
                while self._valid(Piece(candidate.kind, rotation, x, candidate.y + 1)):
                    candidate = Piece(candidate.kind, rotation, x, candidate.y + 1)
                landings.append(Landing(rotation, x, candidate.y))
        return tuple(landings)

    def _simulate_landing(self, landing: Landing) -> tuple[list[list[Cell]], int, bool]:
        board = [row.copy() for row in self._board]
        piece = Piece(self.current.kind, landing.rotation, landing.x, landing.y)
        top_out = False
        for x, y in piece_cells(piece):
            if y < 0:
                top_out = True
            else:
                board[y][x] = piece.kind

        remaining = [row for row in board if any(cell is None for cell in row)]
        cleared = BOARD_HEIGHT - len(remaining)
        board = [
            *([None for _ in range(BOARD_WIDTH)] for _ in range(cleared)),
            *remaining,
        ]
        if not top_out:
            next_piece = Piece(self.next_kind, 0, 3, -1)
            top_out = not self._valid_on(next_piece, board)
        return board, cleared, top_out

    @staticmethod
    def _board_metrics(board: list[list[Cell]]) -> tuple[int, int, int, int]:
        heights: list[int] = []
        holes = 0
        for x in range(BOARD_WIDTH):
            first_filled = next(
                (y for y in range(BOARD_HEIGHT) if board[y][x] is not None),
                BOARD_HEIGHT,
            )
            heights.append(BOARD_HEIGHT - first_filled)
            holes += sum(board[y][x] is None for y in range(first_filled + 1, BOARD_HEIGHT))
        max_height = max(heights, default=0)
        aggregate_height = sum(heights)
        bumpiness = sum(abs(left - right) for left, right in zip(heights, heights[1:]))
        return holes, max_height, aggregate_height, bumpiness

    def placement_options(self) -> tuple[PlacementOption, ...]:
        options: list[PlacementOption] = []
        for landing in self.legal_landings():
            board, cleared, top_out = self._simulate_landing(landing)
            holes, max_height, aggregate_height, bumpiness = self._board_metrics(board)
            options.append(
                PlacementOption(
                    id=(f"{self.current.kind}-r{landing.rotation}-x{landing.x}-y{landing.y}"),
                    landing=landing,
                    lines=cleared,
                    holes=holes,
                    max_height=max_height,
                    aggregate_height=aggregate_height,
                    bumpiness=bumpiness,
                    top_out=top_out,
                )
            )
        return tuple(options)

    def apply_placement(self, placement: PlacementOption | Landing) -> int:
        """Validate and apply an agent-selected landing as one atomic action."""

        landing = placement.landing if isinstance(placement, PlacementOption) else placement
        legal = (
            placement in self.placement_options()
            if isinstance(placement, PlacementOption)
            else landing in self.legal_landings()
        )
        if self.game_over or not legal:
            raise ValueError("placement is not legal for the current game state")
        self.current = Piece(self.current.kind, landing.rotation, landing.x, landing.y)
        return self._lock_current()

    def snapshot(self) -> GameSnapshot:
        return GameSnapshot(
            board=tuple(tuple(row) for row in self._board),
            current=self.current,
            ghost_y=self._ghost_y() if not self.game_over else self.current.y,
            next_kind=self.next_kind,
            score=self.score,
            lines=self.lines,
            level=self.level,
            game_over=self.game_over,
            pieces=self.pieces,
        )
