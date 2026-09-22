from dataclasses import FrozenInstanceError

import pytest

from tetris_for_laya.game import BOARD_HEIGHT, BOARD_WIDTH, Landing, TetrisGame


def test_snapshot_is_immutable_and_seeded_bag_is_reproducible() -> None:
    first = TetrisGame(seed=42).snapshot()
    second = TetrisGame(seed=42).snapshot()

    assert first == second
    assert len(first.board) == BOARD_HEIGHT
    assert all(len(row) == BOARD_WIDTH for row in first.board)
    assert first.current.kind != first.next_kind

    with pytest.raises(FrozenInstanceError):
        first.score = 99  # type: ignore[misc]


def _seed_for(kind: str) -> int:
    return next(seed for seed in range(100) if TetrisGame(seed).snapshot().current.kind == kind)


def test_movement_rotation_and_wall_kick_stay_inside_the_board() -> None:
    game = TetrisGame(seed=_seed_for("I"))

    while game.move_right():
        pass

    assert game.snapshot().current.x == 6
    assert game.rotate()
    rotated = game.snapshot().current
    assert rotated.rotation == 1
    assert rotated.x == 6

    while game.move_right():
        pass
    assert game.snapshot().current.x == 7

    # Rotating the vertical I at the right wall succeeds by kicking one cell left.
    assert game.rotate()
    assert game.snapshot().current.x == 6


def test_ghost_and_hard_drop_lock_the_piece() -> None:
    game = TetrisGame(seed=_seed_for("I"))
    before = game.snapshot()

    assert before.ghost_y == 18
    assert game.hard_drop() == 19

    after = game.snapshot()
    assert after.pieces == 1
    assert sum(cell is not None for row in after.board for cell in row) == 4
    assert after.score == 38


def test_gravity_and_soft_drop_have_distinct_scoring() -> None:
    game = TetrisGame(seed=_seed_for("O"))

    assert game.tick()
    assert game.snapshot().current.y == 0
    assert game.snapshot().score == 0

    assert game.soft_drop()
    assert game.snapshot().current.y == 1
    assert game.snapshot().score == 1


def test_legal_landings_cover_unique_rotations_and_board_columns() -> None:
    game = TetrisGame(seed=_seed_for("I"))

    landings = game.legal_landings()

    assert len(landings) == 17  # seven horizontal positions plus ten vertical ones
    assert Landing(rotation=0, x=0, y=18) in landings
    assert Landing(rotation=1, x=-2, y=16) in landings
    assert {landing.rotation for landing in landings} == {0, 1}


def test_placement_metrics_match_the_atomically_applied_board() -> None:
    game = TetrisGame(seed=_seed_for("I"))
    option = next(
        option
        for option in game.placement_options()
        if option.landing == Landing(rotation=0, x=0, y=18)
    )

    assert (option.lines, option.holes, option.max_height) == (0, 0, 1)
    assert (option.aggregate_height, option.bumpiness, option.top_out) == (4, 1, False)
    assert game.apply_placement(option) == 0

    snapshot = game.snapshot()
    assert snapshot.pieces == 1
    assert snapshot.board[-1][:4] == ("I", "I", "I", "I")
    assert snapshot.score == 0  # Policy placement is not a manual hard drop.


def test_seeded_bags_contain_each_tetromino_once() -> None:
    game = TetrisGame(seed=17)
    kinds: list[str] = []

    for _ in range(14):
        kinds.append(game.snapshot().current.kind)
        option = min(
            game.placement_options(),
            key=lambda item: (item.holes, item.aggregate_height, item.bumpiness),
        )
        game.apply_placement(option)

    assert set(kinds[:7]) == set("IJLOSTZ")
    assert set(kinds[7:14]) == set("IJLOSTZ")
    assert not game.snapshot().game_over


def test_line_clear_uses_classic_scoring_and_updates_level() -> None:
    game = TetrisGame(seed=6, start_level=2)

    for _ in range(80):
        options = game.placement_options()
        clearing = [option for option in options if option.lines]
        if clearing:
            option = max(clearing, key=lambda item: item.lines)
            before = game.snapshot()
            cleared = game.apply_placement(option)
            expected = (0, 40, 100, 300, 1200)[cleared] * (before.level + 1)
            assert game.snapshot().score - before.score == expected
            assert game.snapshot().lines == before.lines + cleared
            assert game.snapshot().level == 2 + game.snapshot().lines // 10
            break

        game.apply_placement(
            min(
                options,
                key=lambda item: (
                    item.holes,
                    item.max_height,
                    item.aggregate_height,
                    item.bumpiness,
                ),
            )
        )
    else:
        pytest.fail("deterministic low-stack play never produced a line clear")


def test_top_out_ends_the_game_and_disables_further_actions() -> None:
    game = TetrisGame(seed=0)

    for _ in range(20):
        options = game.placement_options()
        top_out = next((option for option in options if option.top_out), None)
        if top_out is not None:
            game.apply_placement(top_out)
            break
        game.apply_placement(max(options, key=lambda item: (item.max_height, item.holes)))

    assert game.snapshot().game_over
    assert game.legal_landings() == ()
    assert game.placement_options() == ()
    assert not game.move_left()
    assert game.hard_drop() == 0
