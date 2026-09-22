# tetris_for_laya

A pure-Python Rich terminal Tetris environment where a local Laya model chooses every tetromino
placement on Apple Silicon.

![Laya playing Tetris in the Rich terminal UI](docs/assets/demo.gif)

The demo was recorded from a real local run with the multilingual MLX checkpoint and seed 7. Every
visible placement calls Laya; the falling animation was resampled to 10 FPS to keep the GIF small.

The game engine is implemented independently. Its restrained three-column terminal presentation
is inspired by [`samtay/tetris`](https://github.com/samtay/tetris), and its model integration follows
the explicit planner → typed decision → safety shield pattern demonstrated by
[`mizorewww/laya-mlx`](https://github.com/mizorewww/laya-mlx).

## Highlights

- Deterministic 10 × 20 Tetris with a seeded 7-bag, wall kicks, ghost pieces, scoring and levels
- Local FP16 inference through `laya-mlx`; no cloud API and no downloads during play
- Visible model probabilities, proposed placement, executed placement and inference latency
- An explicit safety shield whose interventions are labeled and counted
- Human keyboard mode using the same engine and Rich interface

## Requirements

- Apple Silicon Mac running macOS 14 or later
- Python 3.11 or later (`uv` will select an installed compatible Python)
- A true-color terminal at least 78 columns × 22 rows
- About 700 MiB for the default multilingual checkpoint

## Install

```bash
uv sync --extra dev --python 3.12
uv run hf download aac6fef/laya-multilingual-mlx \
  --local-dir models/laya-multilingual-mlx
```

If Hugging Face is unreachable through the direct connection:

```bash
export all_proxy=http://127.0.0.1:7897
uv run hf download aac6fef/laya-multilingual-mlx \
  --local-dir models/laya-multilingual-mlx
```

The model is downloaded explicitly. Gameplay enables Hugging Face offline mode and never downloads
weights in the middle of a run.

## Play

Let Laya play, one typed decision per tetromino:

```bash
uv run tetris-for-laya
```

Play manually:

```bash
uv run tetris-for-laya --player human
```

Manual controls:

| Action | Keys |
|---|---|
| Move | `←` / `→` or `h` / `l` |
| Rotate | `↑` or `k` |
| Soft drop | `↓` or `j` |
| Hard drop | `Space` |
| Restart | `r` |
| Quit | `q`, `Esc`, or `Ctrl-C` |

Run a finite, non-rendered model check:

```bash
uv run tetris-for-laya --headless --pieces 150 --seed 7
```

Other useful flags:

```text
--model PATH       Local MLX checkpoint (default: models/laya-multilingual-mlx)
--level 0..9       Starting level
--fps 1..240       Laya drop-animation speed
--optimize         Enable MLX compilation and the bounded prompt cache
--no-alt-screen    Leave the final Rich frame in terminal scrollback
```

`--optimize` has a noticeable first-use compilation cost and is intended for longer runs; eager
mode is the better default for a quick game.

## What Laya actually does

This is a feature-assisted typed-decision environment, not an end-to-end vision agent.

For each tetromino, the pure game engine enumerates every legal straight-drop landing and simulates
its result. A deterministic planner keeps up to two strongest safe-frontier candidates using line
clears, holes, aggregate height and bumpiness. Laya receives those exact consequences as a `choice`
question and selects the landing. The TUI shows its original probabilities, proposed placement,
executed placement and measured inference time. If only one non-top-out candidate remains, the
planner executes it without calling Laya and the screen explicitly shows `PLANNER ONLY` and
`SHIELD APPLIED`.

When the stack reaches 12 rows, an explicit safety shield may replace a materially worse proposal
with the planner's best candidate. Every intervention is labeled `SHIELD APPLIED` and counted in the
final JSON summary. It is never attributed to Laya. This mirrors the honest planner-assisted design
of the Laya MLX Snake demo while keeping Laya responsible for the normal placement decision.

## Architecture

```text
src/tetris_for_laya/
├── game.py    deterministic rules, snapshots, landing simulation and board metrics
├── policy.py  Laya prompt, bounded candidate planner and safety shield
├── ui.py      pure Rich renderables; no model or game-loop side effects
└── cli.py     raw keyboard input, Rich Live loop, animation and headless runner
```

## Verify

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

## References and acknowledgements

- [`mizorewww/laya-mlx`](https://github.com/mizorewww/laya-mlx) provides the native MLX runtime and
  the Snake demo that inspired this environment's typed-decision and visible-shield integration.
- [`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya) is the upstream Laya implementation
  and training project.
- [`aac6fef/laya-multilingual-mlx`](https://huggingface.co/aac6fef/laya-multilingual-mlx) is the
  pre-converted FP16 checkpoint used by default.
- [`samtay/tetris`](https://github.com/samtay/tetris) inspired the restrained terminal visual
  grammar. No source code or assets were copied.

This repository is MIT-licensed. Laya, `laya-mlx`, their model weights and their dependencies retain
their respective licenses and notices.
