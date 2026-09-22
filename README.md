# tetris_for_laya

A pure-Python Rich terminal Tetris environment where a local Laya model plays one keyboard
action at a time on Apple Silicon.

![Laya playing Tetris in the Rich terminal UI](docs/assets/demo.gif)

The demo records consecutive real game states from the multilingual MLX checkpoint, seed 7.
Every frame follows one fresh Laya decision and one executed input. The GIF is replayed at the
measured average tick rate of the recorded run, without interpolated movement or a preselected
landing animation.

The game engine is implemented independently. Its restrained three-column terminal presentation
is inspired by [`samtay/tetris`](https://github.com/samtay/tetris), and its model integration follows
the explicit planner → typed decision → safety shield pattern demonstrated by
[`mizorewww/laya-mlx`](https://github.com/mizorewww/laya-mlx).

## Highlights

- Deterministic 10 × 20 Tetris with a seeded 7-bag, wall kicks, ghost pieces, scoring and levels
- Local FP16 inference through `laya-mlx`; no cloud API and no downloads during play
- A fresh model decision for every LEFT, RIGHT, ROTATE, DOWN or WAIT action
- Visible action probabilities, proposed/executed keys, step count and inference latency
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

Let Laya play, one typed decision per keyboard action:

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
--fps 1..240       Optional action-rate cap; default runs as fast as Laya completes each tick
--optimize         Enable MLX compilation and the bounded prompt cache
--no-alt-screen    Leave the final Rich frame in terminal scrollback
```

`--optimize` has a noticeable first-use compilation cost and is intended for longer runs; eager
mode is the better default for a quick game.

## What Laya actually does

This is a feature-assisted typed-decision environment, not an end-to-end vision agent.

Each control cycle observes the current piece position, rotation and gravity phase, then asks Laya
one `choice` question over five actions:

| Key | Immediate effect |
|---|---|
| `LEFT` | Move one column left if unobstructed |
| `RIGHT` | Move one column right if unobstructed |
| `ROTATE` | Rotate clockwise once, using the same wall kicks as human play |
| `DOWN` | Move one row down, or lock if already resting |
| `WAIT` | Leave the piece alone and let the natural gravity clock advance |

One game tick performs one fresh inference, applies exactly one input and renders the resulting
state. The next tick starts immediately after that cycle completes: no fixed sleep, overlapping
inference or queued actions. A piece usually takes many model calls to settle. The agent never calls
the atomic placement or hard-drop helpers. Its controls use deterministic simulation time: gravity
advances one row every four inputs,
including blocked inputs. `DOWN` on that fourth input counts as the gravity row, so it never drops
twice. A newly spawned piece is not moved by the previous piece's locking input. An explicit `--fps`
adds a rate cap for slow observation without changing physics. Headless mode uses the same tick loop.

A planner supplies **lookahead guidance** for each key: the best reachable future board and the
number of further inputs needed. It explores the engine's real movement, collision, rotation and
gravity rules, scoring line clears, holes, aggregate height and bumpiness. The graph is cached while
the board stays fixed and queried from the actual state every step. These are future estimates, not
the immediate result of a key, and no planned path is automatically executed.

Laya still makes every action proposal. A visible safety shield can replace an unknown/blocked key,
a repeated state, or a key whose reachable futures all top out when a safer key exists. At high stack
heights it can also reject materially worse outcomes. The TUI retains the original probabilities and
shows the proposed and executed keys; every override is labeled `SHIELD APPLIED` and counted.

In a local seed-7 smoke run, **50 pieces required 1,841 model calls and 1,841 engine steps**, clearing
8 lines without topping out, with 256 explicit shield interventions. Mean model inference was
27.17 ms, excluding planner/rendering time (about 36 headless ticks per second overall). This assisted
single-step result is not comparable to
the original release's one-decision-per-piece scores, and inference latency varies by hardware.

## Architecture

```text
src/tetris_for_laya/
├── game.py    deterministic single-step rules, pure previews, snapshots and board metrics
├── policy.py  per-action Laya prompt, reachable-state lookahead and visible safety shield
├── ui.py      pure Rich renderables; no model or game-loop side effects
└── cli.py     observe → decide → step → render loop and headless runner
```

## Verify

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

The regressions check real intermediate motion, collisions, gravity timing, lock/spawn transitions,
repeated inference and that neither atomic placement nor hard drop can be used by the Laya loop.

To record a fresh demo on macOS (requires FFmpeg and the downloaded model):

```bash
uv run python scripts/record_demo.py
```

The recorder uses the same policy, `game.step()` and Rich renderer, saving every consecutive step.
Only the resulting GIF is kept; temporary rendering frames are removed automatically.

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
