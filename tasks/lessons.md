# Project lessons

- Laya must observe the current board and choose one keyboard action on every step.
  Selecting a final landing once per tetromino and animating it is not step-by-step control.
  Verify real intermediate engine states and model call counts, and record demos from those states.
- A game tick accepts exactly one model action. Default pacing follows completed inference:
  do not impose an arbitrary animation FPS cap, overlap decisions, or queue multiple actions.
- Include WAIT as a first-class action for natural gravity. A deliberate no-op is valid input;
  it must not be confused with a blocked movement or replaced by the safety shield for that reason.
- Do not phrase the policy as though every tick requires movement. Explicitly prefer WAIT when the
  pose is already suitable, discourage reversible left/right motion, and keep the TUI action list in
  sync whenever the model's action space changes.
