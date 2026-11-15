# Working on spinescope (instructions for AI coding agents and humans)

Read DESIGN.md first. Key rules:

1. **No real patient data, ever.** Use the synthetic phantom (`spinescope.testing.phantom`) or public
   datasets under `$SPINESCOPE_DATA` (default `~/data/public`). Never read, copy or reference images or
   files from anywhere else on this machine (e.g. ~/Downloads, other projects). Never commit imaging
   files (`.dcm`, `.nii`, `.mha`, ...) or images derived from them; `docs/assets/` may contain images
   rendered from the phantom or from CC-licensed public data, with attribution in `docs/assets/README.md`.
2. **Nothing hand-picked** (DESIGN.md principle 2): no hard-coded slice indices, pixel coordinates,
   per-study offsets or thresholds in absolute signal units. Defaults go in `spinescope/config.py`, in mm
   or relative units, with a source comment.
3. **Privacy**: never put DICOM attributes outside `KEEP_TAGS` into outputs; never serialize `meta`
   keys that start with `_`; never write file paths of input data into outputs.
4. **Quality bar**: type hints on public functions, short docstrings, `logging` instead of `print` in
   library code, deterministic results (seeded RNG), unit tests for every public function (phantom-based),
   `ruff check`, `ruff format`, `mypy` and `pytest` all clean before you finish.
5. Run tools with the project venv: `.venv/bin/python -m pytest -q`, `.venv/bin/ruff check .`,
   `.venv/bin/mypy`. Package indexes go through the configured proxy (uv/pip config); don't change it.
6. Work on your own git branch / worktree when told to; commit with clear messages; don't push.
