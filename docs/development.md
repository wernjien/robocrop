# Development

## Tests

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests/ -q
```

On Windows, use `.venv\Scripts\python`.

## Launchers

`robocrop` (for Mac and Linux) and `robocrop.cmd` (for Windows) create `.venv`
on the first run, and reinstall the requirements whenever a requirements file
changes. They then run `python -m robocrop` with `src` on the path.
`--setup` stops after the install, once it has checked that the libraries
load. They read these environment variables:

| Variable | Effect |
|---|---|
| `ROBOCROP_PYTHON` | The Python used to create `.venv` (default: `python`, then `python3` or `py`) |
| `ROBOCROP_VENV` | Where to create the environment (default: `.venv` beside the launcher) |
| `ROBOCROP_NO_VLM=1` | Skip the caption libraries (torch and transformers) |
| `ROBOCROP_SKIP_SYNC=1` | Never run pip, even if the requirements changed |

Downloaded models are cached in `~/.cache/robocrop`, or in the folder that
`ROBOCROP_CACHE` names.

## Adding a detector

1. Add a module to `src/robocrop/detectors/` with a `BaseDetector` subclass.
   `yunet.py` is a good model to copy.
2. Any `--detector-opt` keys are passed to its `__init__` as keyword arguments.
3. Have `detect()` return its `Region` list through `self._finalize()`, which
   applies `--min-score` and sorts the largest first.
4. Add the class to `_BACKENDS`, and a one-line description to `DESCRIPTIONS`,
   in `src/robocrop/detectors/__init__.py`.
