Set-Location package-python
uv run python -m ruff check --fix .
uv run python -m ruff format .
uv run maturin develop --release
uv run pytest tests/core/test_tabarena.py::test_tabarena_save_load -s
Set-Location ..
