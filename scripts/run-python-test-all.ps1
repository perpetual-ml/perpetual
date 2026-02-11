Set-Location package-python
uv run python -m ruff check --fix .
uv run python -m ruff format .
uv run maturin develop --release
uv run pytest .
Set-Location ..
