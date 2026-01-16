Set-Location package-python
uv sync
.venv\Scripts\activate
uv pip install pip
uv pip install -r pyproject.toml --extra dev
Set-Location ..
