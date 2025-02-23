Set-Location python-package
uv sync
.venv\Scripts\activate
uv pip install pip
uv pip install -r pyproject.toml --extra dev
Set-Location ..
