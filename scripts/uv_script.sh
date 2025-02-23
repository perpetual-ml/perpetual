cd python-package
uv sync
source .venv\Scripts\activate
uv pip install pip
uv pip install -r pyproject.toml --extra dev
cd ..
