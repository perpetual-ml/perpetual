cd package-python
python -m ruff check --fix .
python -m ruff format .
maturin develop --release
pytest .
cd ..