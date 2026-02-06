Set-Location package-python
python -m ruff check --fix .
python -m ruff format .
maturin develop --release
pytest tests/test_sklearn.py::test_sklearn_compat_ranking -s
Set-Location ..
