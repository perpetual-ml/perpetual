Set-Location python-package
python -m black python/perpetual/
python -m black tests/
python -m black examples/
maturin develop --release
pytest tests/test_sklearn.py::test_sklearn_compat_ranking -s
Set-Location ..
