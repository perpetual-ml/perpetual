Set-Location python-package
maturin develop --release
pytest tests/test_booster.py::test_polars -s
Set-Location ..