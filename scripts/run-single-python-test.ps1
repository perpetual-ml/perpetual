Set-Location python-package
maturin develop --release
pytest tests/test_booster.py::test_booster_no_variance -s
Set-Location ..