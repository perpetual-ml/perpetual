Set-Location python-package
maturin develop --release
pytest tests/test_booster.py::test_booster_terminate_missing_features -s
Set-Location ..