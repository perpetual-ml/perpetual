Set-Location python-package
maturin develop --release
pytest tests/test_booster.py::test_booster_max_cat -s
Set-Location ..