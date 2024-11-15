Set-Location python-package
maturin develop --release
pytest tests/test_openml.py::test_sensory -s
Set-Location ..