Set-Location python-package
maturin develop --release
pytest tests/test_multi_output.py::test_multi_output -s
Set-Location ..