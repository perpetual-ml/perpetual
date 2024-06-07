Set-Location python-package
maturin develop --release
pytest .
Set-Location ..