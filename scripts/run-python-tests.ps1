Set-Location python-package
python -m black python/perpetual/
python -m black tests/
python -m black examples/
maturin develop --release
pytest .
Set-Location ..