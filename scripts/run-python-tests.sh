cd python-package
python -m black python/perpetual/
python -m black tests/
maturin develop --release
pytest .
cd ..