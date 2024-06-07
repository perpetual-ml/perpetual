cd python-package
black python/perpetual/
black tests/
maturin develop --release
pytest .
cd ..