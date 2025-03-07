Set-Location python-package
python -m black python/perpetual/
python -m black tests/
python -m black examples/
maturin develop --release
pytest tests/test_save_load.py::TestSaveLoadFunctions::test_booster_saving_with_monotone_constraints -s
Set-Location ..