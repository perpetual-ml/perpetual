# Perpetual

## Python API Reference

<img alt="PyPI - Version" src="https://img.shields.io/pypi/v/perpetual?link=https%3A%2F%2Fpypi.org%2Fproject%2Fperpetual%2F">

<img alt="Crates.io Version" src="https://img.shields.io/crates/v/perpetual?link=https%3A%2F%2Fcrates.io%2Fcrates%2Fperpetual">

The `PerpetualBooster` class is currently the only public facing class in the package, and can be used to train gradient boosted decision tree ensembles with multiple objective functions.

::: perpetual.PerpetualBooster

## Logging output

Info is logged while the model is being trained if the `log_iterations` parameter is set to a value greater than `0` while fitting the booster. The logs can be printed to stdout while training like so.

```python
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

model = PerpetualBooster(log_iterations=1)
model.fit(X, y)

# INFO:perpetual.perpetualbooster:Completed iteration 0 of 10
# INFO:perpetual.perpetualbooster:Completed iteration 1 of 10
# INFO:perpetual.perpetualbooster:Completed iteration 2 of 10
```

The log output can also be captured in a file also using the `logging.basicConfig()` `filename` option.

```python
import logging
logging.basicConfig(filename="training-info.log")
logging.getLogger().setLevel(logging.INFO)

model = PerpetualBooster(log_iterations=10)
model.fit(X, y)
```
