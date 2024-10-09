# Contributing to `perpetual`

## Development Setup

For development, it is assumed you have stable rust installed, and at least python 3.9. Additionally, in your python environment, you will need to install [`maturin`](https://github.com/PyO3/maturin).

### To run Rust tests

The rust test depend on some artifacts that are generated from a python script. You can either follow the steps in the python tests section, or run the following in an environment that is running python.

```sh
cd python-package
# Install the project in editable mode and all development dependencies
python -m pip install -e .[dev]
# You can now return to the rood directory and run the tests...
cd ..
python -m pip install pandas seaborn
python scripts/make_resources.py
```

If you have rust and the cargo package manager installed all you need to do to run the rust tests is in the root of the repository run the following command.

```sh
cargo test
```

### To run the python tests

Prior to running the tests, you should install `python-package` in editable mode. To do this, from the project root directory you can run the following.

```sh
cd python-package
# Install the project in editable mode and all development dependencies
python -m pip install -e .[dev]
# You can now return to the root directory and run the tests...
cd ..

# Prior to running the tests, build all required test artifacts
python scripts/make_resources.py

# Now you can run the tests.
# on Linux...
source scripts/run-python-tests.sh
```

The test script can also be run from powershell.

```powershell
# on Windows (powershell)
.\scripts\run-python-tests.ps1
```

This script, builds the package in release mode, installs it, and then runs the test. Because of this, it is useful to run this whenever you want to test out a change in the python package.

## Benchmarking

Benchmarking is run using the [`criterion`](https://github.com/bheisler/criterion.rs) Rust crate.
To run the benchmarks, you can run the following command from your terminal.

```sh
cargo bench
```

specific benchmarks can be targeted by referring to them by name.

```sh
cargo bench "fast sum"
```

## Pre-commit

The [`pre-commit`](https://pre-commit.com/) framework should be installed and used to ensure all commits meet the required formatting, and linting checks prior to a commit being made to the repository.

```sh
# Install pre-commit, either right in your default python install
# or using a tool such as pipx (https://pypa.github.io/pipx/)
python -m pip install pre-commit

# In the root of the repository
pre-commit install
```

## Serialization

The saving and loading of the model is all handled by the [`serde`](https://docs.rs/serde/1.0.163/serde/) and [`serde_json`](https://docs.rs/serde_json/latest/serde_json/) crates.

Because of this you will see the following attribue calls sprinkled throughout the package.

```rust
#[derive(Deserialize, Serialize)]
```

Additionally in order to not break backwards compatibility with models saved in previous versions, any new items added to the `Tree` or `PerpetualBooster` struts, should have a default value defined. This way models can be loaded, even if they were saved before the new fields was added.
A default value can be added for a fields using the `#[#[serde(default = "default_sample_method")]]` attribute. Where the string that default is referring to must be the name of a valid function, the following is a complete example of this.

```rust
use crate::sampler::SampleMethod, Sampler;

#[derive(Deserialize, Serialize)]
pub struct PerpetualBooster {
    // ...
    #[serde(default = "default_sample_method")]
    pub sample_method: SampleMethod,
    // ...
}

fn default_sample_method() -> SampleMethod {
    SampleMethod::None
}
```
