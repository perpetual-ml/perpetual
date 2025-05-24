# Contributing to `perpetual`

## Development Setup

The repository comes with a predefined set of development tools in the `Makefile` which can be listed by running `make help` or `make`. Before you start contributing or developing further it is required that you initialize the artifacts and dependencies needed for conducting the unit-tests. To do this run the following command:

```sh
make init
```

For development, it is assumed you have stable Rust installed, and at least Python 3.9.

### Running tests

To run the tests in the repository use `make test` - this will run all tests available in the repository. For Python or Rust specific tests that has no downstream or upstream changes use `make py-test` or `make rust-test`, respectively. 

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

The [`pre-commit`](https://pre-commit.com/) framework should be installed and used to ensure all commits meet the required formatting, and linting checks prior to a commit being made to the repository. This is installed by `make init` too.

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
