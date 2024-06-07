# Building with polars is sooo slow.
# It's only there for the example, so let's remove it
# in the regular build process.
# Requires toml package
import shutil

import toml

ct = toml.load("Cargo.toml")

del ct["dev-dependencies"]
del ct["bench"]

with open("Cargo.toml", "w") as file:
    toml.dump(ct, file)

# Also delete the rust example.
shutil.rmtree("examples")
