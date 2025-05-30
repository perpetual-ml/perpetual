#!/bin/sh
rm -Rf vendor vendor.tar.xz
cargo vendor
tar -cJ --no-xattrs -f vendor.tar.xz vendor
rm -Rf vendor
