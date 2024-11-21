# Contributing

Contributions are welcome.
This document outlines some programming conventions that are not caught by automated tools.

## Variable Names

Variables referring to a filepath should be suffixed with `_fpath`.

## Function Names

Prefer "make" over "build" when constructing objects, and use "get" when constructing primitives (like string paths or config values).
Only use "setup" for functions that don't return anything.

## Testing

If you discover a bug, try to add a test, even if the test is slow and requires disk or network access.
