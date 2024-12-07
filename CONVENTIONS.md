# Conventions

This document outlines some programming conventions that are not caught by automated tools.

* Use types where possible, including `jaxtyping` hints.
* Decorate functions with `beartype.beartype` unless they use a `jaxtyping` hint, in which case use `jaxtyped(typechecker=beartype.beartype)`.
* Variables referring to a filepath should be suffixed with `_fpath`. Directories are `_dpath`.
* Prefer `make` over `build` when naming functions that construct objects, and use `get` when constructing primitives (like string paths or config values).
* Only use `setup` for naming functions that don't return anything.
