"""
Expands dicts with lists into a list of dicts.

Might be expanded in the future to support pseudo-random sampling from distributions to support random hyperparameter search, as in [this file](https://github.com/samuelstevens/sax/blob/main/sax/sweep.py).
"""

import collections.abc

import beartype

Primitive = float | int | bool | str


@beartype.beartype
def expand(
    config: dict[str, Primitive | list[Primitive]],
) -> collections.abc.Iterator[dict[str, Primitive]]:
    """
    Expands dicts with lists into a list of dicts.
    """
    yield from _expand_discrete(config)


@beartype.beartype
def _expand_discrete(
    config: dict[str, Primitive | list[Primitive]],
) -> collections.abc.Iterator[dict[str, Primitive]]:
    """
    Expands any list values in `config`
    """
    if not config:
        yield config
        return

    key, value = config.popitem()

    if isinstance(value, list):
        # Expand
        for c in _expand_discrete(config):
            for v in value:
                yield {**c, key: v}
    else:
        for c in _expand_discrete(config):
            yield {**c, key: value}
