from . import config


def test_expand():
    cfg = {"lr": [1, 2, 3]}
    expected = [{"lr": 1}, {"lr": 2}, {"lr": 3}]
    actual = list(config.expand(cfg))

    assert expected == actual


def test_expand_two_fields():
    cfg = {"lr": [1, 2], "wd": [3, 4]}
    expected = [
        {"lr": 1, "wd": 3},
        {"lr": 1, "wd": 4},
        {"lr": 2, "wd": 3},
        {"lr": 2, "wd": 4},
    ]
    actual = list(config.expand(cfg))

    assert expected == actual


def test_expand_nested():
    cfg = {"sae": {"dim": [1, 2, 3]}}
    expected = [{"sae": {"dim": 1}}, {"sae": {"dim": 2}}, {"sae": {"dim": 3}}]
    actual = list(config.expand(cfg))

    assert expected == actual


def test_expand_nested_and_unnested():
    cfg = {"sae": {"dim": [1, 2]}, "lr": [3, 4]}
    expected = [
        {"sae": {"dim": 1}, "lr": 3},
        {"sae": {"dim": 1}, "lr": 4},
        {"sae": {"dim": 2}, "lr": 3},
        {"sae": {"dim": 2}, "lr": 4},
    ]
    actual = list(config.expand(cfg))

    assert expected == actual
