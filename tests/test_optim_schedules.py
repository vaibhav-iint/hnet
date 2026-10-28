import math

import pytest

from hnet.optim import build_schedule


def test_wsd_inverse_sqrt_paper():
    # paper: 10% linear warmup, 20% inverse-square-root (1 - sqrt) decay
    s = build_schedule({"name": "wsd", "warmup": 0.1, "decay": 0.2, "decay_shape": "inverse_sqrt"}, 100)
    assert s(0) == pytest.approx(0.1)
    assert s(9) == pytest.approx(1.0)
    assert all(s(t) == 1.0 for t in range(10, 80))
    assert s(80) == pytest.approx(1 - math.sqrt(1 / 20))
    assert s(89) == pytest.approx(1 - math.sqrt(10 / 20))
    assert s(99) == pytest.approx(0.0)
    vals = [s(t) for t in range(80, 100)]
    assert all(a > b for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("shape,mid", [("linear", 0.5), ("cosine", 0.5), ("inverse_sqrt", 1 - math.sqrt(0.5)),
                                       ("none", 1.0)])
def test_wsd_shapes_and_min_lr(shape, mid):
    s = build_schedule({"name": "wsd", "warmup": 10, "decay": 20, "decay_shape": shape, "min_lr": 0.1}, 100)
    assert s(5) == pytest.approx(0.6)
    if shape == "none":
        assert s(99) == 1.0
        return
    assert s(89) == pytest.approx(0.1 + 0.9 * mid)
    assert s(99) == pytest.approx(0.1)


def test_constant_and_cosine():
    c = build_schedule("constant", 50)
    assert all(c(t) == 1.0 for t in range(50))
    c = build_schedule({"name": "constant", "warmup": 0.1}, 50)
    assert c(0) == pytest.approx(0.2) and c(4) == 1.0 and c(49) == 1.0
    k = build_schedule({"name": "cosine", "warmup": 0, "min_lr_ratio": 0.0}, 100)
    assert k(49) == pytest.approx(0.5)
    assert k(99) == pytest.approx(0.0)
    with pytest.raises(AssertionError):
        build_schedule({"name": "wsd", "warmup": 0.6, "decay": 0.6}, 100)
