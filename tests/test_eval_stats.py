import numpy as np
import pytest

from hnet.eval import stats as st


def test_mean_ci_covers_truth():
    rng = np.random.default_rng(0)
    x = (rng.random(2000) < 0.6).astype(float)
    ci = st.mean_ci(x, n_boot=500, seed=1)
    assert ci["lo"] < 0.6 < ci["hi"]
    se = np.sqrt(0.6 * 0.4 / 2000)
    assert ci["se"] == pytest.approx(se, rel=0.2)
    assert ci["hi"] - ci["lo"] == pytest.approx(2 * 1.96 * se, rel=0.2)


def test_bootstrap_deterministic_and_ratio():
    rng = np.random.default_rng(1)
    a, b = rng.random(300), rng.random(300) + 1
    assert st.ratio_ci(a, b, n_boot=200, seed=3) == st.ratio_ci(a, b, n_boot=200, seed=3)
    assert st.ratio_ci(a, b, n_boot=0)["value"] == pytest.approx(a.sum() / b.sum())


def test_macro_mean_and_average():
    v = np.array([1, 1, 1, 1, 0, 0], dtype=float)
    g = ["a", "a", "a", "a", "b", "b"]
    assert st.macro_mean_ci(v, g, n_boot=50)["value"] == pytest.approx(0.5)
    x, y = np.ones(10), np.zeros(30)
    avg = st.average_ci([(lambda i: x[i].mean(), 10, None), (lambda i: y[i].mean(), 30, None, 3.0)], n_boot=20)
    assert avg["value"] == pytest.approx(0.25)


def test_paired_bootstrap():
    rng = np.random.default_rng(2)
    base = (rng.random(3000) < 0.5).astype(float)
    better = np.where(rng.random(3000) < 0.1, 1.0, base)  # flips ~5% of items to correct
    d = st.paired_bootstrap(better, base, n_boot=500)
    assert d["value"] > 0 and d["lo"] > 0 and d["p_value"] < 0.01
    same = st.paired_bootstrap(base, base.copy(), n_boot=200)
    assert same["value"] == 0 and same["p_value"] == 1.0


@pytest.mark.parametrize("method", ["powerlaw", "quadratic_log", "isotonic"])
def test_multiplier_bpb_curve(method):
    t = np.geomspace(5e8, 1e10, 20)
    f = lambda x: 0.70 + 2.0 * (x / 1e9) ** -0.3  # noqa: E731
    y = f(t)
    # candidate at 2.5e9 tokens with the bpb the baseline reaches at 7.5e9 -> multiplier 3
    m = st.data_multiplier(t, y, 2.5e9, float(f(7.5e9)), method=method, n_boot=0)
    assert not m["increasing"]
    assert m["multiplier"] == pytest.approx(3.0, rel=0.03 if method != "isotonic" else 0.06)
    assert m["in_range"]


def test_multiplier_scores_with_noise_and_ci():
    rng = np.random.default_rng(0)
    t = np.geomspace(1e9, 2e11, 30)
    truth = lambda x: 0.30 + 0.05 * np.log(x / 1e9)  # noqa: E731
    y = truth(t) + rng.normal(0, 0.003, len(t))
    m = st.data_multiplier(t, y, 1e10, float(truth(4e10)), method="quadratic_log", n_boot=200, seed=1)
    assert m["increasing"]
    assert m["multiplier"] == pytest.approx(4.0, rel=0.15)
    assert m["ci_lo"] < 4.0 < m["ci_hi"]
    # unreachable -> nan for isotonic (no extrapolation)
    m2 = st.data_multiplier(t, y, 1e10, 0.99, method="isotonic", n_boot=0)
    assert np.isnan(m2["multiplier"])


def test_pav_monotone():
    y = np.array([1, 3, 2, 4, 3.5, 5])
    p = st._pav(y)
    assert np.all(np.diff(p) >= 0) and p.sum() == pytest.approx(y.sum())
