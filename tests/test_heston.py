import numpy as np
import pytest
from heston import HestonParams, MarketDynamics, simulate_heston


@pytest.fixture
def p():
    return HestonParams(kappa=2, theta=.04, xi=.4, rho=-.7, v0=.04)


@pytest.fixture
def q():
    return MarketDynamics(measure="Q", spot0=100, risk_free_rate=.1, dividend_yield=0)


def test_shapes_and_valid_values(p, q):
    paths = simulate_heston(p, q, n_paths=100, n_steps=126, seed=7)
    assert paths.spot.shape == (100, 127)
    assert paths.variance.shape == (100, 127)
    assert paths.time_years.shape == (127,)
    assert np.all(paths.spot > 0)
    assert np.all(paths.variance >= 0)
    assert np.all(np.isfinite(paths.raw_variance))
    assert paths.measure == "Q"


def test_reproducible(p, q):
    a = simulate_heston(p, q, n_paths=14, n_steps=10, seed=121)
    b = simulate_heston(p, q, n_paths=14, n_steps=10, seed=121)
    assert np.array_equal(a.spot, b.spot)
    assert np.array_equal(a.raw_variance, b.raw_variance)


def test_feller_and_mean_formula(p):
    assert p.satisfies_feller
    assert p.feller_ratio == pytest.approx(1.0)
    assert p.expected_variance(2) == pytest.approx(.04)


def test_physical_measure_explicit():
    with pytest.raises(ValueError, match="require"):
        MarketDynamics(measure="P", spot0=100, risk_free_rate=.1)
    m = MarketDynamics(measure="P", spot0=100, risk_free_rate=.1, physical_total_return=.16)
    assert m.spot_drift == pytest.approx(.16)
    with pytest.raises(ValueError, match="omitted"):
        MarketDynamics(measure="Q", spot0=100, risk_free_rate=.1, physical_total_return=.16)


def test_no_vol_of_vol_steady_variance():
    p = HestonParams(kappa=2, theta=.04, xi=0, rho=-.7, v0=.04)
    q = MarketDynamics(measure="Q", spot0=100, risk_free_rate=.1)
    a = simulate_heston(p, q, n_paths=32, n_steps=50)
    np.testing.assert_allclose(a.variance, .04, rtol=0, atol=1e-14)


def test_martingale_monte_carlo_sanity(p, q):
    a = simulate_heston(p, q, n_paths=20000, n_steps=126, seed=15)
    disc = a.spot[:, -1] * np.exp(-q.spot_drift * a.time_years[-1])
    stderr = disc.std(ddof=1) / np.sqrt(len(disc))
    assert abs(disc.mean() - q.spot0) < 5 * stderr


def test_negative_invalid_model():
    with pytest.raises(ValueError, match="rho"):
        HestonParams(kappa=2, theta=.04, xi=.4, rho=-1.5, v0=.04)
