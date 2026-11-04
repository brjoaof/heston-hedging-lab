import math

import numpy as np
import pytest

from heston import HestonParams, MarketDynamics
from pricing import black_scholes_price, heston_price
from volatility import implied_volatility, heston_volatility_surface


def test_black_scholes_implied_volatility_roundtrip():
    for kind in ("call", "put"):
        for strike in (90.0, 100.0, 110.0):
            for tau in (0.1, 0.5, 1.0):
                for volatility in (0.12, 0.2, 0.5):
                    price = black_scholes_price(
                        100, strike, tau, volatility, rate=0.05,
                        dividend=0.01, option_type=kind,
                    )
                    recovered = implied_volatility(
                        price, spot=100, strike=strike, maturity=tau,
                        rate=0.05, dividend=0.01, option_type=kind,
                    )
                    assert recovered == pytest.approx(volatility, abs=2e-8)


def test_implied_volatility_bounds():
    assert implied_volatility(0, spot=100, strike=100, maturity=.5) == 0
    with pytest.raises(ValueError, match="no-arbitrage"):
        implied_volatility(-1, spot=100, strike=100, maturity=.5)
    with pytest.raises(ValueError, match="no finite"):
        implied_volatility(100, spot=100, strike=100, maturity=.5)
    with pytest.raises(ValueError, match="positive"):
        implied_volatility(5, spot=100, strike=100, maturity=0)
    with pytest.raises(ValueError, match="finite"):
        implied_volatility(float("nan"), spot=100, strike=100, maturity=.5)


def test_flat_surface_when_heston_variance_is_constant():
    params = HestonParams(kappa=2, theta=.04, xi=0, rho=-.7, v0=.04)
    market = MarketDynamics(measure="Q", spot0=100, risk_free_rate=.05)
    surface = heston_volatility_surface(
        params, market, maturities_days=[21, 63, 126],
        log_moneyness=[-.12, 0, .12],
    )
    assert surface.implied_volatility.shape == (3, 3)
    np.testing.assert_allclose(surface.implied_volatility, .2, atol=1e-8)
    for i, days in enumerate(surface.maturities_days):
        forward = 100 * math.exp(.05 * days / 252)
        np.testing.assert_allclose(surface.strikes[i], forward * np.exp(surface.log_moneyness))


def test_heston_prices_recovered_from_implied_volatility():
    params = HestonParams(kappa=2, theta=.04, xi=.4, rho=-.7, v0=.04)
    market = MarketDynamics(measure="Q", spot0=100, risk_free_rate=.10)
    surface = heston_volatility_surface(
        params, market, maturities_days=[21, 63, 126],
        log_moneyness=[-.12, 0, .12],
    )
    assert np.all(np.isfinite(surface.implied_volatility))
    assert np.all(surface.implied_volatility > 0)
    # The Heston leverage effect produces non-flat IV in this configuration.
    assert np.ptp(surface.implied_volatility[0]) > .01
    for i, day in enumerate(surface.maturities_days):
        for j, strike in enumerate(surface.strikes[i]):
            tau = day / 252
            heston = heston_price(params, market, strike=strike, maturity=tau)
            bs = black_scholes_price(
                100, strike, tau, surface.implied_volatility[i, j], rate=.10,
            )
            assert bs == pytest.approx(heston, abs=5e-8)


def test_surface_uses_requested_current_state_and_q_only():
    params = HestonParams(kappa=2, theta=.04, xi=.4, rho=-.7, v0=.04)
    market = MarketDynamics(measure="Q", spot0=100, risk_free_rate=.1)
    grid = dict(maturities_days=[21, 63], log_moneyness=[-.08, 0, .08])
    original = heston_volatility_surface(params, market, **grid)
    changed = heston_volatility_surface(params, market, spot=105, variance=.09, **grid)
    assert changed.spot == 105 and changed.variance == .09
    assert np.mean(changed.implied_volatility) > np.mean(original.implied_volatility)
    with pytest.raises(ValueError, match="Q"):
        heston_volatility_surface(
            params, MarketDynamics(measure="P", spot0=100, risk_free_rate=.1,
                                   physical_total_return=.15), **grid,
        )
    with pytest.raises(ValueError, match="increasing"):
        heston_volatility_surface(params, market, maturities_days=[63, 21],
                                  log_moneyness=[-.08, 0, .08])

@pytest.fixture(scope="module")
def lookup():
    from volatility import build_volatility_lookup
    params = HestonParams(kappa=2, theta=.04, xi=.4, rho=-.7, v0=.04)
    market = MarketDynamics(measure="Q", spot0=100, risk_free_rate=.10)
    grid = dict(
        maturities_days=[21, 63, 126],
        log_moneyness=[-.1, -.05, 0, .05, .1],
        variance_nodes=[0, .002, .005, .01, .02, .04, .06, .1, .2, .4, .6],
    )
    return params, market, build_volatility_lookup(params, market, **grid)


def test_lookup_path_shape_and_same_initial_state(lookup):
    params, market, table = lookup
    from heston import simulate_heston
    paths = simulate_heston(params, market, n_paths=5, n_steps=6, seed=9)
    values = table.evaluate(paths.variance)
    assert values.shape == (5, 7, 3, 5)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0)
    initial = heston_volatility_surface(
        params, market, maturities_days=[21, 63, 126],
        log_moneyness=[-.1, -.05, 0, .05, .1],
    )
    np.testing.assert_allclose(values[0, 0], initial.implied_volatility, atol=1e-12)
    np.testing.assert_allclose(values[:, 0], values[0, 0][None] * np.ones((5, 1, 1)), atol=1e-12)


def test_lookup_interpolation_against_independent_fourier_prices(lookup):
    params, market, table = lookup
    for v in (.003, .015, .035, .075, .14, .31):
        expected = heston_volatility_surface(
            params, market, variance=v, maturities_days=[21, 63, 126],
            log_moneyness=[-.1, -.05, 0, .05, .1],
        )
        approx = table.evaluate(np.array(v))
        # Absolute IV accuracy: 0.0005 = 5 basis points in volatility units.
        np.testing.assert_allclose(approx, expected.implied_volatility, atol=5e-4)


def test_lookup_moneyness_scaling_and_black_scholes_repricing(lookup):
    params, market, table = lookup
    v = .035
    sigma = table.evaluate(np.array(v))
    tenor_idx, moneyness_idx = 1, 3  # τ=63d, log(K/F)=0.05
    tau = table.maturities_days[tenor_idx] / 252
    k = table.log_moneyness[moneyness_idx]
    for spot in (80.0, 130.0):
        strike = spot * np.exp(.1 * tau + k)
        direct = heston_price(params, market, strike=strike, maturity=tau,
                              option_type="call", spot=spot, variance=v)
        bs = black_scholes_price(spot, strike, tau,
                                 sigma[tenor_idx, moneyness_idx], rate=.1)
        assert bs == pytest.approx(direct, abs=0.002)


def test_lookup_rejects_extrapolation_and_invalid_grids(lookup):
    from volatility import build_volatility_lookup
    params, market, table = lookup
    for bad in (-.001, float("nan"), .601):
        with pytest.raises(ValueError, match="variance|Variance"):
            table.evaluate(np.array([bad]))
    with pytest.raises(ValueError, match="variance_nodes"):
        build_volatility_lookup(
            params, market, maturities_days=[21], log_moneyness=[0],
            variance_nodes=[.04, .03, .1],
        )
    with pytest.raises(ValueError, match="Q"):
        build_volatility_lookup(
            params, MarketDynamics(measure="P", spot0=100, risk_free_rate=.1,
                                   physical_total_return=.12),
            maturities_days=[21], log_moneyness=[0],
            variance_nodes=[0, .04, .1],
        )
