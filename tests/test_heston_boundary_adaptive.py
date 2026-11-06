"""Regression tests for Heston Fourier inversion near the variance boundary."""
import math

import numpy as np
import pytest

from heston import HestonParams, MarketDynamics, simulate_heston
from pricing import heston_greeks_batch, heston_greeks_strikes_batch
from portfolio import fixed_eniac_book
from hedging import value_book_paths


@pytest.fixture
def market():
    return MarketDynamics(measure='Q', spot0=100., risk_free_rate=math.log(1.1))


@pytest.mark.parametrize('xi', [0.25, 0.30, 0.40])
def test_failure_near_expiry_at_zero_variance_has_arbitrage_consistent_price(xi, market):
    params = HestonParams(kappa=1., theta=.04, xi=xi, rho=.7, v0=.04)
    spot, strike, maturity = 94.258753, 104., 1/250
    actual = heston_greeks_batch(
        params, market, strike=strike, maturity=maturity,
        spots=np.array([spot]), variances=np.array([0.]),
    )
    # S/K ~ 0.906 and virtually zero short-dated future variance.
    assert 0 <= actual.price[0] < 1e-7
    assert 0 <= actual.delta[0] < 1e-7
    assert 0 <= actual.gamma[0] < 1e-7
    assert np.isfinite(actual.variance_sensitivity[0])


@pytest.mark.parametrize('xi,expected_price,expected_delta', [
    (0.25, .04337951860, .75482114106),
    (0.40, .041897220, .807349306),
])
def test_at_money_extreme_boundary_retains_time_value_and_delta(xi, expected_price, expected_delta, market):
    params = HestonParams(1., .04, xi, .7, .04)
    actual = heston_greeks_batch(
        params, market, strike=100., maturity=1/250,
        spots=np.array([100.]), variances=np.array([0.]),
    )
    assert actual.price[0] == pytest.approx(expected_price, rel=1e-5, abs=5e-8)
    assert actual.delta[0] == pytest.approx(expected_delta, rel=1e-5, abs=5e-8)
    assert actual.gamma[0] > 0


@pytest.mark.parametrize('xi', [0.25, 0.4])
def test_multi_strike_near_boundary_agrees_with_single_strike(xi, market):
    params = HestonParams(1., .04, xi, .7, .04)
    spots = np.array([94.258753, 98., 99., 100., 104.])
    variance = np.zeros(len(spots))
    strikes = np.array([96., 100., 104.])
    multi = heston_greeks_strikes_batch(
        params, market, strikes=strikes, maturity=1/250,
        spots=spots, variances=variance,
    )
    for j, strike in enumerate(strikes):
        one = heston_greeks_batch(
            params, market, strike=strike, maturity=1/250,
            spots=spots, variances=variance,
        )
        for field in ('price', 'delta', 'gamma', 'variance_sensitivity'):
            np.testing.assert_allclose(
                getattr(multi, field)[:, j], getattr(one, field),
                rtol=1e-8, atol=1e-7,
            )
    assert np.all(multi.price >= 0)
    assert np.all(multi.delta >= 0)
    assert np.all(multi.gamma >= 0)


def test_path_valuation_with_high_xi_and_zero_variance(market):
    params = HestonParams(1., .04, .4, .7, .04)
    paths = simulate_heston(params, market, n_paths=40, n_steps=126,
                            trading_days_per_year=250, seed=2202)
    marks = value_book_paths(
        fixed_eniac_book(), paths, params, market,
        trading_days_per_year=250, include_contract_prices=True,
        include_bucket_deltas=True,
    )
    for x in (marks.value, marks.delta, marks.gamma,
              marks.variance_sensitivity, marks.contract_prices, marks.bucket_delta):
        assert np.all(np.isfinite(x))
    assert np.all(marks.contract_prices >= 0)
