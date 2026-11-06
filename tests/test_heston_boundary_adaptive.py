"""Regression tests for Heston Fourier inversion near the variance boundary."""
import math

import numpy as np
import pytest

from heston import HestonParams, MarketDynamics, simulate_heston
from pricing import (heston_greeks_batch, heston_greeks_strikes_batch,
                     _adaptive_heston_greeks_near_boundary)
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


@pytest.mark.parametrize("xi", [0.25, 0.40])
def test_itm_short_expiry_small_variance_uses_otm_put_parity(xi, market):
    """A too-large oscillatory cutoff previously halved the ITM call value.

    The xi=.40 case used to return price=5.142131 while its exact
    no-arbitrage lower bound was approximately 10.284262.
    """
    params = HestonParams(1.0, 0.04, xi, 0.7, 0.04)
    spot, strike, maturity, variance = 106.24767, 96.0, 1 / 250, 0.0001084851
    call = heston_greeks_batch(
        params, market, strike=strike, maturity=maturity,
        spots=np.array([spot]), variances=np.array([variance]),
    )
    put = heston_greeks_batch(
        params, market, strike=strike, maturity=maturity,
        spots=np.array([spot]), variances=np.array([variance]),
        option_type="put",
    )
    parity = spot - strike * math.exp(-market.risk_free_rate * maturity)
    assert call.price[0] >= parity - 1e-7
    assert call.price[0] == pytest.approx(parity, abs=1e-7)
    assert call.delta[0] == pytest.approx(1.0, abs=1e-7)
    assert call.gamma[0] >= 0.0
    assert put.price[0] >= 0.0
    assert call.price[0] - put.price[0] == pytest.approx(parity, abs=1e-8)
    assert call.delta[0] - put.delta[0] == pytest.approx(1.0, abs=1e-8)
    assert call.gamma[0] == pytest.approx(put.gamma[0], abs=1e-8)
    assert np.isfinite(call.variance_sensitivity[0])


@pytest.mark.parametrize("xi", [0.25, 0.40])
def test_itm_multistrike_near_boundary_is_arbitrage_consistent(xi, market):
    params = HestonParams(1.0, 0.04, xi, 0.7, 0.04)
    spot, variance, maturity = 106.24767, 0.0001084851, 1 / 250
    strikes = np.array([96.0, 100.0, 104.0])
    result = heston_greeks_strikes_batch(
        params, market, strikes=strikes, maturity=maturity,
        spots=np.array([spot]), variances=np.array([variance]),
    )
    lower = np.maximum(spot - strikes * math.exp(-market.risk_free_rate * maturity), 0.0)
    assert np.all(result.price[0] >= lower - 1e-7)
    assert np.all(result.delta[0] >= -1e-7)
    assert np.all(result.delta[0] <= 1.0 + 1e-7)
    assert np.all(result.gamma[0] >= 0.0)
    assert np.all(np.isfinite(result.variance_sensitivity))
    assert result.price[0, 0] == pytest.approx(lower[0], abs=1e-7)


def test_strike_batch_high_order_removes_hidden_aliasing_at_small_variance(market):
    """The 192-node result can be within bounds yet have a wrong price."""
    params = HestonParams(1.0, 0.04, 0.4, 0.7, 0.04)
    spots = np.array([94.0, 100.0, 106.0])
    strikes = np.array([96.0, 100.0, 104.0])
    result = heston_greeks_strikes_batch(
        params, market, strikes=strikes, maturity=1/250,
        spots=spots, variances=np.full(len(spots), 0.001),
    )
    for i, spot in enumerate(spots):
        for j, strike in enumerate(strikes):
            ref = _adaptive_heston_greeks_near_boundary(
                params, market, spot=float(spot), variance=0.001,
                strike=float(strike), maturity=1/250,
            )
            assert result.price[i, j] == pytest.approx(ref.price, abs=2e-6)
            assert result.delta[i, j] == pytest.approx(ref.delta, abs=2e-6)
            assert result.gamma[i, j] == pytest.approx(ref.gamma, abs=2e-6)


def test_far_otm_short_expiry_with_zero_variance_stays_nonnegative(market):
    params = HestonParams(1.0, 0.04, 0.4, 0.7, 0.04)
    result = heston_greeks_batch(
        params, market, strike=104., maturity=2/250,
        spots=np.array([84.890705]), variances=np.array([0.]),
    )
    assert 0 <= result.price[0] < 1e-6
    assert 0 <= result.delta[0] < 1e-5
    assert 0 <= result.gamma[0] < 1e-5
