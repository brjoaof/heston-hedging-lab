"""Numerical regression for shared Heston Fourier characteristic terms."""

import math

import numpy as np
import pytest

from hedging import value_book_paths
from portfolio import expiry_cashflows
from heston import HestonParams, MarketDynamics, simulate_heston
from portfolio import fixed_eniac_book
from pricing import heston_greeks_batch, heston_greeks_strikes_batch


@pytest.mark.parametrize("xi", [0.0, 0.05, 0.2, 0.4])
@pytest.mark.parametrize("maturity", [1 / 250, 10 / 250, 42 / 250, 126 / 250])
def test_shared_strike_quadrature_matches_legacy(xi, maturity):
    market = MarketDynamics(measure="Q", spot0=100, risk_free_rate=math.log(1.1))
    params = HestonParams(kappa=1.0, theta=0.2, xi=xi, rho=0.7, v0=0.2)
    spots = np.array([68., 83., 94., 99., 100., 105., 117., 139.])
    variances = np.array([0.005, 0.03, 0.1, 0.18, 0.2, 0.3, 0.52, 0.9])
    strikes = np.array([96., 100., 104.])
    for option_type in ("call", "put"):
        joint = heston_greeks_strikes_batch(
            params, market, strikes=strikes, maturity=maturity,
            spots=spots, variances=variances, option_type=option_type,
        )
        individual = [
            heston_greeks_batch(
                params, market, strike=float(strike), maturity=maturity,
                spots=spots, variances=variances, option_type=option_type,
            )
            for strike in strikes
        ]
        for field in ("price", "delta", "gamma", "variance_sensitivity", "initial_volatility_vega"):
            expect = np.column_stack([getattr(x, field) for x in individual])
            np.testing.assert_allclose(getattr(joint, field), expect, rtol=1e-9, atol=2e-9)


@pytest.mark.parametrize("xi", [0.0, 0.05, 0.4])
def test_joint_marking_preserves_all_option_and_book_fields(xi):
    market = MarketDynamics(measure="Q", spot0=100, risk_free_rate=math.log(1.1))
    params = HestonParams(kappa=1, theta=0.2, xi=xi, rho=0.7, v0=0.2)
    paths = simulate_heston(
        params, market, n_paths=4, n_steps=126,
        trading_days_per_year=250, seed=1234,
    )
    opts = dict(trading_days_per_year=250, include_contract_prices=True, include_bucket_deltas=True)
    book = fixed_eniac_book()
    npaths, ntimes = paths.spot.shape
    cumul = [np.zeros_like(paths.spot) for _ in range(4)]
    original_contract = np.zeros((npaths,ntimes,book.n_contracts))
    original_bucket = np.zeros((npaths,ntimes,3))
    mats = (42,84,126)
    for i, contract in enumerate(book.contracts):
        for day in range(contract.maturity_days):
            g = heston_greeks_batch(params, market, strike=contract.strike,
                 maturity=(contract.maturity_days-day)/250,
                 spots=paths.spot[:,day], variances=paths.variance[:,day],
                 option_type=contract.option_type)
            amt = contract.quantity*contract.multiplier
            original_contract[:,day,i] = g.price
            original_bucket[:,day,mats.index(contract.maturity_days)] += amt*g.delta
            for a,f in zip(cumul,("price","delta","gamma","variance_sensitivity")):
                a[:,day] += amt*getattr(g,f)
    reference = dict(zip(("value","delta","gamma","variance_sensitivity"),cumul))
    reference.update(settlements=expiry_cashflows(book, paths.spot),
                     contract_prices=original_contract, bucket_delta=original_bucket)
    optimized = value_book_paths(fixed_eniac_book(), paths, params, market, **opts)
    for field in ("value", "delta", "gamma", "variance_sensitivity", "settlements", "contract_prices", "bucket_delta"):
        np.testing.assert_allclose(getattr(optimized,field), reference[field], rtol=1e-10, atol=2e-9)


def test_multi_strike_input_validation():
    m = MarketDynamics(measure="Q", spot0=100, risk_free_rate=0.05)
    p = HestonParams(kappa=1, theta=0.2, xi=0.2, rho=0.7, v0=0.2)
    with pytest.raises(ValueError):
        heston_greeks_strikes_batch(p, m, strikes=np.array([]), maturity=0.1,
                                   spots=np.array([100.]), variances=np.array([0.2]))
    with pytest.raises(ValueError):
        heston_greeks_strikes_batch(p, m, strikes=np.array([100., -1.]), maturity=0.1,
                                   spots=np.array([100.]), variances=np.array([0.2]))
