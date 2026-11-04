"""Independent hand-computed identities for the three-forward hedge convention."""
import math

import numpy as np
import pytest

from heston import HestonPaths, HestonParams, MarketDynamics, simulate_heston
from portfolio import fixed_eniac_book, expiry_cashflows
from hedging import (BookPaths, execute_forward_trades, forward_prices,
                     initial_forward_positions, simulate_forward_hedge,
                     value_book_paths)


def _paths(spots, measure="Q"):
    spot = np.asarray(spots, dtype=float)
    variance = np.full_like(spot, 0.04)
    return HestonPaths(np.arange(spot.shape[1]) / 250, spot,
                       variance, variance.copy(), measure)


def _marks(paths, values=None, settlements=None):
    zero = np.zeros_like(paths.spot)
    return BookPaths(
        zero.copy() if values is None else np.asarray(values, dtype=float),
        zero.copy(), zero.copy(), zero.copy(),
        zero.copy() if settlements is None else np.asarray(settlements, dtype=float),
    )


def _simulate(spots, initial_positions, actions=None, **kwargs):
    paths = _paths(spots)
    n_paths, n_times = paths.spot.shape
    return simulate_forward_hedge(
        paths, _marks(paths), MarketDynamics("Q", 100, 0.0),
        initial_positions=np.asarray(initial_positions, dtype=float),
        actions=(np.zeros((n_paths, n_times-1, 3)) if actions is None else actions),
        expiries_days=(1, 2, 3),
        **kwargs,
    )


def test_fixed_eniac_book_maturities_strikes_and_signed_payoffs():
    book = fixed_eniac_book()
    assert book.n_contracts == 9
    assert [(c.maturity_days, c.strike, c.quantity) for c in book.contracts] == [
        (t, k, -1) for t in (42, 84, 126) for k in (96, 100, 104)
    ]
    spot = np.full((1, 127), 110.0)
    cf = expiry_cashflows(book, spot)
    np.testing.assert_allclose(cf[0, [42, 84, 126]], -np.array([30., 30., 30.]))
    assert np.count_nonzero(cf) == 3


def test_forward_prices_have_correct_time_units():
    r = math.log(1.10)
    f = forward_prices(np.array([100.0]), 0, (42, 84, 126), r)
    np.testing.assert_allclose(f[0], 100 * (1.10 ** (np.array([42, 84, 126])/250)))
    np.testing.assert_allclose(forward_prices(np.array([123.]), 42, (42,), r), [[123.]])


def test_open_close_costs_and_positions():
    result = _simulate([[100., 100., 100., 100.]], [[1., 0., 0.]],
                       spread_half_pcts=(.01, .02, .03))
    # Open: 1 + 0.5; forced close at first maturity: 1 + 0.5.
    assert result.initial_cost[0] == pytest.approx(1.5)
    assert result.transaction_cost[0] == pytest.approx(3.0)
    assert result.spread_cost[0] == pytest.approx(2.0)
    assert result.fixed_cost[0] == pytest.approx(1.0)
    assert result.trades[0] == 2
    assert result.turnover[0] == pytest.approx(2.)
    np.testing.assert_allclose(result.position[0, :, 0], [1, 0, 0, 0])
    assert result.terminal_pnl[0] == pytest.approx(-3.)


def test_current_action_earns_current_interval_not_next():
    # With no initial position, the t=0 buy must earn S1-S0 (10), too.
    actions = np.zeros((1, 3, 3))
    actions[0, 0, 2] = 1.0
    result = _simulate([[100., 110., 120., 130.]], [[0., 0., 0.]], actions,
                       spread_half_pcts=(0., 0., 0.), fixed_cost=0.)
    np.testing.assert_allclose(result.hedge_pnl, [[10., 10., 10.]])
    assert result.terminal_pnl[0] == pytest.approx(30.)


def test_each_bucket_stops_on_its_own_maturity():
    result = _simulate([[100., 110., 120., 130.]], [[1., 1., 1.]],
                       spread_half_pcts=(0., 0., 0.), fixed_cost=0.)
    np.testing.assert_allclose(result.hedge_pnl, [[30., 20., 10.]])
    np.testing.assert_allclose(result.position[0],
                               [[1,1,1], [0,1,1], [0,0,1], [0,0,0]])
    assert result.terminal_pnl[0] == pytest.approx(60.)


def test_min_trade_bounds_and_costs_on_executed_action_only():
    pos = np.array([[0., 2.95, 1.]])
    action = np.array([[0.005, 1., -2.]])
    mid = np.array([[100., 200., 300.]])
    new, actual, spread, fixed = execute_forward_trades(
        pos, action, mid, np.array([True, True, False]),
        spread_half_pcts=(.01, .02, .03), min_trade=.01,
        max_trade_increment=1., max_position=3., fixed_cost=.5)
    np.testing.assert_allclose(new[0], [0., 3., 1.])
    np.testing.assert_allclose(actual[0], [0., .05, 0.])
    assert spread[0] == pytest.approx(.2)
    assert fixed[0] == pytest.approx(.5)


def test_short_option_premium_settlement_and_no_hedge():
    paths = _paths([[100., 105., 108., 110.]])
    marks = _marks(paths, values=[[-12., -7., -5., 0.]],
                   settlements=[[0., 0., 0., -8.]])
    result = simulate_forward_hedge(
        paths, marks, MarketDynamics("Q", 100, 0),
        initial_positions=np.zeros((1, 3)), actions=np.zeros((1, 3, 3)),
        expiries_days=(1, 2, 3), spread_half_pcts=(0., 0., 0.), fixed_cost=0.)
    # Receive 12 initially for the written option; pay 8 on expiry.
    assert result.wealth[0, 0] == pytest.approx(0.)
    assert result.terminal_pnl[0] == pytest.approx(4.)


def test_Q_only_and_bad_shapes_rejected():
    paths = _paths([[100., 100., 100., 100.]])
    kwargs = dict(initial_positions=np.zeros((1, 3)), actions=np.zeros((1, 3, 3)),
                  expiries_days=(1, 2, 3))
    with pytest.raises(ValueError, match="Q"):
        simulate_forward_hedge(paths, _marks(paths),
                               MarketDynamics("P", 100, 0, physical_total_return=0), **kwargs)
    with pytest.raises(ValueError, match="shape"):
        simulate_forward_hedge(paths, _marks(paths),
                               MarketDynamics("Q", 100, 0),
                               initial_positions=np.zeros((1,3)),
                               actions=np.zeros((3, 3)), expiries_days=(1,2,3))


def test_integrated_heston_Q_initial_delta_and_marking():
    book = fixed_eniac_book()
    p = HestonParams(kappa=1., theta=.2, xi=0., rho=.7, v0=.04)
    market = MarketDynamics("Q", spot0=100., risk_free_rate=math.log(1.10))
    paths = simulate_heston(p, market, n_paths=1, n_steps=126,
                            trading_days_per_year=250, seed=99)
    marks = value_book_paths(book, paths, p, market, trading_days_per_year=250)
    positions = initial_forward_positions(
        book, paths, p, market, trading_days_per_year=250)
    assert positions.shape == (1, 3)
    assert np.all(positions >= 0) and np.all(positions <= 3)
    assert np.isfinite(marks.value).all()
    result = simulate_forward_hedge(
        paths, marks, market, initial_positions=positions,
        actions=np.zeros((1, 126, 3)), trading_days_per_year=250)
    assert np.isfinite(result.terminal_pnl).all()
    assert np.count_nonzero(result.position[:, -1, :]) == 0
    assert np.count_nonzero(marks.settlements) == 3


def test_initial_cost_is_financed_from_t0_and_expiry_close_is_charged():
    r = math.log(1.10)
    paths = _paths([[100., 100., 100., 100.]])
    res = simulate_forward_hedge(
        paths, _marks(paths), MarketDynamics("Q", 100., r),
        initial_positions=np.array([[1., 0., 0.]]),
        actions=np.zeros((1, 3, 3)), expiries_days=(1, 2, 3),
        spread_half_pcts=(.01, .02, .03),
        fixed_cost=.5, trading_days_per_year=250,
    )
    f0 = 100 * math.exp(r / 250)
    open_cost = .01 * f0 + .5
    close_cost = .01 * 100 + .5
    # Buy initially, fund spread/fixed for one day, realize F1-F0,
    # and pay mandatory close exactly once.
    expected_wealth_day1 = -open_cost * math.exp(r / 250) + (100 - f0) - close_cost
    assert res.wealth[0, 0] == pytest.approx(-open_cost)
    assert res.wealth[0, 1] == pytest.approx(expected_wealth_day1)
    assert res.trades[0] == 2
