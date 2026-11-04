import numpy as np
import pytest

from heston import HestonParams, MarketDynamics, simulate_heston
from portfolio import EuropeanOption, OptionBook
from pricing import heston_greeks, heston_greeks_batch
from hedging import value_book_paths, simulate_hedge, summarize_hedge


def market():
    return MarketDynamics("Q", 100.0, 0.0)


def params():
    return HestonParams(kappa=2, theta=.04, xi=.4, rho=-.7, v0=.04)


def test_batch_pricer_agrees_with_adaptive_reference():
    p, m = params(), market()
    for maturity in (1/252, 21/252, 126/252):
        s = np.array([90., 100., 110.])
        v = np.array([.03, .04, .08])
        batch = heston_greeks_batch(p, m, strike=100., maturity=maturity, spots=s, variances=v)
        for j in range(len(s)):
            ref = heston_greeks(p, m, strike=100., maturity=maturity, spot=s[j], variance=v[j])
            assert batch.price[j] == pytest.approx(ref.price, abs=5e-4)
            assert batch.delta[j] == pytest.approx(ref.delta, abs=5e-3)
            assert batch.gamma[j] == pytest.approx(ref.gamma, abs=5e-3)
            assert batch.variance_sensitivity[j] == pytest.approx(ref.variance_sensitivity, abs=5e-2)


def test_batch_put_call_parity_and_no_arbitrage():
    p, m = params(), market()
    s = np.array([80, 100, 120])
    v = np.array([.01, .04, .08])
    t = .2
    c = heston_greeks_batch(p,m,strike=100,maturity=t,spots=s,variances=v)
    put = heston_greeks_batch(p,m,strike=100,maturity=t,spots=s,variances=v,option_type="put")
    np.testing.assert_allclose(c.price-put.price, s-100, atol=1e-8)
    np.testing.assert_allclose(c.delta-put.delta, 1., atol=1e-8)
    assert np.all(c.price >= 0)
    assert np.all(put.price >= 0)


def test_batch_deterministic_volatility_is_black_scholes():
    p = HestonParams(2,.04,0,-.7,.04)
    m = market()
    batch = heston_greeks_batch(p,m,strike=100,maturity=.25,spots=np.array([100.]),variances=np.array([.04]))
    ref = heston_greeks(p,m,strike=100,maturity=.25)
    assert batch.price[0] == pytest.approx(ref.price,abs=1e-8)
    assert batch.delta[0] == pytest.approx(ref.delta,abs=2e-5)


def test_expiry_is_paid_once_and_absent_from_book_value():
    p = HestonParams(2, .04, 0, 0, .04)
    paths = simulate_heston(p, market(), n_paths=8, n_steps=3,seed=2)
    book = OptionBook("case",(EuropeanOption("a", 2, 100, 1), EuropeanOption("b", -1, 100, 3)))
    result = value_book_paths(book,paths,p,market())
    # The first option is no longer held after day 1.
    one = heston_greeks_batch(p,market(),strike=100,maturity=2/252,
                             spots=paths.spot[:,1],variances=paths.variance[:,1])
    np.testing.assert_allclose(result.value[:,1],-one.price,atol=1e-9)
    np.testing.assert_allclose(result.value[:,3],0)
    np.testing.assert_allclose(result.settlements[:,1],2*np.maximum(paths.spot[:,1]-100,0))
    np.testing.assert_allclose(result.settlements[:,3],-np.maximum(paths.spot[:,3]-100,0))


def test_self_financing_book_and_hedge_accounting_without_costs():
    p = HestonParams(2, 0, 0, 0, 0)
    m = MarketDynamics("Q", 100, 0)
    paths = simulate_heston(p,m,n_paths=4,n_steps=2,seed=3)
    book = OptionBook("deep_itm",(EuropeanOption("a",1,80,2),))
    v = value_book_paths(book,paths,p,m)
    assert np.allclose(v.delta[:,:2],1.)
    for name in ("no_hedge","delta_daily","delta_periodic","delta_band"):
        z=simulate_hedge(paths,v,m,strategy=name,proportional_cost=0,
                         fixed_cost=0,band_width=.5)
        np.testing.assert_allclose(z.terminal_pnl,0,atol=1e-10)
        np.testing.assert_allclose(z.position[:,-1],0)
        assert np.all(z.transaction_cost == 0)


def test_costs_and_no_trade_region():
    p=HestonParams(2, 0, 0, 0, 0)
    m=market()
    paths=simulate_heston(p,m,n_paths=2,n_steps=2,seed=3)
    b=OptionBook("itm",(EuropeanOption("a",1,80,2),))
    v=value_book_paths(b,paths,p,m)
    band=simulate_hedge(paths,v,m,strategy="delta_band",band_width=2.,fixed_cost=.5)
    assert np.all(band.trades == 0)
    delta=simulate_hedge(paths,v,m,strategy="delta_daily",proportional_cost=.01,fixed_cost=.5)
    # Position -1 bought at t0, then closed at maturity: two trades.
    assert np.all(delta.trades == 2)
    np.testing.assert_allclose(delta.transaction_cost, 2*(.5+1.),atol=1e-12)
    np.testing.assert_allclose(delta.terminal_pnl, -delta.transaction_cost, atol=1e-12)


def test_summarize_tail_losses():
    from hedging import HedgeResults
    losses=np.arange(1,101,dtype=float)
    pnl=-losses
    r=HedgeResults(np.zeros((100,1)),pnl,np.zeros(100),np.zeros(100),np.zeros(100),np.zeros((100,1)))
    sm=summarize_hedge(r)
    assert sm["mean_pnl"]==-50.5
    assert sm["cvar95_loss"]==pytest.approx(98.)


def test_cvar_handles_mass_at_quantile():
    # One severe loss out of 64 trajectories; the other losses are zero.
    # The worst 5% tail has mass 3.2 observations, so CVaR = 100/3.2.
    from hedging import HedgeResults
    losses = np.zeros(64)
    losses[0] = 100.0
    r = HedgeResults(
        np.zeros((64, 1)), -losses, np.zeros(64), np.zeros(64),
        np.zeros(64), np.zeros((64, 1)),
    )
    summary = summarize_hedge(r)
    assert summary["q95_loss"] == pytest.approx(0.0)
    assert summary["cvar95_loss"] == pytest.approx(31.25)


def test_mandatory_final_liquidation_overrides_min_trade():
    # A tiny residual position after a larger intermediate rebalance must
    # still be liquidated at the evaluation horizon.
    from heston import HestonPaths
    from hedging import BookPaths
    spot = np.full((1, 3), 100.0)
    zeros = np.zeros_like(spot)
    paths = HestonPaths(
        np.arange(3) / 252, spot, np.full_like(spot, 0.04),
        np.full_like(spot, 0.04), "P",
    )
    data = BookPaths(
        value=zeros, delta=np.array([[1.0, 0.005, 0.005]]),
        gamma=zeros, variance_sensitivity=zeros, settlements=zeros,
    )
    physical_market = MarketDynamics("P", 100.0, 0.0, physical_total_return=0.0)
    result = simulate_hedge(
        paths, data, physical_market, strategy="delta_daily",
        proportional_cost=0.01, fixed_cost=0.5, min_trade=0.01,
    )
    assert result.position[0, -1] == pytest.approx(0.0)
    assert result.trades[0] == 3
