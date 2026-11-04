import numpy as np
import pytest

from heston import HestonParams, MarketDynamics
from portfolio import EuropeanOption, OptionBook
from pricing import black_scholes_price, heston_price, heston_greeks, price_book


@pytest.fixture
def setup():
    return (
        HestonParams(kappa=2, theta=.04, xi=.4, rho=-.7, v0=.04),
        MarketDynamics(measure="Q", spot0=100, risk_free_rate=.10),
    )


def test_put_call_parity(setup):
    p, market = setup
    for strike in (85., 100., 115.):
        for maturity in (1 / 252, 63 / 252, .5):
            call = heston_price(p, market, strike=strike, maturity=maturity)
            put = heston_price(p, market, strike=strike, maturity=maturity, option_type="put")
            parity = 100 - strike * np.exp(-.10 * maturity)
            assert call - put == pytest.approx(parity, abs=1e-7)


def test_zero_vol_of_vol_exact_black_scholes(setup):
    _, market = setup
    p = HestonParams(kappa=2, theta=.04, xi=0, rho=-.7, v0=.04)
    for option_type in ("call", "put"):
        for strike in (85, 100, 115):
            result = heston_price(p, market, strike=strike, maturity=.5, option_type=option_type)
            benchmark = black_scholes_price(100, strike, .5, .2, .10, 0, option_type)
            assert result == pytest.approx(benchmark, abs=1e-10)


def test_zero_vol_of_vol_mean_reversion(setup):
    _, market = setup
    p = HestonParams(kappa=2, theta=.09, xi=0, rho=0, v0=.01)
    T=.7
    integrated=.09*T+(.01-.09)*(1-np.exp(-2*T))/2
    expected = black_scholes_price(100, 100, T, np.sqrt(integrated/T), .1)
    assert heston_price(p, market, strike=100, maturity=T) == pytest.approx(expected, abs=1e-10)


def test_prices_decrease_with_strike_and_obey_bounds(setup):
    p, market = setup
    prices = [heston_price(p, market, strike=k, maturity=.5) for k in (85., 95., 100., 105., 115.)]
    assert all(a > b >= 0 for a, b in zip(prices, prices[1:]))
    for k, c in zip((85, 95, 100, 105, 115), prices):
        assert c >= max(100-k*np.exp(-.05),0) - 1e-8
        assert c <= 100


def test_maturity_zero_payoff_and_nonnegative_short_maturity(setup):
    p, market = setup
    assert heston_price(p, market, strike=105, maturity=0, spot=110) == pytest.approx(5)
    assert heston_price(p, market, strike=105, maturity=0, spot=110, option_type="put") == pytest.approx(0)
    assert heston_price(p, market, strike=105, maturity=1/252) >= 0


def test_spot_delta_gamma_and_variance_sensitivity(setup):
    p, market = setup
    g = heston_greeks(p, market, strike=100, maturity=.25)
    assert 0 < g.delta < 1
    assert g.gamma > 0
    assert g.variance_sensitivity > 0
    assert g.initial_volatility_vega == pytest.approx(.4*g.variance_sensitivity)
    g_put = heston_greeks(p, market, strike=100, maturity=.25, option_type="put")
    assert g.delta - g_put.delta == pytest.approx(1, abs=2e-6)
    assert g.gamma == pytest.approx(g_put.gamma, abs=2e-6)


def test_book_is_linear_for_contract_prices_not_hedging_costs(setup):
    p, market = setup
    b = OptionBook("book", (
        EuropeanOption("a", 2, 100, 63, "call"),
        EuropeanOption("b", -1, 105, 21, "put"),
    ))
    g = price_book(b, p, market, trading_days_per_year=252)
    a = heston_greeks(p, market, strike=100, maturity=63/252)
    other = heston_greeks(p, market, strike=105, maturity=21/252, option_type="put")
    assert g.price == pytest.approx(2*a.price-other.price)
    assert g.delta == pytest.approx(2*a.delta-other.delta)
    assert g.gamma == pytest.approx(2*a.gamma-other.gamma)
    after_21_days = price_book(b, p, market, day=21)
    a_after = heston_greeks(p, market, strike=100, maturity=42/252)
    assert after_21_days.price == pytest.approx(2*a_after.price)
    assert price_book(b, p, market, day=63).price == 0.


def test_pricing_requires_q_and_valid_inputs(setup):
    p, m = setup
    phys = MarketDynamics(measure="P", spot0=100, risk_free_rate=.1, physical_total_return=.15)
    with pytest.raises(ValueError, match="risk-neutral"):
        heston_price(p, phys, strike=100, maturity=.5)
    with pytest.raises(ValueError, match="positive"):
        heston_price(p, m, strike=-100, maturity=.5)
    with pytest.raises(ValueError, match="nonnegative"):
        heston_price(p, m, strike=100, maturity=-.5)
