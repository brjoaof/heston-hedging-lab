"""European option pricing under risk-neutral Heston dynamics.

Heston (1993) characteristic-function formula, evaluated with SciPy quadrature.
The Greeks use finite differences and are suitable as numerical references,
not yet as a high-throughput engine for RL simulation.
"""

from dataclasses import dataclass
import math
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from heston import HestonParams, MarketDynamics
from portfolio import OptionBook


def black_scholes_price(
    spot: float, strike: float, maturity: float, volatility: float,
    rate: float = 0.0, dividend: float = 0.0, option_type: str = "call",
) -> float:
    """European Black–Scholes price; used as an independent limiting benchmark."""
    if not (math.isfinite(spot) and spot > 0 and math.isfinite(strike) and strike > 0):
        raise ValueError("spot and strike must be finite and positive")
    if not (math.isfinite(maturity) and maturity >= 0 and math.isfinite(volatility) and volatility >= 0):
        raise ValueError("maturity and volatility must be finite and nonnegative")
    if not (math.isfinite(rate) and math.isfinite(dividend)):
        raise ValueError("rate and dividend must be finite")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    if maturity == 0:
        return float(max((spot - strike) if option_type == "call" else (strike - spot), 0.0))
    s_disc = spot * math.exp(-dividend * maturity)
    k_disc = strike * math.exp(-rate * maturity)
    if volatility == 0:
        return float(max(s_disc - k_disc if option_type == "call" else k_disc - s_disc, 0.0))

    stdev = volatility * math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - dividend + 0.5 * volatility**2) * maturity) / stdev
    d2 = d1 - stdev
    if option_type == "call":
        return float(s_disc * norm.cdf(d1) - k_disc * norm.cdf(d2))
    return float(k_disc * norm.cdf(-d2) - s_disc * norm.cdf(-d1))


def _heston_cf(
    u: complex | np.ndarray, *, spot: float, variance: float,
    maturity: float, params: HestonParams, rate: float, dividend: float,
) -> complex | np.ndarray:
    """Characteristic function of log(S_T) under Q (stable g-formulation)."""
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    iu = 1j * u
    b = kappa - rho * xi * iu
    d = np.sqrt(b**2 + xi**2 * (u**2 + iu))
    # Avoid subtracting nearly equal complex numbers when xi is small.
    b_minus_d = -xi**2 * (u**2 + iu) / (b + d)
    g = b_minus_d / (b + d)
    exp_dt = np.exp(-d * maturity)
    log_ratio = np.log1p(-g * exp_dt) - np.log1p(-g)
    c = (kappa * theta / xi**2) * (b_minus_d * maturity - 2 * log_ratio)
    d_coef = (b_minus_d / xi**2) * (1 - exp_dt) / (1 - g * exp_dt)
    return np.exp(iu * (np.log(spot) + (rate - dividend) * maturity) + c + d_coef * variance)


def heston_price(
    params: HestonParams, market: MarketDynamics, *,
    strike: float, maturity: float, option_type: str = "call",
    spot: float | None = None, variance: float | None = None,
) -> float:
    """Price a European call/put using Heston's two Fourier probability integrals.

    `spot` and `variance` override the initial state: useful for revaluation
    along simulated paths. `params` and `market` must describe Q dynamics.
    The resulting price is a model price, not an executable bid/ask quote.
    """
    if market.measure != "Q":
        raise ValueError("Heston option prices require risk-neutral (Q) dynamics")
    s = market.spot0 if spot is None else float(spot)
    v = params.v0 if variance is None else float(variance)
    if not math.isfinite(s) or s <= 0 or not math.isfinite(strike) or strike <= 0:
        raise ValueError("spot and strike must be finite and positive")
    if not math.isfinite(v) or v < 0 or not math.isfinite(maturity) or maturity < 0:
        raise ValueError("variance and maturity must be finite and nonnegative")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")

    if maturity == 0:
        return float(max(s - strike if option_type == "call" else strike - s, 0.0))

    r, q = market.risk_free_rate, market.dividend_yield
    discounted_spot = s * math.exp(-q * maturity)
    discounted_strike = strike * math.exp(-r * maturity)

    # xi=0: variance follows a deterministic mean-reverting path. Its
    # integrated variance gives the exact Black–Scholes-equivalent price.
    if params.xi == 0:
        integrated_v = params.theta * maturity + (v - params.theta) * (-math.expm1(-params.kappa * maturity)) / params.kappa
        return black_scholes_price(s, strike, maturity, math.sqrt(max(integrated_v, 0.0) / maturity), r, q, option_type)

    forward = s * math.exp((r - q) * maturity)
    log_strike = math.log(strike)

    def integrand(u: float, j: int) -> float:
        z = u - 1j if j == 1 else u
        cf = _heston_cf(z, spot=s, variance=v, maturity=maturity,
                        params=params, rate=r, dividend=q)
        if j == 1:
            cf /= forward
        return float(np.real(np.exp(-1j * u * log_strike) * cf / (1j * u)))

    p1 = 0.5 + quad(lambda u: integrand(u, 1), 0, np.inf,
                    epsabs=1e-9, epsrel=1e-8, limit=250)[0] / math.pi
    p2 = 0.5 + quad(lambda u: integrand(u, 2), 0, np.inf,
                    epsabs=1e-9, epsrel=1e-8, limit=250)[0] / math.pi
    call = discounted_spot * p1 - discounted_strike * p2
    lower = max(discounted_spot - discounted_strike, 0.0)
    upper = discounted_spot
    # Only compensate for numerical roundoff; never mask a material
    # arbitrage-bound violation caused by an unstable parameter setting.
    tolerance = 1e-8 * max(1.0, s)
    if not math.isfinite(call) or call < lower - tolerance or call > upper + tolerance:
        raise FloatingPointError("Fourier price violated call no-arbitrage bounds")
    call = min(max(call, lower), upper)
    if option_type == "call":
        return float(call)
    return float(call - discounted_spot + discounted_strike)


@dataclass(frozen=True)
class OptionGreeks:
    price: float
    delta: float
    gamma: float
    variance_sensitivity: float   # dPrice / d(current instantaneous variance)
    initial_volatility_vega: float  # dPrice / d(sqrt(current instantaneous variance))


def heston_greeks(
    params: HestonParams, market: MarketDynamics, *,
    strike: float, maturity: float, option_type: str = "call",
    spot: float | None = None, variance: float | None = None,
) -> OptionGreeks:
    """Finite-difference state sensitivities; not implied-volatility vega.

    Delta/Gamma hold the current Heston variance fixed. The variance
    sensitivity holds all model parameters fixed. The last field follows
    by the chain rule for sqrt(v), and is NOT a market-IV vega.
    """
    s = market.spot0 if spot is None else float(spot)
    v = params.v0 if variance is None else float(variance)
    if maturity <= 0:
        raise ValueError("Greeks at expiry are not defined by this routine")
    p0 = heston_price(params, market, strike=strike, maturity=maturity,
                      option_type=option_type, spot=s, variance=v)
    ds = min(s * 0.25, max(1e-4, 1e-3 * s))
    up = heston_price(params, market, strike=strike, maturity=maturity,
                      option_type=option_type, spot=s + ds, variance=v)
    down = heston_price(params, market, strike=strike, maturity=maturity,
                        option_type=option_type, spot=s - ds, variance=v)
    delta = (up - down) / (2 * ds)
    gamma = (up - 2 * p0 + down) / ds**2

    dv = max(1e-5, 1e-3 * v)
    hi = heston_price(params, market, strike=strike, maturity=maturity,
                      option_type=option_type, spot=s, variance=v + dv)
    if v >= dv:
        lo = heston_price(params, market, strike=strike, maturity=maturity,
                          option_type=option_type, spot=s, variance=v - dv)
        sens = (hi - lo) / (2 * dv)
    else:
        sens = (hi - p0) / dv
    return OptionGreeks(
        price=p0,
        delta=float(delta),
        gamma=float(gamma),
        variance_sensitivity=float(sens),
        initial_volatility_vega=float(2 * math.sqrt(v) * sens),
    )


def price_book(
    book: OptionBook, params: HestonParams, market: MarketDynamics, *,
    day: int = 0, spot: float | None = None, variance: float | None = None,
    trading_days_per_year: int = 252,
) -> OptionGreeks:
    """Signed value and sensitivities of contracts still alive at `day`.

    Settled cashflows of expired contracts are NOT included in this value.
    They must be accounted for separately in the hedging cash account.
    """
    if isinstance(day, bool) or not isinstance(day, int) or day < 0:
        raise ValueError("day must be a nonnegative integer")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive")
    totals = np.zeros(5)
    for contract in book.contracts:
        remaining = contract.maturity_days - day
        if remaining <= 0:
            continue
        greek = heston_greeks(
            params, market, strike=contract.strike,
            maturity=remaining / trading_days_per_year,
            option_type=contract.option_type, spot=spot, variance=variance,
        )
        totals += contract.quantity * contract.multiplier * np.array([
            greek.price, greek.delta, greek.gamma,
            greek.variance_sensitivity, greek.initial_volatility_vega,
        ])
    return OptionGreeks(*map(float, totals))


@lru_cache(maxsize=6)
def _legendre_nodes(n: int):
    from numpy.polynomial.legendre import leggauss
    return leggauss(n)



def _adaptive_heston_greeks_near_boundary(
    params: HestonParams, market: MarketDynamics, *,
    spot: float, variance: float, strike: float, maturity: float,
) -> OptionGreeks:
    """Carr-Madan damped Fourier inversion for rare unstable boundary states.

    The standard P1/P2 inversion subtracts two nearly equal large numbers and
    is unreliable for near-zero variance at one-day expiry. This formula
    integrates the out-of-the-money call *directly*, including analytical
    Delta, Gamma and dPrice/dv. QUADPACK's oscillatory integration handles
    the high-frequency short-maturity phase without distorting the Heston
    dynamics or silently replacing the price with intrinsic value.

    This is intentionally a rare-state fallback, not the main path pricer.
    """
    import warnings
    from scipy.integrate import IntegrationWarning, quad

    r, q = market.risk_free_rate, market.dividend_yield
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    if xi <= 0:
        raise ValueError("Adaptive Fourier fallback requires xi>0")
    # C = S exp(-qT) exp(-alpha*k)/pi * integral[...],
    # k = log(K/F), alpha>0, F=S exp((r-q)T).
    alpha = 1.0
    log_moneyness = math.log(strike/spot) - (r-q)*maturity
    amplitude = math.exp(-alpha*log_moneyness) / math.pi
    discount_s = math.exp(-q*maturity)

    def terms(u: float) -> tuple[complex, complex, complex, complex]:
        z = complex(u, -(alpha + 1.0))
        iz = 1j*z
        b = kappa - rho*xi*iz
        d = np.sqrt(b*b + xi*xi*(z*z+iz))
        bmd = -xi*xi*(z*z+iz)/(b+d)
        g = bmd/(b+d)
        e = np.exp(-d*maturity)
        c = (kappa*theta/(xi*xi))*(
            bmd*maturity - 2*(np.log1p(-g*e)-np.log1p(-g))
        )
        D = (bmd/(xi*xi))*(1-e)/(1-g*e)
        cf = np.exp(c + D*variance)
        denominator = alpha + 1j*u
        return (cf/(denominator*(denominator+1)),
                cf/denominator, cf, cf*D/(denominator*(denominator+1)))

    # At v=0 and short T the characteristic function decays only at very
    # high frequencies. Instead of integrating to infinity (which can produce
    # a convincing but wrong answer when k≈0), truncate at an explicit point
    # where the *damped* characteristic function is numerically negligible.
    cutoff = 100.0
    for _ in range(25):
        if abs(terms(cutoff)[2]) < 1e-12:
            break
        cutoff *= 2
    else:
        raise FloatingPointError("Could not bound Heston Fourier integration tail")

    integrals = []
    for field in range(4):
        value = 0.0
        for imaginary in (False, True):
            if imaginary and abs(log_moneyness) < 1e-15:
                continue

            def component(u: float) -> float:
                entry = terms(u)[field]
                return float(entry.imag if imaginary else entry.real)

            with warnings.catch_warnings():
                warnings.simplefilter("error", IntegrationWarning)
                # QUADPACK's finite-interval oscillatory rule remains stable
                # when |k| is small, unlike its infinite-cycle counterpart.
                integral, _ = quad(
                    component, 0, cutoff,
                    weight="sin" if imaginary else "cos",
                    wvar=log_moneyness, epsabs=1e-10, epsrel=1e-10,
                    limit=700,
                )
            value += integral
        integrals.append(value * amplitude)

    call = spot * discount_s * integrals[0]
    delta = discount_s * integrals[1]
    gamma = discount_s / spot * integrals[2]
    dvar = spot * discount_s * integrals[3]
    lower = max(spot*discount_s - strike*math.exp(-r*maturity), 0.0)
    upper = spot*discount_s
    # Absolute tolerance is well below one cent per 100-unit underlying.
    # Any larger violation indicates a genuine numerical failure.
    if (not all(math.isfinite(x) for x in (call, delta, gamma, dvar)) or
            call < lower-1e-7 or call > upper+1e-7 or
            delta < -1e-7 or delta > discount_s+1e-7 or gamma < -1e-7):
        raise FloatingPointError(
            f"Adaptive Heston Fourier unstable at S={spot:.8g}, v={variance:.8g}, "
            f"K={strike:.8g}, T={maturity:.8g}; "
            f"price={call:.9g} gamma={gamma:.9g} delta={delta:.9g}"
        )
    return OptionGreeks(
        price=max(call, lower), delta=max(0., min(delta, discount_s)),
        gamma=max(gamma, 0.), variance_sensitivity=dvar,
        initial_volatility_vega=2*math.sqrt(variance)*dvar,
    )

def heston_greeks_batch(
    params: HestonParams, market: MarketDynamics, *,
    strike: float, maturity: float, spots: np.ndarray, variances: np.ndarray,
    option_type: str = "call", n_quadrature: int = 192,
) -> OptionGreeks:
    """Vectorized Q price/Greeks at one strike and remaining maturity.

    Integrates the Heston characteristic function using fixed Gauss-Legendre
    quadrature with a rational map to (0, infinity). Accurate in the ordinary
    option region; numerical failures are flagged rather than silently clipped.
    Compare against `heston_greeks` before changing the model parameter range.
    """
    from scipy.stats import norm

    if market.measure != "Q":
        raise ValueError("Risk-neutral (Q) dynamics required for valuation")
    if maturity <= 0 or not math.isfinite(maturity) or strike <= 0 or not math.isfinite(strike):
        raise ValueError("Require positive finite strike and remaining maturity")
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be call or put")
    if n_quadrature < 64:
        raise ValueError("n_quadrature must be at least 64")
    spots, variances = np.broadcast_arrays(np.asarray(spots, float), np.asarray(variances, float))
    if np.any(~np.isfinite(spots)) or np.any(spots <= 0) or np.any(~np.isfinite(variances)) or np.any(variances < 0):
        raise ValueError("Invalid path prices or variances")
    outshape = spots.shape
    ss, vv = spots.reshape(-1), variances.reshape(-1)
    prices = np.empty_like(ss)
    deltas = np.empty_like(ss)
    gammas = np.empty_like(ss)
    dvars = np.empty_like(ss)
    r, q = market.risk_free_rate, market.dividend_yield
    stock_discount = math.exp(-q * maturity)
    strike_discount = math.exp(-r * maturity)

    if params.xi == 0:
        a = -math.expm1(-params.kappa * maturity) / params.kappa
        total_variance = params.theta * maturity + (vv - params.theta) * a
        total_variance = np.maximum(total_variance, 1e-16)
        sd = np.sqrt(total_variance)
        d1 = (np.log(ss / strike) + (r - q) * maturity + .5 * total_variance) / sd
        d2 = d1 - sd
        c = ss * stock_discount * norm.cdf(d1) - strike * strike_discount * norm.cdf(d2)
        d = stock_discount * norm.cdf(d1)
        ga = stock_discount * norm.pdf(d1) / (ss * sd)
        dv = ss * stock_discount * norm.pdf(d1) * a / (2 * sd)
        prices[:], deltas[:], gammas[:], dvars[:] = c, d, ga, dv
    else:
        x, w = _legendre_nodes(n_quadrature)
        z = (x + 1) / 2
        w = w / 2
        kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
        # Damping scale follows the expected integrated variance for this tenor.
        integral_variance = theta * maturity + (vv - theta) * (-math.expm1(-kappa * maturity)) / kappa
        scale = 2 / np.sqrt(np.maximum(integral_variance, 1e-6))
        # Far beyond the expected short-term diffusion scale, the option's
        # time value is negligible at the precision of this pilot. At these
        # coordinates the oscillatory Fourier formula becomes ill-conditioned.
        # This clearly delimited asymptotic shortcut is NOT a general pricer.
        log_forward_m = np.log(ss / strike) + (r-q)*maturity
        deep = (maturity <= 0.10) & (
            np.abs(log_forward_m) >= np.maximum(0.12, 10*np.sqrt(np.maximum(integral_variance, 0)))
        )
        for start in range(0, len(ss), 512):
            stop = min(start + 512, len(ss))
            s = ss[start:stop, None]
            v = vv[start:stop, None]
            u = scale[start:stop, None] * z / (1 - z)
            weights = scale[start:stop, None] * w / (1 - z)**2
            logsk = np.log(s / strike)

            def terms(shifted):
                zz = u - 1j if shifted else u
                iu = 1j * zz
                b = kappa - rho * xi * iu
                d = np.sqrt(b*b + xi*xi*(zz*zz + iu))
                bmd = -xi*xi*(zz*zz + iu) / (b + d)
                g = bmd / (b + d)
                e = np.exp(-d * maturity)
                c = (kappa * theta / xi**2) * (
                    bmd * maturity - 2 * (np.log1p(-g*e) - np.log1p(-g))
                )
                D = (bmd / xi**2) * (1 - e) / (1 - g*e)
                phase = np.exp(1j*u*(logsk + (r-q)*maturity) + c + D*v)
                return phase, D

            phase1, D1 = terms(True)
            phase2, D2 = terms(False)
            P1 = .5 + np.sum(weights * (phase1/(1j*u)).real, axis=1) / math.pi
            P2 = .5 + np.sum(weights * (phase2/(1j*u)).real, axis=1) / math.pi
            prices[start:stop] = ss[start:stop]*stock_discount*P1 - strike*strike_discount*P2
            deltas[start:stop] = stock_discount*P1
            gammas[start:stop] = (stock_discount/ss[start:stop])*np.sum(
                weights*phase1.real, axis=1) / math.pi
            dvars[start:stop] = (
                ss[start:stop]*stock_discount*np.sum(weights*(phase1*D1/(1j*u)).real, axis=1)
                - strike*strike_discount*np.sum(weights*(phase2*D2/(1j*u)).real, axis=1)
            ) / math.pi

            # Near zero variance, deep moneyness and very short tenor, the
            # oscillatory integral may need adaptive quadrature. Avoid using
            # arbitrage-violating numerical prices in a financial backtest.
            lower = np.maximum(ss[start:stop]*stock_discount-strike*strike_discount, 0)
            upper = ss[start:stop]*stock_discount
            if np.any(deep[start:stop]):
                region = np.flatnonzero(deep[start:stop]) + start
                prices[region] = np.maximum(
                    ss[region]*stock_discount-strike*strike_discount, 0)
                deltas[region] = np.where(log_forward_m[region] > 0, stock_discount, 0)
                gammas[region] = 0
                dvars[region] = 0
            invalid = (
                ~np.isfinite(prices[start:stop]) |
                (prices[start:stop] < lower - 1e-4) |
                (prices[start:stop] > upper + 1e-4) |
                (deltas[start:stop] < -1e-3) |
                (deltas[start:stop] > stock_discount + 1e-3) |
                (gammas[start:stop] < -1e-3)
            )
            if np.any(invalid):
                if n_quadrature >= 2048:
                    # An additional Gauss-Legendre refinement may still alias
                    # the near-deterministic, highly oscillatory Fourier tail.
                    # Fall back only for states that violate numerical bounds.
                    for jj in np.flatnonzero(invalid):
                        pos = start + int(jj)
                        accurate = _adaptive_heston_greeks_near_boundary(
                            params, market, spot=float(ss[pos]),
                            variance=float(vv[pos]), strike=strike,
                            maturity=maturity,
                        )
                        prices[pos] = accurate.price
                        deltas[pos] = accurate.delta
                        gammas[pos] = accurate.gamma
                        dvars[pos] = accurate.variance_sensitivity
                    continue
                # Re-evaluate only numerically challenging states at higher
                # resolution, not all path/date pairs.
                idx = np.flatnonzero(invalid) + start
                refined = heston_greeks_batch(
                    params, market, strike=strike, maturity=maturity,
                    spots=ss[idx], variances=vv[idx], option_type="call",
                    n_quadrature=1024 if n_quadrature < 1024 else 2048,
                )
                prices[idx] = refined.price
                deltas[idx] = refined.delta
                gammas[idx] = refined.gamma
                dvars[idx] = refined.variance_sensitivity
            if np.any(~np.isfinite(prices[start:stop])) or np.any(~np.isfinite(deltas[start:stop])):
                raise FloatingPointError("Unstable Heston path valuation")
    # Full-truncation Euler can produce exactly zero effective variance.
    # Near expiration, the standard P1/P2 representation may be inaccurate
    # without violating its coarse bounds (e.g. a small, spurious OTM delta).
    if params.xi > 0 and maturity <= 2/250 + 1e-12:
        for i in np.flatnonzero(vv <= 1e-4):
            accurate = _adaptive_heston_greeks_near_boundary(
                params, market, spot=float(ss[i]), variance=float(vv[i]),
                strike=strike, maturity=maturity,
            )
            prices[i], deltas[i] = accurate.price, accurate.delta
            gammas[i], dvars[i] = accurate.gamma, accurate.variance_sensitivity
    gammas = np.maximum(gammas, 0)  # convexity; tiny negative Fourier noise
    if option_type == "put":
        prices -= ss * stock_discount - strike * strike_discount
        deltas -= stock_discount
    # Tiny integration noise can place otherwise negligible OTM options
    # slightly below zero. Price errors above one basis point are rejected.
    if np.any(prices < -1e-4):
        raise FloatingPointError("Negative Heston option price beyond integration tolerance")
    prices = np.maximum(prices, 0)
    return OptionGreeks(
        price=prices.reshape(outshape), delta=deltas.reshape(outshape),
        gamma=gammas.reshape(outshape), variance_sensitivity=dvars.reshape(outshape),
        initial_volatility_vega=(2*np.sqrt(vv)*dvars).reshape(outshape),
    )


def heston_greeks_strikes_batch(
    params: HestonParams, market: MarketDynamics, *,
    strikes: np.ndarray, maturity: float, spots: np.ndarray, variances: np.ndarray,
    option_type: str = "call", n_quadrature: int = 192,
) -> OptionGreeks:
    """Joint price/Greeks for strikes sharing a Heston state and maturity.

    This is mathematically the same Fourier quadrature as
    ``heston_greeks_batch``, but its strike-independent characteristic-function
    coefficients are calculated only once. Output shapes are (paths, strikes).
    The legacy single-strike implementation remains the numerical reference
    and the adaptive fallback for challenging states.
    """
    ks = np.asarray(strikes, dtype=float)
    if ks.ndim != 1 or len(ks) == 0 or not np.all(np.isfinite(ks)) or np.any(ks <= 0):
        raise ValueError("strikes must be a nonempty vector of positive finite values")
    if market.measure != "Q":
        raise ValueError("Risk-neutral (Q) dynamics required for valuation")
    if not math.isfinite(maturity) or maturity <= 0:
        raise ValueError("Require positive finite remaining maturity")
    if option_type not in {"call", "put"}:
        raise ValueError("option_type must be call or put")
    if n_quadrature < 64:
        raise ValueError("n_quadrature must be at least 64")
    ss, vv = np.broadcast_arrays(np.asarray(spots, float), np.asarray(variances, float))
    if ss.ndim != 1 or not np.all(np.isfinite(ss)) or np.any(ss <= 0) or not np.all(np.isfinite(vv)) or np.any(vv < 0):
        raise ValueError("spots and variances must be positive finite 1-D arrays")
    if params.xi == 0 or len(ks) == 1:
        parts = [heston_greeks_batch(
            params, market, strike=float(k), maturity=maturity,
            spots=ss, variances=vv, option_type=option_type,
            n_quadrature=n_quadrature,
        ) for k in ks]
        return OptionGreeks(**{
            field: np.column_stack([getattr(part, field) for part in parts])
            for field in OptionGreeks.__dataclass_fields__
        })

    size, n_strikes = len(ss), len(ks)
    price = np.empty((size, n_strikes), dtype=float)
    delta = np.empty_like(price)
    gamma = np.empty_like(price)
    dvar = np.empty_like(price)
    r, q = market.risk_free_rate, market.dividend_yield
    discount_s = math.exp(-q*maturity)
    discount_k = math.exp(-r*maturity)
    kappa, theta, xi, rho = params.kappa, params.theta, params.xi, params.rho
    x, w = _legendre_nodes(n_quadrature)
    z = (x + 1)/2
    w = w/2
    integrated_v = theta*maturity + (vv-theta)*(-math.expm1(-kappa*maturity))/kappa
    scale = 2/np.sqrt(np.maximum(integrated_v, 1e-6))
    log_forward_m = np.log(ss[:,None]/ks[None,:]) + (r-q)*maturity
    deep = (maturity <= .10) & (
        np.abs(log_forward_m) >= np.maximum(.12, 10*np.sqrt(np.maximum(integrated_v[:,None], 0)))
    )
    log_ks = np.log(ks)[None,None,:]

    for start in range(0, size, 256):
        stop = min(start+256, size)
        s = ss[start:stop,None]
        v = vv[start:stop,None]
        u = scale[start:stop,None]*z/(1-z)
        weights = scale[start:stop,None]*w/(1-z)**2

        def common_terms(shifted: bool):
            zz = u-1j if shifted else u
            iu = 1j*zz
            b = kappa-rho*xi*iu
            d = np.sqrt(b*b + xi*xi*(zz*zz+iu))
            bmd = -xi*xi*(zz*zz+iu)/(b+d)
            g = bmd/(b+d)
            e = np.exp(-d*maturity)
            c = (kappa*theta/xi**2)*(
                bmd*maturity - 2*(np.log1p(-g*e)-np.log1p(-g))
            )
            D = (bmd/xi**2)*(1-e)/(1-g*e)
            base_phase = np.exp(1j*u*(np.log(s)+(r-q)*maturity) + c + D*v)
            return base_phase, D

        base1,D1 = common_terms(True)
        base2,D2 = common_terms(False)
        strike_factor = np.exp(-1j*u[:,:,None]*log_ks)
        phase1 = base1[:,:,None]*strike_factor
        phase2 = base2[:,:,None]*strike_factor
        inv_iu = 1/(1j*u[:,:,None])
        ww = weights[:,:,None]
        P1 = .5 + np.sum(ww*(phase1*inv_iu).real, axis=1)/math.pi
        P2 = .5 + np.sum(ww*(phase2*inv_iu).real, axis=1)/math.pi
        price[start:stop] = s*discount_s*P1 - ks[None,:]*discount_k*P2
        delta[start:stop] = discount_s*P1
        gamma[start:stop] = discount_s/s * np.sum(ww*phase1.real,axis=1)/math.pi
        dvar[start:stop] = (
            s*discount_s*np.sum(ww*(phase1*D1[:,:,None]*inv_iu).real,axis=1)
            - ks[None,:]*discount_k*np.sum(ww*(phase2*D2[:,:,None]*inv_iu).real,axis=1)
        )/math.pi

        deep_local = deep[start:stop]
        if np.any(deep_local):
            intrinsic = np.maximum(s*discount_s - ks[None,:]*discount_k, 0)
            price[start:stop] = np.where(deep_local, intrinsic, price[start:stop])
            delta[start:stop] = np.where(deep_local, np.where(log_forward_m[start:stop]>0, discount_s,0), delta[start:stop])
            gamma[start:stop] = np.where(deep_local, 0, gamma[start:stop])
            dvar[start:stop] = np.where(deep_local, 0, dvar[start:stop])

        lower = np.maximum(s*discount_s-ks[None,:]*discount_k, 0)
        upper = s*discount_s
        bad = (
            ~np.isfinite(price[start:stop]) |
            (price[start:stop] < lower-1e-4) |
            (price[start:stop] > upper+1e-4) |
            (delta[start:stop] < -1e-3) |
            (delta[start:stop] > discount_s+1e-3) |
            (gamma[start:stop] < -1e-3)
        )
        # Even finite (and hence not flagged) short-expiry P1/P2 estimates
        # can contain spurious OTM prices/deltas near zero variance.
        boundary = ((vv[start:stop] <= 1e-4) &
                    (maturity <= 2/250 + 1e-12))
        for j in range(n_strikes):
            for local_i in np.flatnonzero(boundary):
                pos = start+int(local_i)
                accurate = _adaptive_heston_greeks_near_boundary(
                    params, market, spot=float(ss[pos]),
                    variance=float(vv[pos]), strike=float(ks[j]),
                    maturity=maturity,
                )
                price[pos,j], delta[pos,j] = accurate.price, accurate.delta
                gamma[pos,j], dvar[pos,j] = accurate.gamma, accurate.variance_sensitivity
            indices = np.flatnonzero(bad[:,j] & ~boundary)+start
            if not len(indices):
                continue
            refined = heston_greeks_batch(
                params, market, strike=float(ks[j]), maturity=maturity,
                spots=ss[indices], variances=vv[indices], option_type="call",
                n_quadrature=1024 if n_quadrature<1024 else 2048,
            )
            price[indices,j] = refined.price
            delta[indices,j] = refined.delta
            gamma[indices,j] = refined.gamma
            dvar[indices,j] = refined.variance_sensitivity

    gamma = np.maximum(gamma,0)
    if option_type == "put":
        price -= ss[:,None]*discount_s-ks[None,:]*discount_k
        delta -= discount_s
    if np.any(~np.isfinite(price)) or np.any(~np.isfinite(delta)) or np.any(price < -1e-4):
        raise FloatingPointError("Unstable batched Heston path valuation")
    price = np.maximum(price,0)
    return OptionGreeks(
        price=price, delta=delta, gamma=gamma,
        variance_sensitivity=dvar,
        initial_volatility_vega=2*np.sqrt(vv[:,None])*dvar,
    )
