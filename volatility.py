"""Black-Scholes implied volatility and Heston model-implied volatility surfaces.

The surface describes the *option market model*, independently of which
contracts happen to be held in a particular portfolio.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator

from heston import HestonParams, MarketDynamics
from pricing import black_scholes_price, heston_price


def implied_volatility(
    price: float, *, spot: float, strike: float, maturity: float,
    rate: float = 0.0, dividend: float = 0.0, option_type: str = "call",
) -> float:
    """Find the Black-Scholes volatility reproducing an observed option price.

    Requires positive time to maturity. A price exactly at its lower no-arbitrage
    bound has implied volatility zero; the upper bound requires infinite
    volatility and is therefore rejected. Inverting the out-of-the-money side
    improves numerical stability for in-the-money options.
    """
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    if not all(map(math.isfinite, (price, spot, strike, maturity, rate, dividend))):
        raise ValueError("All inputs must be finite")
    if spot <= 0 or strike <= 0 or maturity <= 0:
        raise ValueError("spot, strike and maturity must be positive")

    discounted_spot = spot * math.exp(-dividend * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    parity = discounted_spot - discounted_strike
    lower = max(parity, 0.0) if option_type == "call" else max(-parity, 0.0)
    upper = discounted_spot if option_type == "call" else discounted_strike
    tolerance = 1e-13 * max(1.0, discounted_spot, discounted_strike)
    if price < lower - tolerance or price > upper + tolerance:
        raise ValueError("Option price violates Black-Scholes no-arbitrage bounds")
    if price <= lower + tolerance:
        return 0.0
    if price >= upper - tolerance:
        raise ValueError("Option price at the upper bound has no finite implied volatility")

    # A deep in-the-money call is inverted through its equivalent OTM put,
    # and vice versa. This avoids large intrinsic-value cancellations.
    if parity > 0:
        otm_type = "put"
        otm_price = price - parity if option_type == "call" else price
    else:
        otm_type = "call"
        otm_price = price if option_type == "call" else price + parity

    def error(vol: float) -> float:
        return black_scholes_price(
            spot, strike, maturity, vol, rate, dividend, otm_type
        ) - otm_price

    high = 1.0
    while error(high) < 0 and high < 32.0:
        high *= 2
    if error(high) < 0:
        raise ValueError("Implied volatility exceeds the supported search interval")
    return float(brentq(error, 0.0, high, xtol=1e-12, rtol=1e-12))


@dataclass(frozen=True)
class ImpliedVolSurface:
    maturities_days: np.ndarray     # (n_maturities,)
    log_moneyness: np.ndarray       # (n_strikes,): log(K / forward)
    strikes: np.ndarray             # (n_maturities, n_strikes)
    implied_volatility: np.ndarray  # (n_maturities, n_strikes)
    spot: float
    variance: float


def heston_volatility_surface(
    params: HestonParams, market: MarketDynamics, *,
    maturities_days: list[int], log_moneyness: list[float],
    spot: float | None = None, variance: float | None = None,
    trading_days_per_year: int = 252,
) -> ImpliedVolSurface:
    """Compute a Heston-IV grid for a *single* current market state.

    Strikes are defined by K = F(t,T) exp(log_moneyness). The result can be
    recomputed at later states; this function does not precompute a full
    surface at every step of every Monte Carlo path.
    """
    if market.measure != "Q":
        raise ValueError("Implied volatility surfaces require Q pricing parameters")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive")
    days = np.asarray(maturities_days)
    logs = np.asarray(log_moneyness, dtype=float)
    if (days.ndim != 1 or len(days) == 0 or
            not np.issubdtype(days.dtype, np.integer) or
            np.any(days <= 0) or np.any(np.diff(days) <= 0)):
        raise ValueError("maturities_days must be increasing positive integers")
    if (logs.ndim != 1 or len(logs) == 0 or
            not np.all(np.isfinite(logs)) or np.any(np.diff(logs) <= 0)):
        raise ValueError("log_moneyness must be increasing finite numbers")

    s = market.spot0 if spot is None else float(spot)
    v = params.v0 if variance is None else float(variance)
    if not math.isfinite(s) or s <= 0 or not math.isfinite(v) or v < 0:
        raise ValueError("spot must be positive and variance nonnegative")

    strikes = np.empty((len(days), len(logs)))
    iv = np.empty_like(strikes)
    for i, day in enumerate(days):
        tau = int(day) / trading_days_per_year
        forward = s * math.exp((market.risk_free_rate - market.dividend_yield) * tau)
        for j, k in enumerate(logs):
            strike = forward * math.exp(float(k))
            # Price OTM calls/puts for improved numerical accuracy.
            opt_type = "put" if k < 0 else "call"
            option_price = heston_price(
                params, market, strike=strike, maturity=tau,
                option_type=opt_type, spot=s, variance=v,
            )
            strikes[i, j] = strike
            iv[i, j] = implied_volatility(
                option_price, spot=s, strike=strike, maturity=tau,
                rate=market.risk_free_rate, dividend=market.dividend_yield,
                option_type=opt_type,
            )
    return ImpliedVolSurface(days.astype(int), logs, strikes, iv, s, v)


@dataclass(frozen=True)
class VolatilityLookup:
    """Heston IV precomputed by variance for fixed Q parameters and tenors.

    At fixed log(K/F), Heston IV is independent of the absolute spot level.
    Each path therefore needs only its current variance to look up an IV
    surface. The table is for constant-*tenor* surfaces, not fixed options.
    """
    maturities_days: np.ndarray          # (n_tenors,)
    log_moneyness: np.ndarray            # (n_moneyness,)
    variance_nodes: np.ndarray           # (n_nodes,)
    implied_volatility: np.ndarray       # (n_nodes, n_tenors, n_moneyness)

    def evaluate(self, variance: np.ndarray) -> np.ndarray:
        """Return surfaces for arbitrary shaped variance inputs.

        For (n_paths, n_steps+1) variance inputs, results have shape
        (n_paths, n_steps+1, n_tenors, n_moneyness). Never extrapolate.
        """
        v = np.asarray(variance, dtype=float)
        if not np.all(np.isfinite(v)) or np.any(v < 0):
            raise ValueError("Current variance must be finite and nonnegative")
        lo, hi = self.variance_nodes[0], self.variance_nodes[-1]
        if np.any(v < lo) or np.any(v > hi):
            raise ValueError(
                f"Variance outside interpolation interval [{lo:g}, {hi:g}]; "
                "extend volatility_surface.variance_nodes in config.json"
            )
        result = PchipInterpolator(
            self.variance_nodes, self.implied_volatility, axis=0,
            extrapolate=False,
        )(v)
        if not np.all(np.isfinite(result)) or np.any(result < 0):
            raise FloatingPointError("Invalid interpolated implied volatility")
        return result


def build_volatility_lookup(
    params: HestonParams, market: MarketDynamics, *,
    maturities_days: list[int], log_moneyness: list[float],
    variance_nodes: list[float], trading_days_per_year: int = 252,
) -> VolatilityLookup:
    """Amortize Heston Fourier pricing across millions of simulated states.

    Precompute accurate reference IVs at specified variance nodes. Cubic,
    shape-preserving interpolation is used only along the variance axis;
    the strike/tenor coordinates remain exactly as specified by the user.
    """
    if market.measure != "Q":
        raise ValueError("Volatility lookup requires Q pricing parameters")
    nodes = np.asarray(variance_nodes, dtype=float)
    if (nodes.ndim != 1 or len(nodes) < 3 or
            not np.all(np.isfinite(nodes)) or np.any(nodes < 0) or
            np.any(np.diff(nodes) <= 0)):
        raise ValueError("variance_nodes must have >=3 increasing nonnegative finite values")

    surfaces = [
        heston_volatility_surface(
            params, market, maturities_days=maturities_days,
            log_moneyness=log_moneyness, variance=float(node),
            trading_days_per_year=trading_days_per_year,
        )
        for node in nodes
    ]
    return VolatilityLookup(
        maturities_days=surfaces[0].maturities_days,
        log_moneyness=surfaces[0].log_moneyness,
        variance_nodes=nodes,
        implied_volatility=np.stack([s.implied_volatility for s in surfaces]),
    )
