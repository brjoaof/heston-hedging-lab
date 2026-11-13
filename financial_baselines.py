"""Reproducible financial hedge baselines on *frozen* Heston-Q episode pools.

Reuses the SAC financial engine and its [0,max_position], bounded-increment,
min-trade, bid/ask and fixed-fee execution conventions. No policy is trained.
All marks and Delta Heston originate from the frozen selected pricer (COS).

Black-20% and Black-IV are *partial* Black deltas: implied volatility, if used,
is held fixed in the Black derivative. Heston MV uses dC/dv (not Black vega).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import ndtr

from forward_experiments import evaluate_baseline
from hedging import ForwardHedgeResults, execute_forward_trades, forward_prices, simulate_forward_hedge
from heston import HestonParams
from portfolio import fixed_eniac_book
from pricing_cos import heston_cos_greeks_strikes_batch
from rl_env import ForwardEnvConfig, ForwardEpisodePool


PERIOD_DAYS = 21
STRATEGIES = (
    "no_hedge", "hold_initial", "heston_daily", "heston_periodic_21",
    "heston_band_0p5", "black20_daily", "black20_periodic_21",
    "heston_mv_daily", "heston_mv_periodic_21",
    "black_iv_daily", "black_iv_periodic_21",
)
DEFAULT_STRATEGIES = STRATEGIES[:-2]  # Black-IV optional: inversion can be ill-conditioned


def black_call_delta(spots: np.ndarray, strikes: np.ndarray, maturity: float, *,
                     risk_free_rate: float, dividend_yield: float, sigma: np.ndarray | float
                     ) -> np.ndarray:
    """Spot Delta of a European Black-Scholes call for fixed input sigma."""
    s, k, sig = np.broadcast_arrays(np.asarray(spots, dtype=float),
                                     np.asarray(strikes, dtype=float),
                                     np.asarray(sigma, dtype=float))
    if (maturity <= 0 or not math.isfinite(maturity) or
            np.any(s <= 0) or np.any(k <= 0) or np.any(sig < 0) or
            not all(np.isfinite(a).all() for a in (s, k, sig))):
        raise ValueError("Invalid Black delta inputs")
    tau_root = math.sqrt(maturity)
    fwd_log = np.log(s / k) + (risk_free_rate - dividend_yield) * maturity
    sig_safe = np.maximum(sig, 1.e-12)
    d1 = (fwd_log + 0.5 * sig_safe**2 * maturity) / (sig_safe * tau_root)
    p = np.where(sig > 1.e-12, ndtr(d1),
                 np.where(fwd_log > 0, 1., np.where(fwd_log < 0, 0., 0.5)))
    return math.exp(-dividend_yield * maturity) * p


def black_call_price(spots: np.ndarray, strikes: np.ndarray, maturity: float, *,
                     risk_free_rate: float, dividend_yield: float, sigma: np.ndarray | float
                     ) -> np.ndarray:
    """Vectorized discounted Black-Scholes call price, including sigma=0."""
    s, k, sig = np.broadcast_arrays(np.asarray(spots, dtype=float),
                                     np.asarray(strikes, dtype=float),
                                     np.asarray(sigma, dtype=float))
    if (maturity <= 0 or not math.isfinite(maturity) or
            np.any(s <= 0) or np.any(k <= 0) or np.any(sig < 0) or
            not all(np.isfinite(a).all() for a in (s, k, sig))):
        raise ValueError("Invalid Black price inputs")
    dfr = math.exp(-risk_free_rate * maturity)
    dfq = math.exp(-dividend_yield * maturity)
    std = np.maximum(sig * math.sqrt(maturity), 1.e-12)
    d1 = (np.log(s / k) + (risk_free_rate - dividend_yield) * maturity +
          0.5 * sig**2 * maturity) / std
    d2 = d1 - std
    call = s * dfq * ndtr(d1) - k * dfr * ndtr(d2)
    return np.where(sig > 1.e-12, call, np.maximum(s * dfq - k * dfr, 0.))


def invert_black_call_iv(spots: np.ndarray, strikes: np.ndarray, maturity: float, *,
                         market_rate: float, dividend_yield: float,
                         call_prices: np.ndarray, tolerance: float = 1.e-7,
                         max_sigma: float = 5., iterations: int = 48) -> np.ndarray:
    """Vectorized monotone bisection; returns sigma=0 at intrinsic precision.

    Deep ITM/OTM calls can be ill-conditioned. This returns an *equivalent*
    sigma where identifiable; never claims unique IV at nearly zero vega.
    """
    s, k, p = np.broadcast_arrays(np.asarray(spots, dtype=float),
                                   np.asarray(strikes, dtype=float),
                                   np.asarray(call_prices, dtype=float))
    dfr = math.exp(-market_rate*maturity)
    dfq = math.exp(-dividend_yield*maturity)
    intrinsic = np.maximum(s*dfq-k*dfr, 0.)
    upper = s*dfq
    if (np.any(~np.isfinite(p)) or
            np.any(p < intrinsic - tolerance) or np.any(p > upper + tolerance)):
        raise ValueError("Heston call price violates Black no-arbitrage bounds")
    target = np.clip(p, intrinsic, upper)
    active = (target > intrinsic + tolerance) & (target < upper-tolerance)
    lo = np.zeros_like(target)
    hi = np.full_like(target, max_sigma)
    # fail explicitly if sigma ceiling cannot reproduce observed option value
    ceiling = black_call_price(s, k, maturity, risk_free_rate=market_rate,
                               dividend_yield=dividend_yield, sigma=hi)
    if np.any(active & (ceiling < target - tolerance)):
        raise ValueError("Black IV not bracketed by max_sigma")
    for _ in range(iterations):
        mid = 0.5*(lo+hi)
        guess = black_call_price(s, k, maturity, risk_free_rate=market_rate,
                                 dividend_yield=dividend_yield, sigma=mid)
        lo = np.where(active & (guess < target), mid, lo)
        hi = np.where(active & (guess >= target), mid, hi)
    return np.where(active, 0.5*(lo+hi), 0.)


def _bucket_deltas(pool: ForwardEpisodePool, cfg: ForwardEnvConfig,
                   *, source: Literal["heston", "black20", "black_iv", "heston_mv"],
                   heston: HestonParams | None = None,
                   cos_n_terms: int = 512, cos_L: float = 24.) -> np.ndarray:
    """Signed option-book spot sensitivities by maturity, at every live day.

    Heston delta directly uses the stored COS marks; all alternatives use
    the same frozen Heston paths, with no look-ahead beyond the decision day.
    """
    if pool.book_paths.bucket_delta is None:
        raise ValueError("Frozen Heston pool lacks bucket deltas")
    if source == "heston":
        return pool.book_paths.bucket_delta
    if source not in ("black20", "black_iv", "heston_mv"):
        raise ValueError("Unknown delta source")
    if source == "heston_mv" and heston is None:
        raise ValueError("Heston parameters needed for minimum-variance delta")
    if source == "black_iv" and pool.book_paths.contract_prices is None:
        raise ValueError("Black IV requires saved Heston call contract prices")
    book = fixed_eniac_book()
    maturity_index = {m: j for j, m in enumerate(cfg.expiries_days)}
    n_paths, n_times = pool.paths.spot.shape
    n_buckets = len(cfg.expiries_days)
    output = np.zeros((n_paths, n_times, n_buckets), dtype=np.float64)
    r, q = pool.market_q.risk_free_rate, pool.market_q.dividend_yield
    for maturity in cfg.expiries_days:
        contracts = [(i,c) for i,c in enumerate(book.contracts) if c.maturity_days == maturity]
        if not contracts or any(c.option_type != "call" for _,c in contracts):
            raise ValueError("Financial baselines currently require the ENIAC short-call book")
        strikes = np.array([c.strike for _, c in contracts], dtype=float)[None,:]
        amounts = np.array([c.quantity*c.multiplier for _,c in contracts], dtype=float)
        bucket = maturity_index[maturity]
        for day in range(maturity):
            s = pool.paths.spot[:,day,None]
            tau = (maturity-day)/cfg.trading_days_per_year
            if source in ("black20", "black_iv"):
                if source == "black20":
                    vol = .20
                else:
                    price = pool.book_paths.contract_prices[:,day,[i for i,_ in contracts]]
                    vol = invert_black_call_iv(s, strikes, tau,
                                                market_rate=r, dividend_yield=q,
                                                call_prices=price)
                local_delta = black_call_delta(s, strikes, tau,
                                               risk_free_rate=r, dividend_yield=q,
                                               sigma=vol)
            else:
                greeks = heston_cos_greeks_strikes_batch(
                    heston, pool.market_q, strikes=strikes[0], maturity=tau,
                    spots=s[:,0], variances=pool.paths.variance[:,day],
                    option_type="call", n_terms=cos_n_terms, L=cos_L,
                )
                local_delta = greeks.delta + (heston.rho*heston.xi/s) * greeks.variance_sensitivity
            output[:,day,bucket] = np.sum(local_delta*amounts[None,:], axis=1)
    if source == "heston_mv" and heston.xi == 0:
        np.testing.assert_allclose(output, pool.book_paths.bucket_delta,
                                   rtol=1.e-8, atol=1.e-8)
    if np.any(~np.isfinite(output)):
        raise ValueError("Nonfinite baseline Delta")
    return output


def _position_targets(delta: np.ndarray, pool: ForwardEpisodePool,
                      cfg: ForwardEnvConfig) -> np.ndarray:
    n_paths,n_times,n_buckets = delta.shape
    if (delta.shape != (len(pool.paths.spot), pool.paths.spot.shape[1],
                       len(cfg.expiries_days))):
        raise ValueError("Delta shape mismatch")
    days = np.arange(n_times)[:,None]
    alive = days < np.asarray(cfg.expiries_days)[None,:]
    time_to_exp = np.maximum(np.asarray(cfg.expiries_days)[None,:]-days, 0.)
    df_dspot = np.exp((pool.market_q.risk_free_rate-pool.market_q.dividend_yield)*
                       time_to_exp / cfg.trading_days_per_year)
    target = np.clip(-delta / df_dspot[None,:,:], 0., cfg.max_position)
    target *= alive[None,:,:]
    return target


def _actions_to_target(pool: ForwardEpisodePool, cfg: ForwardEnvConfig,
                       targets: np.ndarray, *, frequency: int) -> tuple[np.ndarray,np.ndarray]:
    if type(frequency) is not int or frequency < 1:
        raise ValueError("frequency must be a positive integer")
    n_paths, n_times, n_buckets = targets.shape
    initial = targets[:,0,:].copy() # specific strategy's time-zero positions and costs
    pos = initial.copy()
    actions = np.zeros((n_paths, n_times-1, n_buckets), dtype=float)
    for day in range(n_times-1):
        if day % frequency:
            request = np.zeros_like(pos)
        else:
            request = np.clip(targets[:,day,:]-pos,
                              -cfg.max_trade_increment, cfg.max_trade_increment)
        actions[:,day,:] = request
        forward_mid = forward_prices(pool.paths.spot[:,day],day,cfg.expiries_days,
                                     pool.market_q.risk_free_rate,
                                     dividend_yield=pool.market_q.dividend_yield,
                                     trading_days_per_year=cfg.trading_days_per_year)
        pos,_,_,_ = execute_forward_trades(
            pos,request,forward_mid,np.asarray(cfg.expiries_days)>day,
            spread_half_pcts=cfg.spread_half_pcts, fixed_cost=cfg.fixed_cost,
            min_trade=cfg.min_trade, max_trade_increment=cfg.max_trade_increment,
            max_position=cfg.max_position,
        )
        pos[:,np.asarray(cfg.expiries_days)==day+1] = 0.
    return initial,actions


def evaluate_financial_baseline(pool: ForwardEpisodePool, cfg: ForwardEnvConfig,
                                strategy: str, *,
                                heston: HestonParams | None = None,
                                cos_n_terms: int = 512,
                                cos_L: float = 24.,
                                cached_deltas: dict | None = None
                                ) -> ForwardHedgeResults:
    """Return Q terminal P&L with the *same* forward execution as SAC."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    old = {"no_hedge":"no_hedge", "hold_initial":"hold_initial",
           "heston_daily":"delta_daily", "heston_periodic_21":"delta_periodic_21",
           "heston_band_0p5":"delta_band_0p5"}
    if strategy in old:
        return evaluate_baseline(pool,cfg,old[strategy])
    source = ("black20" if strategy.startswith("black20") else
              "black_iv" if strategy.startswith("black_iv") else "heston_mv")
    frequency = PERIOD_DAYS if "periodic_21" in strategy else 1
    cache = {} if cached_deltas is None else cached_deltas
    if source not in cache:
        cache[source] = _bucket_deltas(pool,cfg,source=source,heston=heston,
                                       cos_n_terms=cos_n_terms,cos_L=cos_L)
    targets = _position_targets(cache[source],pool,cfg)
    initial, actions = _actions_to_target(pool,cfg,targets,frequency=frequency)
    return simulate_forward_hedge(
        pool.paths,pool.book_paths,pool.market_q,
        initial_positions=initial,actions=actions,
        expiries_days=cfg.expiries_days,spread_half_pcts=cfg.spread_half_pcts,
        fixed_cost=cfg.fixed_cost,min_trade=cfg.min_trade,
        max_trade_increment=cfg.max_trade_increment,max_position=cfg.max_position,
        trading_days_per_year=cfg.trading_days_per_year,
    )
