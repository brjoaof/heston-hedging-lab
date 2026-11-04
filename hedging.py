"""Daily option-book marking and transparent, non-RL hedging baselines.

Signed option book: positive quantity is long. Hedge shares offset its delta.
At inception the cash account is -book market value, so buying/selling the
book is financed; all subsequent option settlements enter the same cash
account. This defines zero initial wealth before trading costs.
"""
from dataclasses import dataclass
import math

import numpy as np

from heston import HestonParams, HestonPaths, MarketDynamics
from portfolio import OptionBook, expiry_cashflows
from pricing import heston_greeks_batch


@dataclass(frozen=True)
class BookPaths:
    value: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    variance_sensitivity: np.ndarray
    settlements: np.ndarray


def value_book_paths(
    book: OptionBook, paths: HestonPaths, params_q: HestonParams,
    market_q: MarketDynamics, *, trading_days_per_year: int = 252,
) -> BookPaths:
    """Revalue each *live* option on physical or Q paths, always using Q prices.

    Values are ex-settlement: a contract that expires on day t contributes
    zero to value[t], and its payoff enters settlements[t] exactly once.
    """
    if market_q.measure != "Q":
        raise ValueError("Option valuation requires Q market parameters")
    if trading_days_per_year <= 0:
        raise ValueError("Invalid trading_days_per_year")
    values = [np.zeros_like(paths.spot) for _ in range(4)]
    settlements = expiry_cashflows(book, paths.spot)
    n_days = paths.spot.shape[1]
    for contract in book.contracts:
        if contract.maturity_days >= n_days:
            raise ValueError("Option matures after the simulated horizon")
        amount = contract.quantity * contract.multiplier
        for day in range(contract.maturity_days):
            remaining = (contract.maturity_days - day) / trading_days_per_year
            greeks = heston_greeks_batch(
                params_q, market_q, strike=contract.strike, maturity=remaining,
                spots=paths.spot[:, day], variances=paths.variance[:, day],
                option_type=contract.option_type,
            )
            for target, field in zip(values, ("price", "delta", "gamma", "variance_sensitivity")):
                target[:, day] += amount * getattr(greeks, field)
    return BookPaths(*values, settlements)


@dataclass(frozen=True)
class HedgeResults:
    wealth: np.ndarray  # mark-to-market including option book, (paths, times)
    terminal_pnl: np.ndarray
    trades: np.ndarray    # trades per path
    turnover: np.ndarray  # total absolute shares traded
    transaction_cost: np.ndarray
    position: np.ndarray  # underlying position at each day end


def simulate_hedge(
    paths: HestonPaths, book_paths: BookPaths, market: MarketDynamics, *,
    strategy: str, proportional_cost: float = 0.001,
    fixed_cost: float = 0.5, band_width: float = 0.5,
    rebalance_every: int = 21, min_trade: float = 0.01,
    trading_days_per_year: int = 252,
) -> HedgeResults:
    """Trade underlying only; include interest, dividends and final liquidation.

    Strategies: no_hedge, delta_daily, delta_periodic, delta_band.
    The delta_band strategy trades to the band boundary, not to its centre.
    All policies observe information at t and trade at S_t (no look-ahead).
    """
    if strategy not in {"no_hedge", "delta_daily", "delta_periodic", "delta_band"}:
        raise ValueError("Unknown hedge strategy")
    if not all(
        math.isfinite(x) and x >= 0 for x in (proportional_cost, fixed_cost, band_width, min_trade)
    ):
        raise ValueError("Costs and band must be finite and nonnegative")
    if rebalance_every < 1 or trading_days_per_year < 1:
        raise ValueError("Invalid hedge frequency")
    spot = paths.spot
    value, delta = book_paths.value, book_paths.delta
    if spot.shape != value.shape or spot.shape != delta.shape or spot.shape != book_paths.settlements.shape:
        raise ValueError("Path, book value, delta, and settlements shapes must agree")
    n_paths, n_times = spot.shape
    shares = np.zeros(n_paths)
    cash = -value[:, 0].copy()
    total_cost = np.zeros(n_paths)
    trades = np.zeros(n_paths, int)
    turnover = np.zeros(n_paths)
    wealth = np.zeros_like(spot)
    position = np.zeros_like(spot)
    dt = 1.0/trading_days_per_year
    growth = math.exp(market.risk_free_rate*dt)
    # Ex-dividend spot dynamics; simple discrete approximation of cash dividends.
    dividends = math.expm1(market.dividend_yield*dt)

    for day in range(n_times):
        if day:
            cash = cash*growth + shares*spot[:, day - 1]*dividends
            cash += book_paths.settlements[:, day]

        if day == n_times-1:
            target = np.zeros(n_paths)  # liquidation at the evaluation horizon
        elif strategy == "no_hedge":
            target = shares.copy()
        else:
            wanted = -delta[:, day]
            if strategy == "delta_daily" or (strategy == "delta_periodic" and day % rebalance_every == 0):
                target = wanted
            elif strategy == "delta_band":
                gap = shares - wanted
                target = np.where(np.abs(gap) > band_width, wanted + np.sign(gap)*band_width, shares)
            else:
                target = shares.copy()
        change = target - shares
        trade = (np.abs(change) >= min_trade) | ((day == n_times - 1) & (change != 0))
        change = np.where(trade, change, 0.0)
        trade_cost = proportional_cost*spot[:, day]*np.abs(change) + fixed_cost*trade
        cash -= change*spot[:, day] + trade_cost
        shares += change
        total_cost += trade_cost
        turnover += np.abs(change)
        trades += trade.astype(int)
        position[:, day] = shares
        wealth[:, day] = cash + shares*spot[:, day] + value[:, day]

    return HedgeResults(wealth, wealth[:, -1], trades, turnover, total_cost, position)


def summarize_hedge(results: HedgeResults) -> dict:
    """A positive loss means loss; CVaR95 is mean of worst 5% losses."""
    losses = -results.terminal_pnl
    # Empirical quantile and Rockafellar-Uryasev CVaR, including ties at VaR.
    alpha = 0.95
    var95 = np.quantile(losses, alpha, method="inverted_cdf")
    cvar95 = var95 + np.maximum(losses - var95, 0.0).mean() / (1.0 - alpha)
    return {
        "mean_pnl": float(results.terminal_pnl.mean()),
        "mean_loss": float(losses.mean()),
        "q95_loss": float(var95),
        "cvar95_loss": float(cvar95),
        "mean_transaction_cost": float(results.transaction_cost.mean()),
        "mean_turnover": float(results.turnover.mean()),
        "mean_trades": float(results.trades.mean()),
    }


# Three-maturity forward-style hedging (separate from the existing spot hedge).
@dataclass(frozen=True)
class ForwardHedgeResults:
    wealth: np.ndarray              # (n_paths, n_steps+1), net of costs
    terminal_pnl: np.ndarray        # (n_paths,)
    position: np.ndarray            # (n_paths, n_steps+1, n_maturities)
    trades: np.ndarray              # executed opening, regular and closing trades
    turnover: np.ndarray            # sum of absolute executed position changes
    transaction_cost: np.ndarray    # cumulative spread + fixed, without financing
    spread_cost: np.ndarray
    fixed_cost: np.ndarray
    initial_cost: np.ndarray        # subset of transaction_cost
    hedge_pnl: np.ndarray           # (n_paths, n_steps), excluding costs


def forward_prices(
    spot: np.ndarray, day: int, expiries_days: tuple[int, ...],
    risk_free_rate: float, *, dividend_yield: float = 0.0,
    trading_days_per_year: int = 250,
) -> np.ndarray:
    """Mid forwards F(t,T) = S_t exp((r-q)(T-t)) for deterministic rates.

    On/after maturity F(t,T) = S_t for the sole purpose of close accounting;
    no new trades are allowed in expired contracts.
    """
    s = np.asarray(spot, dtype=np.float64)
    if s.ndim != 1 or not np.all(np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("spot must be a one-dimensional array of positive finite prices")
    if not expiries_days or any(type(t) is not int or t < 1 for t in expiries_days):
        raise ValueError("expiries_days must contain positive integers")
    if not math.isfinite(risk_free_rate) or not math.isfinite(dividend_yield):
        raise ValueError("rates must be finite")
    if trading_days_per_year < 1 or day < 0:
        raise ValueError("invalid time grid")
    remaining = np.maximum(np.asarray(expiries_days, dtype=float) - day, 0.0)
    return s[:, None] * np.exp(
        (risk_free_rate - dividend_yield) * remaining[None, :] / trading_days_per_year
    )


def initial_forward_positions(
    book: OptionBook, paths: HestonPaths, params_q: HestonParams,
    market_q: MarketDynamics, *, expiries_days: tuple[int, ...] = (42, 84, 126),
    trading_days_per_year: int = 250, max_position: float = 3.0,
) -> np.ndarray:
    """Delta-neutral initial forward holdings, clipped to [0, max_position].

    A short call has negative book delta; a long forward has positive
    dF/dS = exp((r-q)*tau). Unlike the ENIAC spot-delta approximation,
    division by this factor matches the *forward* hedge units exactly.
    """
    if market_q.measure != "Q" or paths.measure != "Q":
        raise ValueError("Initial forward hedge requires Q dynamics")
    if not math.isfinite(max_position) or max_position <= 0:
        raise ValueError("max_position must be positive")
    buckets = {m: i for i, m in enumerate(expiries_days)}
    delta = np.zeros((len(paths.spot), len(expiries_days)), dtype=float)
    for c in book.contracts:
        if c.maturity_days not in buckets:
            raise ValueError("option maturity not covered by forward instruments")
        g = heston_greeks_batch(
            params_q, market_q, strike=c.strike,
            maturity=c.maturity_days / trading_days_per_year,
            spots=paths.spot[:, 0], variances=paths.variance[:, 0],
            option_type=c.option_type,
        )
        delta[:, buckets[c.maturity_days]] += c.quantity * c.multiplier * g.delta
    dforward_dspot = np.exp(
        (market_q.risk_free_rate - market_q.dividend_yield)
        * np.asarray(expiries_days)[None, :] / trading_days_per_year
    )
    return np.clip(-delta / dforward_dspot, 0.0, max_position)


def execute_forward_trades(
    position: np.ndarray, action: np.ndarray, forward_mid: np.ndarray,
    active: np.ndarray, *, spread_half_pcts: tuple[float, ...],
    fixed_cost: float = 0.5, min_trade: float = 0.01,
    max_trade_increment: float = 1.0, max_position: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure SAC-Trade execution rule, reusable by a future RL environment.

    Returns next_position, executed_change, spread_cost, fixed_cost per path.
    `active` is a bucket mask at the decision time; expired buckets cannot trade.
    """
    position = np.asarray(position, dtype=float)
    action = np.asarray(action, dtype=float)
    forward_mid = np.asarray(forward_mid, dtype=float)
    active = np.asarray(active, dtype=bool)
    if position.ndim != 2 or action.shape != position.shape or forward_mid.shape != position.shape:
        raise ValueError("position, action and forward_mid must have shape (paths, buckets)")
    if active.shape != (position.shape[1],):
        raise ValueError("active must have one entry per bucket")
    if not (np.all(np.isfinite(position)) and np.all(np.isfinite(action)) and
            np.all(np.isfinite(forward_mid)) and np.all(forward_mid > 0)):
        raise ValueError("positions, actions and forward prices must be finite")
    if not (all(math.isfinite(x) and x >= 0 for x in
                (fixed_cost, min_trade, max_trade_increment, max_position))
            and max_trade_increment > 0 and max_position > 0):
        raise ValueError("invalid trade limits or costs")
    spread = np.asarray(spread_half_pcts, dtype=float)
    if spread.shape != (position.shape[1],) or not np.all(np.isfinite(spread)) or np.any(spread < 0):
        raise ValueError("invalid half-spreads")
    if np.any(position < 0) or np.any(position > max_position):
        raise ValueError("positions outside allowed bounds")
    desired = np.clip(position + np.clip(action, -max_trade_increment, max_trade_increment),
                      0.0, max_position)
    requested_change = desired - position
    executed_change = np.where(
        active[None, :] & (np.abs(requested_change) >= max(min_trade, 1e-12)),
        requested_change, 0.0,
    )
    new_position = position + executed_change
    spread_paid = np.sum(spread[None, :] * forward_mid * np.abs(executed_change), axis=1)
    fixed_paid = fixed_cost * np.count_nonzero(executed_change, axis=1)
    return new_position, executed_change, spread_paid, fixed_paid


def simulate_forward_hedge(
    paths: HestonPaths, book_paths: BookPaths, market_q: MarketDynamics, *,
    initial_positions: np.ndarray, actions: np.ndarray,
    expiries_days: tuple[int, ...] = (42, 84, 126),
    spread_half_pcts: tuple[float, ...] = (0.01, 0.02, 0.03),
    fixed_cost: float = 0.5, min_trade: float = 0.01,
    max_trade_increment: float = 1.0, max_position: float = 3.0,
    trading_days_per_year: int = 250,
) -> ForwardHedgeResults:
    """Evaluate prechosen vector actions under the ENIAC forward-P&L convention.

    The held forward position generates q_t * (F_{t+1}-F_t) at the *end*
    of each day. Realized differences enter the interest-bearing cash account,
    as an explicit simplified marked-to-market convention. This is NOT a
    complete contractual cashflow model of an uncollateralized OTC forward.

    The option book is marked ex-settlement; its short sale premium finances
    the initial cash account. Opening, rebalancing and mandatory expiry
    closure each incur half-spread + fixed transaction costs.
    """
    if market_q.measure != "Q" or paths.measure != "Q":
        raise ValueError("All forward-hedge trajectories and valuations must use Q")
    if not math.isfinite(market_q.risk_free_rate):
        raise ValueError("invalid interest rate")
    if trading_days_per_year < 1:
        raise ValueError("invalid trading_days_per_year")
    spot = paths.spot
    if spot.ndim != 2 or not np.all(np.isfinite(spot)) or np.any(spot <= 0):
        raise ValueError("invalid spot trajectories")
    n_paths, n_times = spot.shape
    n_steps = n_times - 1
    n_buckets = len(expiries_days)
    if (not n_buckets or tuple(sorted(set(expiries_days))) != expiries_days or
            any(type(x) is not int or x < 1 or x > n_steps for x in expiries_days)):
        raise ValueError("expiries_days must be distinct, increasing, inside the horizon")
    if (len(spread_half_pcts) != n_buckets or
            not all(math.isfinite(x) and x >= 0 for x in spread_half_pcts)):
        raise ValueError("invalid half-spreads")
    for field in ("value", "settlements"):
        data = getattr(book_paths, field)
        if data.shape != spot.shape or not np.all(np.isfinite(data)):
            raise ValueError(f"invalid book_paths.{field}")
    position0 = np.asarray(initial_positions, dtype=float)
    actions = np.asarray(actions, dtype=float)
    if position0.shape != (n_paths, n_buckets):
        raise ValueError("initial_positions shape must be (paths, buckets)")
    if actions.shape != (n_paths, n_steps, n_buckets):
        raise ValueError("actions shape must be (paths, steps, buckets)")
    if not np.all(np.isfinite(position0)) or not np.all(np.isfinite(actions)):
        raise ValueError("invalid action or initial position")
    if np.any(position0 < 0) or np.any(position0 > max_position):
        raise ValueError("initial positions outside allowed bounds")
    if not all(math.isfinite(x) and x >= 0 for x in (fixed_cost, min_trade, max_position)):
        raise ValueError("invalid cost/position settings")

    position = position0.copy()
    stored_positions = np.zeros((n_paths, n_times, n_buckets), dtype=float)
    stored_positions[:, 0, :] = position
    wealth = np.zeros_like(spot)
    hedge_pnl = np.zeros((n_paths, n_steps), dtype=float)
    trades = np.count_nonzero(position, axis=1)
    turnover = np.sum(np.abs(position), axis=1)
    f0 = forward_prices(spot[:, 0], 0, expiries_days, market_q.risk_free_rate,
                        dividend_yield=market_q.dividend_yield,
                        trading_days_per_year=trading_days_per_year)
    spread_total = np.sum(np.asarray(spread_half_pcts)[None, :] * f0 * np.abs(position), axis=1)
    fixed_total = fixed_cost * np.count_nonzero(position, axis=1)
    initial_cost = spread_total + fixed_total
    cash = -book_paths.value[:, 0] - initial_cost
    wealth[:, 0] = cash + book_paths.value[:, 0]
    growth = math.exp(market_q.risk_free_rate / trading_days_per_year)

    for day in range(n_steps):
        active = np.asarray(expiries_days) > day
        f_prev = forward_prices(spot[:, day], day, expiries_days,
                                market_q.risk_free_rate,
                                dividend_yield=market_q.dividend_yield,
                                trading_days_per_year=trading_days_per_year)
        new_position, change, spread, fixed = execute_forward_trades(
            position, actions[:, day, :], f_prev, active,
            spread_half_pcts=spread_half_pcts, fixed_cost=fixed_cost,
            min_trade=min_trade, max_trade_increment=max_trade_increment,
            max_position=max_position,
        )
        spread_total += spread
        fixed_total += fixed
        trades += np.count_nonzero(change, axis=1)
        turnover += np.sum(np.abs(change), axis=1)
        f_next = forward_prices(spot[:, day + 1], day + 1, expiries_days,
                                market_q.risk_free_rate,
                                dividend_yield=market_q.dividend_yield,
                                trading_days_per_year=trading_days_per_year)
        hedge_pnl[:, day] = np.sum(new_position * (f_next - f_prev), axis=1)
        # Trades occur at t; their costs are funded over [t, t+1].
        cash = (cash - spread - fixed) * growth + hedge_pnl[:, day]
        cash += book_paths.settlements[:, day + 1]
        expiring = np.asarray(expiries_days) == day + 1
        if np.any(expiring):
            q_close = new_position[:, expiring]
            forced_spread = np.sum(
                np.asarray(spread_half_pcts)[None, expiring]
                * spot[:, day + 1, None] * np.abs(q_close), axis=1
            )
            forced_fixed = fixed_cost * np.count_nonzero(q_close, axis=1)
            spread_total += forced_spread
            fixed_total += forced_fixed
            trades += np.count_nonzero(q_close, axis=1)
            turnover += np.sum(np.abs(q_close), axis=1)
            cash -= forced_spread + forced_fixed
            new_position[:, expiring] = 0.0
        position = new_position
        stored_positions[:, day + 1] = position
        wealth[:, day + 1] = cash + book_paths.value[:, day + 1]
    return ForwardHedgeResults(
        wealth, wealth[:, -1], stored_positions, trades, turnover,
        spread_total + fixed_total, spread_total, fixed_total, initial_cost, hedge_pnl,
    )
