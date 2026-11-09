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
from pricing import heston_greeks_batch, heston_greeks_strikes_batch


def _pricing_backend(*, method: str, cos_n_terms: int, cos_L: float):
    """Pick one pricing oracle for *all* marks and initial hedge deltas.

    Fourier remains the historical default; COS is opt-in. No implicit
    cross-method fallback is allowed, so datasets cannot silently mix them.
    """
    if method == "fourier":
        return heston_greeks_strikes_batch, {}
    if method == "cos":
        from pricing_cos import heston_cos_greeks_strikes_batch
        if type(cos_n_terms) is not int or cos_n_terms < 16:
            raise ValueError("cos_n_terms must be an integer >= 16")
        if not math.isfinite(cos_L) or cos_L <= 0:
            raise ValueError("cos_L must be positive and finite")
        return heston_cos_greeks_strikes_batch, {
            "n_terms": cos_n_terms, "L": cos_L,
        }
    raise ValueError("pricing method must be 'fourier' or 'cos'")


@dataclass(frozen=True)
class BookPaths:
    value: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    variance_sensitivity: np.ndarray
    settlements: np.ndarray
    contract_prices: np.ndarray | None = None  # optional unsigned mid marks (paths, times, contracts)
    bucket_delta: np.ndarray | None = None  # signed option delta by increasing maturity (paths, times, maturities)


def value_book_paths(
    book: OptionBook, paths: HestonPaths, params_q: HestonParams,
    market_q: MarketDynamics, *, trading_days_per_year: int = 252,
    include_contract_prices: bool = False,
    include_bucket_deltas: bool = False,
    pricing_method: str = "fourier", cos_n_terms: int = 512, cos_L: float = 24.0,
) -> BookPaths:
    """Revalue each *live* option on physical or Q paths, always using Q prices.

    Values are ex-settlement: a contract that expires on day t contributes
    zero to value[t], and its payoff enters settlements[t] exactly once.
    """
    if market_q.measure != "Q":
        raise ValueError("Option valuation requires Q market parameters")
    if trading_days_per_year <= 0:
        raise ValueError("Invalid trading_days_per_year")
    pricing_fn, pricing_kwargs = _pricing_backend(
        method=pricing_method, cos_n_terms=cos_n_terms, cos_L=cos_L,
    )
    values = [np.zeros_like(paths.spot) for _ in range(4)]
    settlements = expiry_cashflows(book, paths.spot)
    n_days = paths.spot.shape[1]
    contract_prices = (np.zeros((*paths.spot.shape, book.n_contracts), dtype=float)
                       if include_contract_prices else None)
    maturities = tuple(sorted({c.maturity_days for c in book.contracts}))
    bucket_delta = (np.zeros((*paths.spot.shape, len(maturities)), dtype=float)
                    if include_bucket_deltas else None)
    maturity_index = {m: i for i, m in enumerate(maturities)}
    # Group contracts with the same maturity and option type. Their Fourier
    # characteristic-function terms are identical except for the strike phase.
    groups: dict[tuple[int, str], list[tuple[int, object]]] = {}
    for contract_index, contract in enumerate(book.contracts):
        if contract.maturity_days >= n_days:
            raise ValueError("Option matures after the simulated horizon")
        groups.setdefault((contract.maturity_days, contract.option_type), []).append((contract_index, contract))
    for (maturity_days, option_type), contracts in groups.items():
        strikes = np.asarray([contract.strike for _, contract in contracts], dtype=float)
        for day in range(maturity_days):
            remaining = (maturity_days-day)/trading_days_per_year
            greeks = pricing_fn(
                params_q, market_q, strikes=strikes, maturity=remaining,
                spots=paths.spot[:,day], variances=paths.variance[:,day],
                option_type=option_type, **pricing_kwargs,
            )
            for j, (contract_index, contract) in enumerate(contracts):
                amount = contract.quantity*contract.multiplier
                if contract_prices is not None:
                    contract_prices[:,day,contract_index] = greeks.price[:,j]
                if bucket_delta is not None:
                    bucket_delta[:,day,maturity_index[maturity_days]] += amount*greeks.delta[:,j]
                for target, field in zip(values,("price","delta","gamma","variance_sensitivity")):
                    target[:,day] += amount*getattr(greeks,field)[:,j]
    return BookPaths(*values, settlements, contract_prices, bucket_delta)


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
    pricing_method: str = "fourier", cos_n_terms: int = 512, cos_L: float = 24.0,
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
    pricing_fn, pricing_kwargs = _pricing_backend(
        method=pricing_method, cos_n_terms=cos_n_terms, cos_L=cos_L,
    )
    buckets = {m: i for i, m in enumerate(expiries_days)}
    delta = np.zeros((len(paths.spot), len(expiries_days)), dtype=float)
    # Group by expiry/type to share characteristic-function terms across strikes.
    groups: dict[tuple[int, str], list[object]] = {}
    for contract in book.contracts:
        if contract.maturity_days not in buckets:
            raise ValueError("option maturity not covered by forward instruments")
        groups.setdefault((contract.maturity_days, contract.option_type), []).append(contract)
    for (maturity_days, option_type), contracts in groups.items():
        strikes = np.array([c.strike for c in contracts], dtype=float)
        g = pricing_fn(
            params_q, market_q, strikes=strikes,
            maturity=maturity_days / trading_days_per_year,
            spots=paths.spot[:, 0], variances=paths.variance[:, 0],
            option_type=option_type, **pricing_kwargs,
        )
        for j, c in enumerate(contracts):
            delta[:, buckets[maturity_days]] += c.quantity * c.multiplier * g.delta[:, j]
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



@dataclass(frozen=True)
class ForwardStep:
    """The cash/position transition for one interval, shared by RL and batch evaluation."""
    position: np.ndarray            # after mandatory close at day+1
    cash: np.ndarray                # financed cash after settlements and closes
    hedge_pnl: np.ndarray
    executed_change: np.ndarray    # requested trades executed at day
    forced_close: np.ndarray       # positive quantity liquidated at day+1
    spread_cost: np.ndarray        # regular + forced closure
    fixed_cost: np.ndarray
    trade_count: np.ndarray
    turnover: np.ndarray


def forward_initial_cash(
    option_value0: np.ndarray, initial_positions: np.ndarray,
    spot0: np.ndarray, market_q: MarketDynamics, *,
    expiries_days: tuple[int, ...] = (42, 84, 126),
    spread_half_pcts: tuple[float, ...] = (0.01, 0.02, 0.03),
    fixed_cost: float = 0.5,
    trading_days_per_year: int = 250,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Short-book premium finances cash; pay opening costs at time zero."""
    if market_q.measure != "Q":
        raise ValueError("Forward hedge requires Q market")
    position = np.asarray(initial_positions, dtype=float)
    spot = np.asarray(spot0, dtype=float)
    option_value = np.asarray(option_value0, dtype=float)
    if position.ndim != 2 or spot.shape != (len(position),) or option_value.shape != spot.shape:
        raise ValueError("invalid initial cash inputs shape")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(option_value)):
        raise ValueError("initial positions and book values must be finite")
    spread = np.asarray(spread_half_pcts, dtype=float)
    if spread.shape != (position.shape[1],) or not np.all(np.isfinite(spread)) or np.any(spread < 0):
        raise ValueError("invalid spreads")
    if not math.isfinite(fixed_cost) or fixed_cost < 0:
        raise ValueError("invalid fixed_cost")
    f0 = forward_prices(spot, 0, expiries_days, market_q.risk_free_rate,
                        dividend_yield=market_q.dividend_yield,
                        trading_days_per_year=trading_days_per_year)
    spread_open = np.sum(spread[None, :] * f0 * np.abs(position), axis=1)
    fixed_open = fixed_cost * np.count_nonzero(position, axis=1)
    opening_cost = spread_open + fixed_open
    return -option_value - opening_cost, opening_cost, spread_open, fixed_open


def advance_forward_hedge(
    position: np.ndarray, cash: np.ndarray,
    spot_now: np.ndarray, spot_next: np.ndarray,
    settlement_next: np.ndarray, action: np.ndarray,
    market_q: MarketDynamics, *, day: int,
    expiries_days: tuple[int, ...] = (42, 84, 126),
    spread_half_pcts: tuple[float, ...] = (0.01, 0.02, 0.03),
    fixed_cost: float = 0.5, min_trade: float = 0.01,
    max_trade_increment: float = 1.0, max_position: float = 3.0,
    trading_days_per_year: int = 250,
) -> ForwardStep:
    """Pure, causal transition: trade at t, hold to t+1, settle/close at t+1.

    Uses the ENIAC simplified daily forward-price-difference convention, not
    a complete OTC forward contract settlement model. All inputs are batches.
    """
    if market_q.measure != "Q":
        raise ValueError("Forward hedge requires Q market")
    pos = np.asarray(position, dtype=float)
    cash = np.asarray(cash, dtype=float)
    sn = np.asarray(spot_now, dtype=float)
    sx = np.asarray(spot_next, dtype=float)
    settle = np.asarray(settlement_next, dtype=float)
    if pos.ndim != 2 or any(v.shape != (len(pos),) for v in (cash, sn, sx, settle)):
        raise ValueError("invalid state or cashflow shapes")
    if not (np.all(np.isfinite(cash)) and np.all(np.isfinite(settle))):
        raise ValueError("cash and settlement must be finite")
    if day < 0:
        raise ValueError("day must be nonnegative")
    forward_now = forward_prices(
        sn, day, expiries_days, market_q.risk_free_rate,
        dividend_yield=market_q.dividend_yield,
        trading_days_per_year=trading_days_per_year,
    )
    forward_next = forward_prices(
        sx, day + 1, expiries_days, market_q.risk_free_rate,
        dividend_yield=market_q.dividend_yield,
        trading_days_per_year=trading_days_per_year,
    )
    active = np.asarray(expiries_days) > day
    new_pos, executed, spread, fixed = execute_forward_trades(
        pos, action, forward_now, active, spread_half_pcts=spread_half_pcts,
        fixed_cost=fixed_cost, min_trade=min_trade,
        max_trade_increment=max_trade_increment, max_position=max_position,
    )
    pnl = np.sum(new_pos * (forward_next - forward_now), axis=1)
    growth = math.exp(market_q.risk_free_rate / trading_days_per_year)
    new_cash = (cash - spread - fixed) * growth + pnl + settle
    trade_count = np.count_nonzero(executed, axis=1)
    turnover = np.sum(np.abs(executed), axis=1)
    forced_close = np.zeros_like(new_pos)
    expiring = np.asarray(expiries_days) == day + 1
    if np.any(expiring):
        forced_close[:, expiring] = new_pos[:, expiring]
        spread_close = np.sum(
            np.asarray(spread_half_pcts)[None, expiring]
            * sx[:, None] * np.abs(forced_close[:, expiring]), axis=1,
        )
        fixed_close = fixed_cost * np.count_nonzero(forced_close, axis=1)
        new_cash -= spread_close + fixed_close
        spread += spread_close
        fixed += fixed_close
        trade_count += np.count_nonzero(forced_close, axis=1)
        turnover += np.sum(np.abs(forced_close), axis=1)
        new_pos[:, expiring] = 0.0
    return ForwardStep(new_pos, new_cash, pnl, executed, forced_close,
                       spread, fixed, trade_count, turnover)

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
    cash, initial_cost, spread_total, fixed_total = forward_initial_cash(
        book_paths.value[:, 0], position, spot[:, 0], market_q,
        expiries_days=expiries_days, spread_half_pcts=spread_half_pcts,
        fixed_cost=fixed_cost, trading_days_per_year=trading_days_per_year,
    )
    wealth[:, 0] = cash + book_paths.value[:, 0]
    for day in range(n_steps):
        step = advance_forward_hedge(
            position, cash, spot[:, day], spot[:, day + 1],
            book_paths.settlements[:, day + 1], actions[:, day, :],
            market_q, day=day, expiries_days=expiries_days,
            spread_half_pcts=spread_half_pcts, fixed_cost=fixed_cost,
            min_trade=min_trade, max_trade_increment=max_trade_increment,
            max_position=max_position,
            trading_days_per_year=trading_days_per_year,
        )
        position, cash = step.position, step.cash
        spread_total += step.spread_cost
        fixed_total += step.fixed_cost
        trades += step.trade_count
        turnover += step.turnover
        hedge_pnl[:, day] = step.hedge_pnl
        stored_positions[:, day + 1] = position
        wealth[:, day + 1] = cash + book_paths.value[:, day + 1]
    return ForwardHedgeResults(
        wealth, wealth[:, -1], stored_positions, trades, turnover,
        spread_total + fixed_total, spread_total, fixed_total, initial_cost, hedge_pnl,
    )
