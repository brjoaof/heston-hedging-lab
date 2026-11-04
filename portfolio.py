"""Variable-size European option books and gross contractual expiry cashflows.

Signed quantities: positive = long option, negative = short option.
Cashflows exclude premium, hedging strategy, financing and transaction costs.
"""
from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray
from typing import Literal


@dataclass(frozen=True)
class EuropeanOption:
    contract_id: str
    quantity: float
    strike: float
    maturity_days: int
    option_type: Literal["call", "put"] = "call"
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.contract_id:
            raise ValueError("contract_id cannot be empty")
        if not math.isfinite(self.quantity) or self.quantity == 0:
            raise ValueError("quantity must be finite and nonzero")
        if not math.isfinite(self.strike) or self.strike <= 0:
            raise ValueError("strike must be positive and finite")
        if isinstance(self.maturity_days, bool) or not isinstance(self.maturity_days, int) or self.maturity_days < 1:
            raise ValueError("maturity_days must be a positive integer")
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be call or put")
        if not math.isfinite(self.multiplier) or self.multiplier <= 0:
            raise ValueError("multiplier must be positive and finite")

    def unsigned_payoff(self, spot_at_expiry: NDArray[np.float64]) -> NDArray[np.float64]:
        sign = 1.0 if self.option_type == "call" else -1.0
        return np.maximum(sign * (spot_at_expiry - self.strike), 0.0)


@dataclass(frozen=True)
class OptionBook:
    book_id: str
    contracts: tuple[EuropeanOption, ...]

    def __post_init__(self) -> None:
        if not self.book_id:
            raise ValueError("book_id cannot be empty")
        if not self.contracts:
            raise ValueError("OptionBook must have at least one contract")
        ids = [c.contract_id for c in self.contracts]
        if len(ids) != len(set(ids)):
            raise ValueError("contract IDs must be unique within a book")

    @property
    def n_contracts(self) -> int:
        return len(self.contracts)



def fixed_eniac_book() -> OptionBook:
    """Nine short European calls: K=96/100/104 at 42/84/126 trading days.

    The ENIAC code represents the liability as positive option values and
    subtracts them from hedging P&L. Here the liability is an explicit short
    option book (signed quantity -1) in the existing accounting convention.
    """
    return OptionBook(
        book_id="eniac-fixed-calls",
        contracts=tuple(
            EuropeanOption(
                contract_id=f"call-{maturity}d-k{strike}",
                quantity=-1.0,
                strike=float(strike),
                maturity_days=maturity,
            )
            for maturity in (42, 84, 126)
            for strike in (96, 100, 104)
        ),
    )

def sample_books(
    *,
    spot_reference: float,
    n_books: int,
    sizes: tuple[int, ...] = (1, 3, 5, 9),
    expiries_days: tuple[int, ...] = (21, 42, 63, 84, 126),
    strike_moneyness_range: tuple[float, float] = (0.85, 1.15),
    max_abs_quantity: int = 3,
    seed: int = 124,
) -> list[OptionBook]:
    """Generate reproducible portfolios of European calls."""
    lo, hi = strike_moneyness_range
    if not math.isfinite(spot_reference) or spot_reference <= 0:
        raise ValueError("spot_reference must be positive")
    if isinstance(n_books, bool) or not isinstance(n_books, int) or n_books < 1:
        raise ValueError("n_books must be a positive integer")
    if not sizes or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in sizes):
        raise ValueError("sizes must contain positive integers")
    if not expiries_days or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in expiries_days):
        raise ValueError("expiries_days must contain positive integers")
    if not (0 < lo <= hi and math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError("invalid strike moneyness range")
    if isinstance(max_abs_quantity, bool) or not isinstance(max_abs_quantity, int) or max_abs_quantity < 1:
        raise ValueError("max_abs_quantity must be a positive integer")
    rng = np.random.default_rng(seed)
    results = []
    for book_index in range(n_books):
        n = int(rng.choice(sizes))
        contracts = []
        for j in range(n):
            magnitude = int(rng.integers(1, max_abs_quantity + 1))
            quantity = magnitude * (1 if rng.random() < 0.5 else -1)
            strike = float(spot_reference * rng.uniform(lo, hi))
            expiry = int(rng.choice(expiries_days))
            contracts.append(EuropeanOption(
                contract_id=f"b{book_index:04d}-c{j:03d}",
                quantity=quantity,
                strike=strike,
                maturity_days=expiry,
            ))
        results.append(OptionBook(book_id=f"book-{book_index:04d}", contracts=tuple(contracts)))
    return results


def expiry_cashflows(book: OptionBook, spot_paths: NDArray[np.float64]) -> NDArray[np.float64]:
    """Signed gross payoffs at every expiry date (no premium/financing/costs).

    Inputs have shape (n_paths, n_steps+1), with one step per trading day.
    Output same shape. No annualization/discounting is applied here.
    """
    spots = np.asarray(spot_paths, dtype=np.float64)
    if spots.ndim != 2 or spots.shape[1] < 2 or not np.all(np.isfinite(spots)) or np.any(spots <= 0):
        raise ValueError("spot_paths must be finite positive 2D array with at least two time columns")
    cashflow = np.zeros_like(spots)
    max_step = spots.shape[1] - 1
    for contract in book.contracts:
        if contract.maturity_days > max_step:
            raise ValueError(f"{contract.contract_id} expires beyond simulated horizon")
        cashflow[:, contract.maturity_days] += (
            contract.quantity * contract.multiplier *
            contract.unsigned_payoff(spots[:, contract.maturity_days])
        )
    return cashflow
