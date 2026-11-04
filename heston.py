"""Vectorized Heston path simulation using full-truncation Euler for variance.

Risk-neutral and physical dynamics are deliberately distinguished. Supplying
P dynamics does not imply any relation between P and Q variance parameters.
This is an initial, approximate numerical method; validate convergence before
using simulated trajectories in research experiments.
"""
from dataclasses import dataclass
from typing import Literal
import math
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class HestonParams:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float

    def __post_init__(self) -> None:
        for field in ("kappa", "theta", "xi", "rho", "v0"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        if self.kappa <= 0 or self.theta < 0 or self.xi < 0 or self.v0 < 0:
            raise ValueError("Require kappa>0 and theta, xi, v0 >=0")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError("rho must be in [-1,1]")

    @property
    def feller_ratio(self) -> float:
        if self.xi == 0:
            return math.inf
        return 2.0 * self.kappa * self.theta / self.xi**2

    @property
    def satisfies_feller(self) -> bool:
        return (2.0 * self.kappa * self.theta > self.xi**2 or math.isclose(2.0 * self.kappa * self.theta, self.xi**2, rel_tol=1e-12, abs_tol=1e-14))

    def expected_variance(self, time_years: float) -> float:
        if time_years < 0:
            raise ValueError("time_years must be nonnegative")
        return self.theta + (self.v0 - self.theta) * math.exp(-self.kappa * time_years)


@dataclass(frozen=True)
class MarketDynamics:
    measure: Literal["Q", "P"]
    spot0: float
    risk_free_rate: float
    dividend_yield: float = 0.0
    physical_total_return: float | None = None

    def __post_init__(self) -> None:
        if self.measure not in ("P", "Q"):
            raise ValueError("measure must be explicitly 'P' or 'Q'")
        for name in ("spot0", "risk_free_rate", "dividend_yield"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.spot0 <= 0:
            raise ValueError("spot0 must be positive")
        if self.measure == "P":
            if self.physical_total_return is None or not math.isfinite(self.physical_total_return):
                raise ValueError("P dynamics require finite physical_total_return")
        elif self.physical_total_return is not None:
            raise ValueError("physical_total_return must be omitted for Q dynamics")

    @property
    def spot_drift(self) -> float:
        """Price drift net of dividend yield; P input is TOTAL expected return."""
        return (self.risk_free_rate if self.measure == "Q" else float(self.physical_total_return)) - self.dividend_yield


@dataclass(frozen=True)
class HestonPaths:
    time_years: NDArray[np.float64]  # (n_steps+1,)
    spot: NDArray[np.float64]        # (n_paths,n_steps+1)
    variance: NDArray[np.float64]    # nonnegative effective v=max(raw_v,0)
    raw_variance: NDArray[np.float64]  # full-truncation internal Euler state, may be negative
    measure: Literal["Q", "P"]


def simulate_heston(
    params: HestonParams,
    market: MarketDynamics,
    *,
    n_paths: int,
    n_steps: int,
    trading_days_per_year: int = 252,
    seed: int = 123,
) -> HestonPaths:
    """Simulate daily Heston samples with correlated normal shocks.

    Full-truncation variance Euler:
      v_raw(next) = v_raw + kappa(theta - max(v_raw,0))dt
                    + xi sqrt(max(v_raw,0))dW_v
    Log-Euler stock uses effective v=max(v_raw,0), preserving positive spot.
    Raw/effective variance are both saved to expose discretization effects.
    """
    if isinstance(n_paths, bool) or not isinstance(n_paths, int) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if isinstance(trading_days_per_year, bool) or not isinstance(trading_days_per_year, int) or trading_days_per_year < 1:
        raise ValueError("trading_days_per_year must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    dt = 1.0 / trading_days_per_year
    root_dt = math.sqrt(dt)
    root_independent = math.sqrt(max(0.0, 1.0 - params.rho**2))
    rng = np.random.default_rng(seed)

    spot = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    raw_v = np.empty_like(spot)
    spot[:, 0] = market.spot0
    raw_v[:, 0] = params.v0

    for t in range(n_steps):
        z_spot = rng.standard_normal(n_paths)
        z_independent = rng.standard_normal(n_paths)
        z_var = params.rho * z_spot + root_independent * z_independent
        v_eff = np.maximum(raw_v[:, t], 0.0)
        raw_v[:, t + 1] = (
            raw_v[:, t]
            + params.kappa * (params.theta - v_eff) * dt
            + params.xi * np.sqrt(v_eff) * root_dt * z_var
        )
        spot[:, t + 1] = spot[:, t] * np.exp(
            (market.spot_drift - 0.5 * v_eff) * dt
            + np.sqrt(v_eff) * root_dt * z_spot
        )

    if not np.all(np.isfinite(spot)) or not np.all(np.isfinite(raw_v)):
        raise FloatingPointError("Heston simulation produced non-finite values")

    return HestonPaths(
        time_years=np.arange(n_steps + 1, dtype=np.float64) * dt,
        spot=spot,
        variance=np.maximum(raw_v, 0.0),
        raw_variance=raw_v,
        measure=market.measure,
    )
