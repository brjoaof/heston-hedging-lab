"""No-transaction band action for the *existing* Heston-Q forward hedge.

Band actions are nonnegative half-widths, NOT changes in forward positions.
The target uses the current Q option-book delta and forward sensitivity, just
like the established delta baselines. All trading, costs and settlements still
happen in ForwardEpisode.step -> advance_forward_hedge (hedging.py).
"""
from __future__ import annotations

import math

import numpy as np

from rl_env import (ForwardEnvConfig, ForwardEpisode, ForwardEpisodePool,
                    HestonForwardGymEnv)

# Execution rule is part of the policy/MDP, not merely evaluation metadata.
BAND_EXECUTION = "delta_center"


def require_current_band_execution(metadata: dict) -> None:
    """Prevent replaying legacy boundary-trained policies under a changed MDP."""
    if metadata.get("band_execution") != BAND_EXECUTION:
        raise ValueError(
            "Incompatible SAC-Band execution: expected delta_center. "
            "Older runs without band_execution used band_boundary and cannot "
            "be evaluated or compared using the new rule; retrain SAC-Band."
        )


def forward_delta_target(
    signed_option_bucket_delta: np.ndarray, *, day: int,
    config: ForwardEnvConfig, pool: ForwardEpisodePool,
) -> np.ndarray:
    """Q-forward positions hedging the current signed option bucket deltas.

    dF/dS = exp((r-q)*remaining_time). The short option book has negative
    delta; the hedge therefore has *positive* forward exposure. No future
    simulated values enter this calculation.
    """
    delta = np.asarray(signed_option_bucket_delta, dtype=float)
    if delta.shape != (len(config.expiries_days),) or not np.all(np.isfinite(delta)):
        raise ValueError("one finite current option delta per maturity is required")
    expiries = np.asarray(config.expiries_days, dtype=float)
    remaining = np.maximum(expiries - day, 0.0) / config.trading_days_per_year
    sensitivity = np.exp((pool.market_q.risk_free_rate - pool.market_q.dividend_yield) * remaining)
    target = np.clip(-delta / sensitivity, 0.0, config.max_position)
    target[expiries <= day] = 0.0
    return target


def band_to_increment(
    position: np.ndarray, forward_target: np.ndarray, half_width: np.ndarray,
    *, max_position: float,
) -> np.ndarray:
    """Trade to the forward-delta target outside the band; hold inside."""
    pos = np.asarray(position, dtype=float)
    target = np.asarray(forward_target, dtype=float)
    width = np.asarray(half_width, dtype=float)
    if not (pos.ndim == 1 and pos.shape == target.shape == width.shape):
        raise ValueError("position, delta target and band must be matching vectors")
    if not all(np.all(np.isfinite(v)) for v in (pos, target, width)):
        raise ValueError("position, delta target and band must be finite")
    if (not math.isfinite(max_position) or max_position <= 0 or
            np.any(width < 0) or np.any(pos < 0) or np.any(pos > max_position) or
            np.any(target < 0) or np.any(target > max_position)):
        raise ValueError("invalid band width or current position")
    center = np.clip(target, 0.0, max_position)
    lower = np.maximum(0.0, center - width)
    upper = np.minimum(max_position, center + width)
    outside = (pos < lower) | (pos > upper)
    desired = np.where(outside, center, pos)
    return desired - pos


class ForwardBandEpisode(ForwardEpisode):
    """Financially identical ForwardEpisode; only interprets SAC actions differently."""

    def __init__(self, pool: ForwardEpisodePool, config: ForwardEnvConfig, *,
                 max_band_width: float = 1.5, seed: int = 123):
        if not math.isfinite(max_band_width) or not 0 < max_band_width <= config.max_position:
            raise ValueError("max_band_width must be positive and finite")
        if not config.include_bucket_deltas:
            raise ValueError("SAC-Band requires current bucket deltas in observations")
        if config.max_trade_increment < config.max_position:
            raise ValueError("SAC-Band execution requires full permitted position range")
        super().__init__(pool, config, seed=seed)
        self.max_band_width = float(max_band_width)

    def step(self, action: np.ndarray):
        widths = np.asarray(action, dtype=float)
        if widths.shape != (len(self.config.expiries_days),) or not np.all(np.isfinite(widths)):
            raise ValueError("SAC-Band action must contain one finite width per maturity")
        if np.any(widths < -1e-6) or np.any(widths > self.max_band_width + 1e-6):
            raise ValueError("SAC-Band action outside [0, max_band_width]")
        if self.path_index is None or self.position is None:
            raise RuntimeError("reset() must precede step()")
        widths = np.clip(widths, 0.0, self.max_band_width)
        signed_delta = self.pool.book_paths.bucket_delta[self.path_index, self.day]
        target = forward_delta_target(signed_delta, day=self.day, config=self.config, pool=self.pool)
        increment = band_to_increment(self.position[0], target, widths,
                                      max_position=self.config.max_position)
        obs, reward, terminated, truncated, info = super().step(increment)
        info["band_width"] = widths.copy()
        info["forward_delta_target"] = target
        info["requested_band_increment"] = increment
        return obs, reward, terminated, truncated, info


class HestonForwardBandGymEnv(HestonForwardGymEnv):
    """Gymnasium SAC-Band adapter: action in [0, max_band_width] for each bucket."""

    def __init__(self, pool: ForwardEpisodePool, config: ForwardEnvConfig, *,
                 max_band_width: float = 1.5, seed: int = 123):
        super().__init__(pool, config, seed=seed)
        self.core = ForwardBandEpisode(pool, config, max_band_width=max_band_width, seed=seed)
        from gymnasium import spaces
        self.action_space = spaces.Box(low=0.0, high=max_band_width,
                                       shape=(len(config.expiries_days),), dtype=np.float32)
