"""SAC-Band contract tests against the same forward financial engine as Trade."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from band_policy import (
    BAND_EXECUTION, ForwardBandEpisode, HestonForwardBandGymEnv,
    band_to_increment, forward_delta_target, require_current_band_execution,
)
from forward_experiments import _target_forward_positions, reward_increments
from hedging import simulate_forward_hedge
from rl_env import ForwardEpisode, prepare_forward_episodes
from train_sac_trade import (
    StreamingQBandGymEnv, _action_bounds, _episode_factory, evaluate_actions,
    experiment_config,
)


@pytest.fixture(scope="module")
def data():
    pool, base = prepare_forward_episodes(
        Path(__file__).resolve().parents[1] / "config.json", n_paths=2, seed=112233,
    )
    config = experiment_config(base, penalty=2.0,
                               max_trade_increment=base.max_position,
                               include_bucket_deltas=True)
    return pool, config


@pytest.mark.parametrize("position,target,width,expected", [
    # Inside band: hold, even if not exactly at the delta reference.
    ([1., 2., .5], [1.1, 2.1, .45], [.2, .2, .2], [0., 0., 0.]),
    # Outside band: execute FULL change back to delta target, not boundary.
    ([.2, 2.8, 1.], [2.7, .5, 1.], [.1, .2, 0.], [2.5, -2.3, 0.]),
    ([.2, 2.8, 1.], [2.7, .5, 1.], [0., 0., 0.], [2.5, -2.3, 0.]),
    ([.2, 2.8, 1.], [2.7, .5, 1.], [1.5, 1.5, 1.5], [2.5, -2.3, 0.]),
    # Exactly on boundary: remain in no-transaction zone.
    ([.8, 1.2, 0.], [1., 1., 0.], [.2, .2, .3], [0., 0., 0.]),
    # Clip band bounds to admissible positions [0, 3].
    ([.4, 2.6, 3.], [.1, 2.9, 2.], [.3, .3, .5], [0., 0., -1.]),
])
def test_center_execution_not_boundary(position, target, width, expected):
    np.testing.assert_allclose(
        band_to_increment(position, target, width, max_position=3.), expected, atol=1e-12,
    )


def test_band_semantics_metadata_protects_old_checkpoints():
    assert BAND_EXECUTION == "delta_center"
    require_current_band_execution({"band_execution": "delta_center"})
    for old in ({}, {"band_execution": "band_boundary"}):
        with pytest.raises(ValueError, match="retrain SAC-Band"):
            require_current_band_execution(old)


def test_band_rejects_invalid_values():
    with pytest.raises(ValueError):
        band_to_increment([1, 1, 1], [1, 1, 1], [-.1, .2, .3], max_position=3.)
    with pytest.raises(ValueError):
        band_to_increment([1, 1, 1], [1, 1], [0, .2, .3], max_position=3.)
    with pytest.raises(ValueError):
        band_to_increment([1, 1, 1], [1, 1, np.nan], [0, .2, .3], max_position=3.)
    with pytest.raises(ValueError):
        band_to_increment([1, 1, 1], [1, 3.01, 1], [0, .2, .3], max_position=3.)


def test_forward_delta_reference_exactly_matches_baseline(data):
    pool, cfg = data
    for day in (0, 1, 21, 42, 83, 84, 125):
        expected = _target_forward_positions(pool, cfg, day)
        for i in range(2):
            actual = forward_delta_target(pool.book_paths.bucket_delta[i, day],
                                          day=day, config=cfg, pool=pool)
            np.testing.assert_allclose(actual, expected[i], atol=1e-12)
    with pytest.raises(ValueError, match="delta"):
        forward_delta_target([1, np.nan, 1], day=0, config=cfg, pool=pool)


def test_band_reset_and_full_size_move_without_trade_increment_cap(data):
    pool, cfg = data
    e = ForwardBandEpisode(pool, cfg)
    obs, info = e.reset(path_index=0)
    assert obs.shape == (20,)
    assert info["initial_cost"] > 0
    pos = e.position[0].copy()
    target = forward_delta_target(pool.book_paths.bucket_delta[0, 0],
                                  day=0, config=cfg, pool=pool)
    _, _, done, truncated, info = e.step(np.zeros(3))
    np.testing.assert_allclose(info["requested_band_increment"], target-pos, atol=1e-12)
    np.testing.assert_allclose(info["executed_change"],
                               np.where(np.abs(target-pos) >= cfg.min_trade, target-pos, 0.0),
                               atol=1e-12)
    assert not done and not truncated
    assert np.max(np.abs(info["requested_band_increment"])) <= cfg.max_position


def test_invalid_band_actions_and_missing_delta_are_rejected(data):
    pool, cfg = data
    with pytest.raises(ValueError, match="bucket deltas"):
        ForwardBandEpisode(pool, replace(cfg, include_bucket_deltas=False))
    with pytest.raises(ValueError, match="full permitted"):
        ForwardBandEpisode(pool, replace(cfg, max_trade_increment=1.0))
    with pytest.raises(ValueError, match="max_band_width"):
        ForwardBandEpisode(pool, cfg, max_band_width=0)
    e = ForwardBandEpisode(pool, cfg)
    with pytest.raises(RuntimeError, match="reset"):
        e.step(np.zeros(3))
    e.reset(path_index=0)
    for action in ([.3, -0.1, 0], [0, 1.51, 0], [0, 0], [0, np.nan, 0]):
        with pytest.raises(ValueError):
            e.step(action)


def test_band_rollout_same_wealth_costs_and_reward_as_batch_engine(data):
    pool, cfg = data
    half_width = np.array([.25, .0, 1.5])
    summary, trace = evaluate_actions(
        pool, cfg, lambda obs: half_width.copy(),
        episode_factory=lambda p, c: ForwardBandEpisode(p, c, max_band_width=1.5),
        action_bounds=(0., 1.5),
    )
    # Independently replay the actual executed *increments* through the financial
    # batch engine; band widths themselves are not increments.
    actions = np.zeros_like(trace["actions"])
    for i in range(len(pool.paths.spot)):
        episode = ForwardBandEpisode(pool, cfg)
        episode.reset(path_index=i)
        for day in range(126):
            _, _, _, _, info = episode.step(half_width)
            actions[i, day] = info["requested_band_increment"]
    batch = simulate_forward_hedge(
        pool.paths, pool.book_paths, pool.market_q,
        initial_positions=pool.initial_positions, actions=actions,
        expiries_days=cfg.expiries_days, spread_half_pcts=cfg.spread_half_pcts,
        fixed_cost=cfg.fixed_cost, min_trade=cfg.min_trade,
        max_trade_increment=cfg.max_trade_increment, max_position=cfg.max_position,
        trading_days_per_year=cfg.trading_days_per_year,
    )
    np.testing.assert_allclose(trace["terminal_pnl"], batch.terminal_pnl, atol=1e-8)
    np.testing.assert_allclose(trace["transaction_cost"], batch.transaction_cost, atol=1e-8)
    np.testing.assert_array_equal(trace["trades"], batch.trades)
    np.testing.assert_allclose(trace["turnover"], batch.turnover, atol=1e-8)
    np.testing.assert_allclose(trace["episode_reward"],
                               (reward_increments(batch)-2*reward_increments(batch)**2).sum(axis=1),
                               atol=1e-8)
    assert np.max(trace["actions"]) <= 1.5
    np.testing.assert_allclose(trace["actions"],
                               np.broadcast_to(half_width, trace["actions"].shape), atol=1e-10)
    assert trace["executed_change"].shape == (2, 126, 3)
    assert summary["n_paths"] == 2


def test_evaluation_bounds_enforced_for_band_not_trade(data):
    pool, cfg = data
    with pytest.raises(ValueError, match="policy action exceeds"):
        evaluate_actions(pool, cfg, lambda obs: np.array([-.01, 0., 0.]),
                         episode_factory=_episode_factory("band", 1.5),
                         action_bounds=_action_bounds("band", cfg, 1.5), max_paths=1)
    with pytest.raises(ValueError, match="policy action exceeds"):
        evaluate_actions(pool, cfg, lambda obs: np.array([1.51, 0., 0.]),
                         episode_factory=_episode_factory("band", 1.5),
                         action_bounds=_action_bounds("band", cfg, 1.5), max_paths=1)


def test_gym_band_env_and_chunk_refresh_if_available(data):
    pytest.importorskip("gymnasium")
    pool, cfg = data
    from gymnasium.utils.env_checker import check_env
    env = HestonForwardBandGymEnv(pool, cfg, max_band_width=1.5)
    try:
        check_env(env, skip_render_check=True)
        obs, _ = env.reset(seed=1, options={"path_index": 0})
        assert obs.shape == (20,)
        assert env.action_space.contains(np.zeros(3, dtype=np.float32))
        assert not env.action_space.contains(np.array([-1, 0, 0], dtype=np.float32))
    finally:
        env.close()
    stream = StreamingQBandGymEnv(
        Path(__file__).resolve().parents[1] / "config.json",
        train_seed=98124, paths_per_chunk=1, n_chunks=2,
        penalty=2., max_trade_increment=cfg.max_position,
        include_bucket_deltas=True, max_band_width=1.5,
    )
    try:
        stream.reset()
        assert isinstance(stream.core, ForwardBandEpisode)
        for _ in range(126):
            _, _, done, truncated, _ = stream.step(np.zeros(3, dtype=np.float32))
        assert done and not truncated
        stream.reset()
        assert isinstance(stream.core, ForwardBandEpisode)
        assert stream.generated_chunks == 2
        assert stream.action_space.contains(np.full(3, 1.5, dtype=np.float32))
    finally:
        stream.close()
