"""Pilot controls and independent financial rollback tests, before actual SAC training."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from forward_experiments import reward_increments
from hedging import simulate_forward_hedge
from rl_env import ForwardEpisode, prepare_forward_episodes
from train_sac_trade import (
    evaluate_actions, experiment_config, StreamingQForwardGymEnv,
    _checkpoint_stem, _validate_model_stem, _save_checkpoint,
)


@pytest.fixture(scope="module")
def small_q_pool():
    config = Path(__file__).resolve().parents[1] / "config.json"
    pool, cfg = prepare_forward_episodes(config, n_paths=2, seed=808808)
    return pool, cfg


def test_pilot_policy_observes_same_deltas_that_band_will_use(small_q_pool):
    pool, base = small_q_pool
    cfg = experiment_config(base, penalty=2.0, max_trade_increment=1.0)
    episode = ForwardEpisode(pool, cfg)
    obs, _ = episode.reset(path_index=0)
    assert len(obs) == 20
    np.testing.assert_allclose(obs[-3:], pool.book_paths.bucket_delta[0, 0], atol=2e-6)
    assert cfg.include_bucket_deltas
    assert base.include_bucket_deltas is False  # legacy/educational environment unchanged


def test_legacy_observation_unchanged_and_three_unit_trade(small_q_pool):
    pool, base = small_q_pool
    widened = experiment_config(base, penalty=2., max_trade_increment=3.,
                                include_bucket_deltas=False)
    episode = ForwardEpisode(pool, widened)
    obs, _ = episode.reset(path_index=0)
    assert obs.shape == (17,)
    _, _, _, _, info = episode.step(np.array([-3., 3., -3.]))
    # At most 3 units are executable in one rebalance, within [0,3].
    assert np.max(np.abs(info["executed_change"])) <= 3.
    assert np.all(episode.position >= 0) and np.all(episode.position <= 3.)


@pytest.mark.parametrize("bad_penalty,bad_increment", [(-1, 1), (0, 0), (0, 4), (np.nan, 1)])
def test_invalid_experiment_controls_are_rejected(small_q_pool, bad_penalty, bad_increment):
    _, base = small_q_pool
    with pytest.raises(ValueError):
        experiment_config(base, penalty=bad_penalty, max_trade_increment=bad_increment)


def test_zero_action_matches_independent_financial_engine_and_reward(small_q_pool):
    pool, base = small_q_pool
    cfg = experiment_config(base, penalty=2., max_trade_increment=1.)
    summary, trace = evaluate_actions(pool, cfg, lambda _: np.zeros(3))
    expected = simulate_forward_hedge(
        pool.paths, pool.book_paths, pool.market_q,
        initial_positions=pool.initial_positions,
        actions=np.zeros((2, 126, 3)),
        expiries_days=cfg.expiries_days,
        spread_half_pcts=cfg.spread_half_pcts, fixed_cost=cfg.fixed_cost,
        min_trade=cfg.min_trade, max_trade_increment=cfg.max_trade_increment,
        max_position=cfg.max_position, trading_days_per_year=cfg.trading_days_per_year,
    )
    np.testing.assert_allclose(trace["terminal_pnl"], expected.terminal_pnl, atol=1e-9)
    np.testing.assert_allclose(trace["transaction_cost"], expected.transaction_cost, atol=1e-9)
    np.testing.assert_allclose(trace["trades"], expected.trades)
    assert trace["actions"].shape == (2, 126, 3)
    increments = reward_increments(expected)
    target_reward = (increments - 2. * increments**2).sum(axis=1)
    np.testing.assert_allclose(trace["episode_reward"], target_reward, atol=1e-7)
    assert summary["mean_episode_reward"] == pytest.approx(target_reward.mean())


def test_path_count_and_actions_guarded(small_q_pool):
    pool, base = small_q_pool
    cfg = experiment_config(base, penalty=0, max_trade_increment=1)
    with pytest.raises(ValueError, match="max_paths"):
        evaluate_actions(pool, cfg, lambda obs: np.zeros(3), max_paths=3)
    with pytest.raises(ValueError, match="policy action exceeds"):
        evaluate_actions(pool, cfg, lambda obs: np.array([2.,0.,0.]), max_paths=1)
    with pytest.raises(ValueError, match="policy must return"):
        evaluate_actions(pool, cfg, lambda obs: np.zeros(2), max_paths=1)


def test_streaming_env_uses_new_q_chunk_only_at_episode_boundary():
    pytest.importorskip("gymnasium")
    config = Path(__file__).resolve().parents[1] / "config.json"
    env = StreamingQForwardGymEnv(config, train_seed=99099,
                                paths_per_chunk=1, n_chunks=2,
                                penalty=2.0, max_trade_increment=1.0,
                                include_bucket_deltas=True)
    try:
        env.reset()
        assert env.generated_chunks == 1
        for _ in range(126):
            _, _, done, truncated, _ = env.step(np.zeros(3))
        assert done and not truncated
        env.reset()
        assert env.generated_chunks == 2
        assert env.chunk_index == 1
        assert env.observation_space.shape == (20,)
    finally:
        env.close()


def test_evaluation_checkpoint_names_are_sorted_and_safe():
    assert _checkpoint_stem(5000) == "step_000000005000"
    assert _checkpoint_stem(30000) == "step_000000030000"
    assert _validate_model_stem("best") == "best"
    assert _validate_model_stem("final") == "final"
    assert _validate_model_stem("step_000000005000") == "step_000000005000"
    for name in ("../best", "step_5000", "../../etc/passwd", "step_000000005000/other"):
        with pytest.raises(ValueError):
            _validate_model_stem(name)
    with pytest.raises(ValueError):
        _checkpoint_stem(0)


def test_checkpoint_saves_model_and_matching_normalization(tmp_path):
    class FakeModel:
        def save(self, path):
            Path(str(path) + ".zip").write_text("model", encoding="utf-8")

    class FakeVecNorm:
        def save(self, path):
            Path(path).write_text("obs moments", encoding="utf-8")

    _save_checkpoint(FakeModel(), FakeVecNorm(), tmp_path, _checkpoint_stem(5000))
    assert (tmp_path / "step_000000005000.zip").read_text() == "model"
    assert (tmp_path / "step_000000005000.vecnormalize.pkl").read_text() == "obs moments"
