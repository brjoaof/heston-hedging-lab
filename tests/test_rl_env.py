"""Contract tests: the RL episode must match the independent batch evaluator."""
import math

import numpy as np
import pytest

from hedging import BookPaths, simulate_forward_hedge
from heston import HestonPaths, MarketDynamics
from rl_env import ForwardEnvConfig, ForwardEpisode, ForwardEpisodePool


def _synthetic_pool(*, rate=math.log(1.1), measure="Q"):
    spots = np.array([
        [100., 106., 104., 110.],
        [100.,  98., 105., 102.],
    ])
    vars_ = np.array([[.04, .05, .07, .10], [.04, .06, .04, .03]])
    n, t = spots.shape
    paths = HestonPaths(np.arange(t)/250, spots, vars_, vars_.copy(), measure)
    # Fabricated deterministic marks let us exercise settlement, cash,
    # trading, and reward without using the Heston pricing implementation.
    contract_mid = np.zeros((n, t, 9))
    for j in range(9):
        expiry = (1, 2, 3)[j // 3]
        for day in range(expiry):
            contract_mid[:, day, j] = 1.0 + (j + 1) * (0.04 + 0.01 * day)
    value = -contract_mid.sum(axis=2)
    settlements = np.array([[0., -4., -8., -6.], [0., 0., -2., -1.]])
    zeros = np.zeros_like(spots)
    marks = BookPaths(value, zeros, zeros, zeros, settlements, contract_mid)
    positions = np.array([[1.2, 2.1, .7], [0.9, .2, 1.3]])
    market = MarketDynamics("Q", 100., rate)
    cfg = ForwardEnvConfig(expiries_days=(1,2,3),
                           spread_half_pcts=(.01, .02, .03),
                           fixed_cost=.5, min_trade=.01,
                           trading_days_per_year=250)
    return ForwardEpisodePool(paths, marks, positions, market), cfg


def test_rl_financial_path_identical_to_batch_engine():
    pool, config = _synthetic_pool()
    actions = np.array([
        [[.3, -.5, 1.0], [0., .004, -.2], [.25, .6, -.8]],
        [[0., .25, -.4], [0., -.05, .4], [-.8, .3, 0.]],
    ])
    expected = simulate_forward_hedge(
        pool.paths, pool.book_paths, pool.market_q,
        initial_positions=pool.initial_positions, actions=actions,
        expiries_days=config.expiries_days,
        spread_half_pcts=config.spread_half_pcts,
        fixed_cost=config.fixed_cost, min_trade=config.min_trade,
        max_trade_increment=config.max_trade_increment,
        max_position=config.max_position,
        trading_days_per_year=config.trading_days_per_year,
    )
    episode = ForwardEpisode(pool, config, seed=13)
    for idx in range(len(pool.paths.spot)):
        initial_obs, initial_info = episode.reset(path_index=idx)
        assert initial_obs.shape == (17,)
        assert initial_obs.dtype == np.float32
        assert initial_info["initial_cost"] == pytest.approx(expected.initial_cost[idx])
        np.testing.assert_allclose(initial_obs[5:8], expected.position[idx, 0], atol=1e-6)
        rewards = []
        for step in range(3):
            next_obs, reward, terminated, truncated, info = episode.step(actions[idx, step])
            rewards.append(reward)
            assert terminated == (step == 2)
            assert not truncated
            assert info["day"] == step + 1
            assert info["wealth"] == pytest.approx(expected.wealth[idx, step + 1])
            assert info["hedge_pnl"] == pytest.approx(expected.hedge_pnl[idx, step])
            np.testing.assert_allclose(next_obs[5:8], expected.position[idx, step + 1], atol=1e-6)
        assert sum(rewards) == pytest.approx(expected.terminal_pnl[idx])
        assert info["total_transaction_cost"] == pytest.approx(expected.transaction_cost[idx])
        assert info["trade_count"] == expected.trades[idx]
        assert info["turnover"] == pytest.approx(expected.turnover[idx])
        assert info["terminal_pnl"] == pytest.approx(expected.terminal_pnl[idx])


def test_no_future_information_in_observation_or_first_step():
    pool, config = _synthetic_pool()
    first = ForwardEpisode(pool, config)
    obs0, _ = first.reset(path_index=0)
    obs1, rew1, _, _, info1 = first.step(np.zeros(3))

    spot2 = pool.paths.spot.copy()
    spot2[:, 2:] *= 3.0
    var2 = pool.paths.variance.copy()
    var2[:, 2:] = .6
    future_marks = pool.book_paths.contract_prices.copy()
    future_marks[:, 2:] *= 4.0
    values2 = pool.book_paths.value.copy()
    values2[:, 2:] *= 4.0
    future_settlement = pool.book_paths.settlements.copy()
    future_settlement[:, 2:] -= 50
    changed = ForwardEpisodePool(
        HestonPaths(pool.paths.time_years, spot2, var2, var2.copy(), "Q"),
        BookPaths(values2, pool.book_paths.delta, pool.book_paths.gamma,
                  pool.book_paths.variance_sensitivity, future_settlement, future_marks),
        pool.initial_positions, pool.market_q,
    )
    second = ForwardEpisode(changed, config)
    alt0, _ = second.reset(path_index=0)
    alt1, alt_reward, _, _, alt_info = second.step(np.zeros(3))
    np.testing.assert_array_equal(alt0, obs0)
    np.testing.assert_array_equal(alt1, obs1)
    assert alt_reward == pytest.approx(rew1)
    assert alt_info["wealth"] == pytest.approx(info1["wealth"])


def test_variance_observable_explicitly_optional():
    pool, cfg = _synthetic_pool()
    assert ForwardEpisode(pool, cfg).reset(path_index=1)[0].shape == (17,)
    cfg_no_var = ForwardEnvConfig(expiries_days=(1,2,3), include_variance=False)
    o, _ = ForwardEpisode(pool, cfg_no_var).reset(path_index=1)
    assert o.shape == (16,)
    assert o[0] == pytest.approx(100.)


def test_reward_quadratic_penalty_and_opening_cost():
    pool, cfg = _synthetic_pool(rate=0.0)
    penalized = ForwardEnvConfig(
        expiries_days=(1,2,3), spread_half_pcts=(.01, .02, .03),
        quadratic_reward_penalty=.001)
    episode = ForwardEpisode(pool, penalized)
    _, info = episode.reset(path_index=0)
    assert info["initial_cost"] > 0
    _, reward, _, _, step_info = episode.step(np.zeros(3))
    pnl = step_info["pnl_step"]
    assert reward == pytest.approx(pnl - .001*pnl*pnl)


def test_reject_bad_action_and_step_after_done():
    pool, config = _synthetic_pool()
    env = ForwardEpisode(pool, config)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(3))
    env.reset(path_index=0)
    with pytest.raises(ValueError, match="finite"):
        env.step(np.array([np.nan, 0., 0.]))
    with pytest.raises(ValueError, match="increment"):
        env.step(np.zeros(2))
    for _ in range(3):
        env.step(np.zeros(3))
    with pytest.raises(RuntimeError, match="terminated"):
        env.step(np.zeros(3))


def test_reseed_episode_selection_is_reproducible():
    pool, config = _synthetic_pool()
    e = ForwardEpisode(pool, config)
    e.set_seed(29)
    seq1 = [e.reset()[1]["path_index"] for _ in range(12)]
    e.set_seed(29)
    seq2 = [e.reset()[1]["path_index"] for _ in range(12)]
    assert seq1 == seq2
    with pytest.raises(ValueError, match="path_index"):
        e.reset(path_index=90)


def test_reject_physical_trajectories():
    pool, config = _synthetic_pool()
    paths = HestonPaths(pool.paths.time_years, pool.paths.spot,
                        pool.paths.variance, pool.paths.raw_variance, "P")
    with pytest.raises(ValueError, match="Q"):
        ForwardEpisode(ForwardEpisodePool(paths, pool.book_paths,
                                          pool.initial_positions, pool.market_q), config)


def test_gymnasium_wrapper_if_installed():
    gym = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env
    from rl_env import HestonForwardGymEnv
    pool, cfg = _synthetic_pool()
    env = HestonForwardGymEnv(pool, cfg)
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=8, options={"path_index": 0})
    assert env.observation_space.contains(obs)
    assert env.action_space.contains(np.zeros(3, dtype=np.float32))
    env.close()


def test_real_q_heston_rollout_and_per_contract_marks():
    from pathlib import Path
    from rl_env import prepare_forward_episodes
    cfg_path = Path(__file__).resolve().parents[1] / "config.json"
    pool, cfg = prepare_forward_episodes(cfg_path, n_paths=1, seed=1001)
    assert pool.paths.measure == "Q"
    assert pool.market_q.measure == "Q"
    assert cfg.expiries_days == (42, 84, 126)
    assert pool.book_paths.contract_prices.shape == (1, 127, 9)
    np.testing.assert_allclose(
        pool.book_paths.value,
        -pool.book_paths.contract_prices.sum(axis=2), rtol=0, atol=1e-8,
    )
    env = ForwardEpisode(pool, cfg)
    obs, info = env.reset(path_index=0)
    assert obs.shape == (17,)
    rewards = []
    for day in range(126):
        obs, reward, done, _, info = env.step(np.zeros(3))
        rewards.append(reward)
        assert np.isfinite(obs).all()
        if day == 41:
            assert obs[5] == pytest.approx(0.)
        if day == 83:
            assert obs[6] == pytest.approx(0.)
    assert done
    assert info["terminal_pnl"] == pytest.approx(sum(rewards))
    assert np.count_nonzero(obs[5:8]) == 0
