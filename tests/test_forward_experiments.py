"""Independent regression tests for Q split provenance and causal reference policies."""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from hedging import BookPaths, simulate_forward_hedge
from heston import HestonPaths, MarketDynamics
from rl_env import ForwardEpisodePool, ForwardEnvConfig
from forward_experiments import (
    baseline_actions, calibrate_reward_candidates, evaluate_baseline, iter_train_chunks,
    load_split, prepare_splits, report_baselines, reward_increments, save_split,
)


def _dummy(tmp_path):
    """Marks made by hand; separate from Heston pricing to isolate policy logic."""
    spot = np.array([
        [100., 104., 106., 101.],
        [100., 102., 98., 107.],
        [100., 99., 101., 105.],
    ])
    v = np.full_like(spot, .04)
    paths = HestonPaths(np.arange(4)/250, spot, v, v.copy(), "Q")
    cp = np.ones((3,4,9))
    cp[:, 1:, 0:3] = 0
    cp[:, 2:, 3:6] = 0
    cp[:, 3:, 6:9] = 0
    value = -cp.sum(axis=2)
    delta = np.array([
        [[-1.5,-1.3,-1.5], [-.7,-.4,-.6], [-2.0,-1.2,-1.1], [0,0,0]],
        [[-1.5,-1.3,-1.5], [-1.1,-.8,-.9], [-1.8,-1.0,-1.4], [0,0,0]],
        [[-1.5,-1.3,-1.5], [-1.2,-.9,-.8], [-1.4,-1.4,-1.4], [0,0,0]],
    ])
    # One-day contract is dead at t=1; second at t=2.
    delta[:, 1:, 0] = 0
    delta[:, 2:, 1] = 0
    b = BookPaths(value, np.zeros_like(spot), np.zeros_like(spot),
                  np.zeros_like(spot), np.zeros_like(spot), cp, delta)
    pos = np.array([[1.5, 1.3, 1.5]]*3)
    cfg = ForwardEnvConfig(expiries_days=(1,2,3),
                           spread_half_pcts=(.01,.02,.03),
                           fixed_cost=.5, min_trade=.01,
                           trading_days_per_year=250)
    market = MarketDynamics("Q", 100., 0.)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"forward_experiment": {
        "measure":"Q", "spot0":100, "risk_free_rate":0,
        "heston": {"kappa": 1., "theta": .2, "xi": .2, "rho": .7, "v0": .04},
        "dividend_yield":0, "trading_days_per_year":250,
        "expiries_days":[1,2,3], "spread_half_pcts":[.01,.02,.03],
        "fixed_cost":.5, "min_trade":.01, "max_trade_increment":1,
        "max_position":3, "quadratic_reward_penalty":0,
        "include_variance":True,
    }}), encoding="utf-8")
    return ForwardEpisodePool(paths, b, pos, market), cfg, config_file


def test_hold_initial_equals_zero_action_batch(tmp_path):
    pool, cfg, _ = _dummy(tmp_path)
    expected = simulate_forward_hedge(
        pool.paths, pool.book_paths, pool.market_q,
        initial_positions=pool.initial_positions,
        actions=np.zeros((3,3,3)), expiries_days=cfg.expiries_days,
        spread_half_pcts=cfg.spread_half_pcts,
        fixed_cost=cfg.fixed_cost, trading_days_per_year=250,
    )
    got = evaluate_baseline(pool, cfg, "hold_initial")
    np.testing.assert_array_equal(got.wealth, expected.wealth)
    np.testing.assert_array_equal(got.transaction_cost, expected.transaction_cost)
    np.testing.assert_allclose(reward_increments(got).sum(axis=1), got.terminal_pnl)


def test_no_hedge_means_no_opening_or_transaction_cost(tmp_path):
    pool, cfg, _ = _dummy(tmp_path)
    r = evaluate_baseline(pool, cfg, "no_hedge")
    assert np.all(r.initial_cost == 0)
    assert np.all(r.transaction_cost == 0)
    assert np.all(r.trades == 0)
    np.testing.assert_allclose(r.position, 0)


def test_delta_daily_is_active_and_bounded(tmp_path):
    pool, cfg, _ = _dummy(tmp_path)
    init, actions = baseline_actions(pool, cfg, "delta_daily")
    assert actions.shape == (3,3,3)
    assert np.max(np.abs(actions)) <= cfg.max_trade_increment
    assert not np.allclose(actions, 0)
    # First bucket has expired by the second decision.
    assert np.all(actions[:, 1:, 0] == 0)
    got = evaluate_baseline(pool, cfg, "delta_daily")
    assert np.isfinite(got.wealth).all()
    assert np.all(got.position[:, -1] == 0)


def test_delta_baselines_see_only_current_day(tmp_path):
    pool, cfg, _ = _dummy(tmp_path)
    _, original = baseline_actions(pool, cfg, "delta_daily")
    d = pool.book_paths.bucket_delta.copy()
    d[:, 2:, :] = -100
    b = pool.book_paths
    alt_book = BookPaths(b.value, b.delta, b.gamma, b.variance_sensitivity,
                         b.settlements, b.contract_prices, d)
    alt = ForwardEpisodePool(pool.paths, alt_book, pool.initial_positions, pool.market_q)
    _, modified = baseline_actions(alt, cfg, "delta_daily")
    np.testing.assert_array_equal(original[:, :2], modified[:, :2])


def test_splits_save_load_verifies_hash_and_configuration(tmp_path):
    pool, _, config_file = _dummy(tmp_path)
    output = tmp_path / "splits"
    metadata = save_split(output, "validation", pool, config_file, seed=2202)
    rebuilt, cfg, loaded = load_split(output, "validation", config_file)
    np.testing.assert_array_equal(rebuilt.paths.spot, pool.paths.spot)
    np.testing.assert_array_equal(rebuilt.book_paths.bucket_delta, pool.book_paths.bucket_delta)
    assert cfg.trading_days_per_year == 250
    assert metadata == loaded
    with pytest.raises(FileExistsError):
        save_split(output, "validation", pool, config_file, seed=2202)
    modified_config = json.loads(config_file.read_text())
    modified_config["forward_experiment"]["fixed_cost"] = 0.8
    config_file.write_text(json.dumps(modified_config))
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_split(output, "validation", config_file)


def test_checksum_detects_corruption(tmp_path):
    pool, _, config_file = _dummy(tmp_path)
    output = tmp_path / "splits"
    save_split(output, "test", pool, config_file, seed=3303)
    path = output / "test.npz"
    with path.open("ab") as file:
        file.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_split(output, "test", config_file)


def test_reject_invalid_split_and_reused_seed(tmp_path):
    pool, cfg, config_file = _dummy(tmp_path)
    with pytest.raises(ValueError, match="Unknown split"):
        save_split(tmp_path, "any", pool, config_file, seed=1)
    with pytest.raises(ValueError, match="seeds must be distinct"):
        prepare_splits(config_file, tmp_path / "new", sizes=(1,1,1), seeds=(9,9,3))
    with pytest.raises(ValueError, match="positive split"):
        prepare_splits(config_file, tmp_path / "new", sizes=(0,1,1))


def test_reward_scale_is_validation_reference_rule(tmp_path):
    pool, cfg, _ = _dummy(tmp_path)
    out = calibrate_reward_candidates(pool, cfg)
    scale = out["p90_absolute_step_pnl"]
    assert scale > 0
    assert [x["relative_penalty_at_p90"] for x in out["candidate_coefficients"]] == [0, .05, .1, .2]
    assert out["candidate_coefficients"][2]["coefficient"] == pytest.approx(.1 / scale)


def test_reporting_writes_q_loss_metrics_and_path_series(tmp_path):
    pool, cfg, _ = _dummy(tmp_path)
    results = report_baselines(pool, cfg, tmp_path / "results", split="validation",
                               config_sha256="mock")
    assert len(results) == 5
    assert (tmp_path / "results/validation_baselines.csv").exists()
    with np.load(tmp_path / "results/validation_baseline_per_path.npz") as data:
        assert data["delta_daily_terminal_pnl"].shape == (3,)
    assert json.loads((tmp_path / "results/validation_baselines.json").read_text())["measure"] == "Q"
    with pytest.raises(ValueError, match="Report only"):
        report_baselines(pool, cfg, tmp_path / "foo", split="train", config_sha256="mock")


def test_real_preparation_reuses_Q_marks_and_deltas(tmp_path):
    from rl_env import prepare_forward_episodes
    config = Path(__file__).resolve().parents[1] / "config.json"
    pool, cfg = prepare_forward_episodes(config, n_paths=1, seed=918)
    assert pool.book_paths.bucket_delta.shape == (1,127,3)
    np.testing.assert_allclose(pool.book_paths.bucket_delta.sum(axis=2), pool.book_paths.delta, atol=1e-9)
    for policy in ("hold_initial", "delta_daily", "delta_periodic_21", "delta_band_0p5", "no_hedge"):
        result = evaluate_baseline(pool, cfg, policy)
        assert np.isfinite(result.terminal_pnl).all()
        np.testing.assert_allclose(reward_increments(result).sum(axis=1), result.terminal_pnl)


def test_train_chunk_seed_protocol_is_reproducible_and_nonrepeating(tmp_path, monkeypatch):
    pool, cfg, config_path = _dummy(tmp_path)
    captured = []
    def fake_preparer(config, *, n_paths, seed):
        captured.append((n_paths, seed))
        return pool, cfg
    monkeypatch.setattr("forward_experiments.prepare_forward_episodes", fake_preparer)
    a = list(iter_train_chunks(config_path, base_seed=551, chunk_size=8, n_chunks=3))
    b = list(iter_train_chunks(config_path, base_seed=551, chunk_size=8, n_chunks=3))
    assert [x[0] for x in a] == [x[0] for x in b]
    assert len({x[0]["seed"] for x in a}) == 3
    assert all(n == 8 for n, _ in captured)
    with pytest.raises(ValueError, match="chunk_size"):
        list(iter_train_chunks(config_path, base_seed=1, chunk_size=0, n_chunks=2))
