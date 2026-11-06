"""Constant band and xi-regime scaffolding: no RL dependencies required."""
import json
from pathlib import Path

import numpy as np
import pytest

from forward_experiments import load_split, save_split
from rl_env import prepare_forward_episodes
from regime_experiments import (
    fixed_band_control, paired_comparison, prepare_regimes, regime_config,
    regime_label, run_control,
)
from train_sac_trade import experiment_config, evaluate_actions
from band_policy import BAND_EXECUTION, ForwardBandEpisode


@pytest.fixture(scope="module")
def tiny():
    original = Path(__file__).resolve().parents[1] / "config.json"
    pool, cfg = prepare_forward_episodes(original, n_paths=2, seed=10234)
    return original, pool, cfg


def test_label_and_config_modify_only_xi(tiny):
    base_path, _, _ = tiny
    original = json.loads(base_path.read_text())
    assert [regime_label(x) for x in (0, .2, .4)] == ["xi_0p00", "xi_0p20", "xi_0p40"]
    for bad in (-.1, float("nan"), float("inf"), .201):
        with pytest.raises(ValueError):
            regime_label(bad)
    changed = regime_config(original, 0)
    assert changed["forward_experiment"]["heston"]["xi"] == 0
    expected = json.loads(base_path.read_text())
    expected["forward_experiment"]["heston"]["xi"] = 0
    assert changed == expected
    assert original["forward_experiment"]["heston"]["xi"] == .2


def test_fixed_band_is_identical_to_action_evaluator(tiny):
    _, pool, base = tiny
    summary, traces = fixed_band_control(pool, base, width=.75)
    cfg = experiment_config(base, penalty=2, max_trade_increment=base.max_position,
                            include_bucket_deltas=True)
    other, arrays = evaluate_actions(
        pool, cfg, lambda _obs: np.full(3, .75),
        episode_factory=lambda p, c: ForwardBandEpisode(p, c, max_band_width=1.5),
        action_bounds=(0, 1.5),
    )
    assert summary == other
    for k in ("terminal_pnl", "transaction_cost", "trades", "turnover",
              "actions", "executed_change", "episode_reward"):
        np.testing.assert_array_equal(traces[k], arrays[k])
    assert summary["mean_transaction_cost"] > 0
    np.testing.assert_allclose(traces["actions"], .75)
    with pytest.raises(ValueError):
        fixed_band_control(pool, base, width=1.51)


def test_paired_matches_and_detects_wrong_shape():
    a = {"terminal_pnl": np.array([1., 2.]), "transaction_cost": np.array([2., 3.]),
         "trades": np.array([4, 5]), "turnover": np.array([1., 1.])}
    b = {"terminal_pnl": np.array([2., 1.]), "transaction_cost": np.array([1., 4.]),
         "trades": np.array([3, 6]), "turnover": np.array([1., 1.])}
    result = paired_comparison(a, b)
    assert result["mean_pnl_advantage_learned"] == 0
    assert result["learned_better_paths"] == 1
    with pytest.raises(ValueError, match="Paired"):
        paired_comparison(a, {**b, "trades": np.array([3])})


def test_control_outputs_and_provenance_checks(tiny, tmp_path):
    source, pool, _ = tiny
    cfg = tmp_path / "config.json"
    cfg.write_text(source.read_text())
    folder = tmp_path / "splits"
    meta = save_split(folder, "validation", pool, cfg, seed=2202)
    out = tmp_path / "control"
    result = run_control(config=cfg, splits_dir=folder, out_dir=out)
    assert result["validation_data_sha256"] == meta["data_sha256"]
    assert result["band_execution"] == BAND_EXECUTION
    assert (out / "validation_constant_band.npz").exists()
    with pytest.raises(FileExistsError):
        run_control(config=cfg, splits_dir=folder, out_dir=out)
    # A SAC-Band result on the same dataset is pairable, with identity showing zero gap.
    model_dir = tmp_path / "band_model"
    model_dir.mkdir()
    (model_dir / "run.json").write_text(json.dumps({
        "action_mode": "band", "financial_config_sha256": meta["config_sha256"],
        "validation_data_sha256": meta["data_sha256"],
        "penalty": 2.0, "max_band_width": 1.5,
        "band_execution": BAND_EXECUTION,
    }))
    (model_dir / "validation_best_evaluation.json").write_text(json.dumps({
        "action_mode": "band", "split": "validation", "model": "best",
        "financial_config_sha256": meta["config_sha256"], "metrics": result["metrics"],
        "band_execution": BAND_EXECUTION,
    }))
    with np.load(out / "validation_constant_band.npz") as f:
        np.savez_compressed(model_dir / "validation_best_evaluation.npz", **{
            k: f[k] for k in ("terminal_pnl", "transaction_cost", "trades", "turnover")})
    compared = run_control(config=cfg, splits_dir=folder, out_dir=tmp_path / "compare",
                           compare_evaluation=model_dir / "validation_best_evaluation.npz")
    assert compared["paired_comparison"]["mean_pnl_advantage_learned"] == 0
    run = json.loads((model_dir / "run.json").read_text())
    run["validation_data_sha256"] = "incorrect"
    (model_dir / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="match"):
        run_control(config=cfg, splits_dir=folder, out_dir=tmp_path / "bad",
                    compare_evaluation=model_dir / "validation_best_evaluation.npz")
    run["validation_data_sha256"] = meta["data_sha256"]
    del run["band_execution"]  # Legacy boundary-trained checkpoint.
    (model_dir / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="retrain SAC-Band"):
        run_control(config=cfg, splits_dir=folder, out_dir=tmp_path / "old_semantics",
                    compare_evaluation=model_dir / "validation_best_evaluation.npz")


def test_prepare_three_regimes_small_and_no_overwrite(tiny, tmp_path):
    source, _, _ = tiny
    root = tmp_path / "regimes"
    manifest = prepare_regimes(source, root, xis=(0.0, .2, .4),
                               sizes=(2, 3, 1), seeds=(1101, 2202, 3303))
    assert len({m["financial_config_sha256"] for m in manifest}) == 3
    assert all(m["sizes"]["validation"] == 3 for m in manifest)
    for xi in (0.0, .2, .4):
        label = regime_label(xi)
        folder = root / label
        pool, cfg, meta = load_split(folder / "splits", "validation", folder / "config.json")
        assert pool.paths.measure == "Q"
        assert meta["heston_q"]["xi"] == xi
        assert len(pool.paths.spot) == 3
        if xi == 0:
            np.testing.assert_allclose(pool.paths.variance[0], pool.paths.variance[1])
    with pytest.raises(FileExistsError):
        prepare_regimes(source, root, xis=(0.,), sizes=(2, 3, 1),
                        seeds=(1101, 2202, 3303))
