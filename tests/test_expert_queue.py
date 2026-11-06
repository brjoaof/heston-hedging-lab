"""Pure-stdlib operational tests; no simulator or SB3 training required."""
import importlib.util
import json
from pathlib import Path

import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "expert_queue.py"
spec = importlib.util.spec_from_file_location("expert_queue", MODULE)
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def make_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "ROOT", tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "forward_experiment": {"measure": "Q", "heston": {"xi": .2}}
    }), encoding="utf-8")
    schedule = {
        "schema_version": 1, "base_config": "config.json", "data_root": "results/regimes",
        "regimes": ["0.00", "0.20"],
        "reserved_operator_regimes": ["0.10"],
        "seeds": [1, 2],
        "split_sizes": {"train": 64, "validation": 5000, "test": 20000},
        "split_seeds": {"train": 1101, "validation": 2202, "test": 3303},
        "train": {"timesteps": 100000, "n_envs": 30, "vec_env": "subproc",
                  "train_chunk_paths": 16, "eval_every": 2500, "resume_every": 500,
                  "progress_every": 500, "learning_starts": 500, "buffer_size": 500000,
                  "batch_size": 256, "gamma": 0.995, "tau": 0.002,
                  "learning_rate": 0.0001, "penalty": 2.0,
                  "max_trade_increment": 1.0, "max_band_width": 1.5,
                  "train_seed_base": 551177}
    }
    source = tmp_path / "schedule.json"
    source.write_text(json.dumps(schedule), encoding="utf-8")
    return queue.load_schedule(source)


def test_regime_schedule_disjoint_and_no_accidental_training(tmp_path, monkeypatch, capsys):
    schedule = make_schedule(tmp_path, monkeypatch)
    queue._run_queue(schedule, "trade", execute=False)
    out = capsys.readouterr().out
    assert "4 jobs" in out and "Preview only" in out
    assert not (tmp_path / "results").exists()
    assert queue._xi("0.2") == ("0.20", "xi_0p20")
    assert queue.train_seed(schedule, "0.20", 1) == queue.train_seed(schedule, "0.20", 1)
    assert queue.train_seed(schedule, "0.20", 1) != queue.train_seed(schedule, "0.00", 1)
    assert queue.train_seed(schedule, "0.20", 1) != queue.train_seed(schedule, "0.20", 2)


def test_duplicate_and_overlapping_regimes_rejected(tmp_path, monkeypatch):
    schedule = make_schedule(tmp_path, monkeypatch)
    schedule["reserved_operator_regimes"] = ["0.20"]
    file = tmp_path / "schedule.json"
    file.write_text(json.dumps(schedule), encoding="utf-8")
    with pytest.raises(ValueError, match="Regime lists"):
        queue.load_schedule(file)


def fake_run(schedule, mode, xi, seed):
    t = schedule["train"]
    config, splits = queue.config_paths(schedule, xi)
    return {
        "action_mode": mode, "measure": "Q", "selected_by": "validation_q95_loss",
        "regime_xi": float(xi), "seed": seed,
        "train_seed": queue.train_seed(schedule, xi, seed),
        "requested_timesteps": t["timesteps"], "timesteps": 100020,
        "n_envs": t["n_envs"], "vec_env": t["vec_env"],
        "paths_per_chunk": t["train_chunk_paths"],
        "eval_every": t["eval_every"], "resume_every": t["resume_every"],
        "progress_every": t["progress_every"], "learning_starts": t["learning_starts"],
        "buffer_size": t["buffer_size"], "batch_size": t["batch_size"],
        "gamma": t["gamma"], "tau": t["tau"],
        "learning_rate": t["learning_rate"], "penalty": t["penalty"],
        "max_trade_increment": 3.0 if mode == "band" else t["max_trade_increment"],
        "max_band_width": t["max_band_width"],
        "band_execution": "delta_center",
        "config_file": str(config.resolve()), "splits_dir": str(splits.resolve()),
    }


def test_queue_resumes_skips_completed_and_rejects_mismatch(tmp_path, monkeypatch):
    schedule = make_schedule(tmp_path, monkeypatch)
    mode, xi, seed = "band", "0.20", 1
    out = queue.result_dir(schedule, mode, xi, seed)
    assert queue.job_state(schedule, mode, xi, seed) == "pending"
    out.mkdir(parents=True)
    (out / "orphan_file").write_text("broken run")
    assert queue.job_state(schedule, mode, xi, seed).startswith("invalid")
    run = fake_run(schedule, mode, xi, seed)
    (out / "run.json").write_text(json.dumps(run), encoding="utf-8")
    assert queue.job_state(schedule, mode, xi, seed).startswith("invalid")
    (out / "recovery").mkdir()
    (out / "recovery" / "latest.json").write_text('{"name":"step_..."}', encoding="utf-8")
    assert queue.job_state(schedule, mode, xi, seed) == "resume"
    (out / "summary.json").write_text(json.dumps({
        "selection_metric": "validation_q95_loss", "final_timesteps": 100020
    }), encoding="utf-8")
    assert queue.job_state(schedule, mode, xi, seed) == "completed"
    run["max_band_width"] = 0.5
    (out / "run.json").write_text(json.dumps(run), encoding="utf-8")
    assert queue.job_state(schedule, mode, xi, seed).startswith("invalid")


def test_prepare_is_idempotent_and_requires_complete_datasets(tmp_path, monkeypatch):
    schedule = make_schedule(tmp_path, monkeypatch)
    schedule["regimes"] = ["0.20"]
    xi = "0.20"
    root = queue.dataset_root(schedule) / "xi_0p20"
    assert queue._check_prepared(schedule, xi) is False
    base = json.loads((tmp_path / "config.json").read_text())
    base["forward_experiment"]["heston"]["xi"] = 0.2
    splits = root / "splits"
    splits.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(base))
    manifest = {"splits": {}}
    for split in queue.SPLITS:
        record = {"seed": schedule["split_seeds"][split],
                  "n_paths": schedule["split_sizes"][split]}
        manifest["splits"][split] = record
        (splits / f"{split}.npz").write_bytes(b"dummy")
        (splits / f"{split}.json").write_text(json.dumps(record))
    (splits / "manifest.json").write_text(json.dumps(manifest))
    assert queue._check_prepared(schedule, xi) is True
    # A partial prepared dataset is not silently deleted.
    (splits / "validation.npz").unlink()
    with pytest.raises(ValueError, match="Invalid frozen dataset"):
        queue._prepare(schedule, dry_run=False)


def test_train_command_is_complete_and_supports_resume(tmp_path, monkeypatch):
    schedule = make_schedule(tmp_path, monkeypatch)
    trade = queue._train_command(schedule, "trade", "0.20", 1, resume=False)
    band = queue._train_command(schedule, "band", "0.20", 1, resume=True)
    assert "--n-envs" in trade and trade[trade.index("--n-envs") + 1] == "30"
    assert "--vec-env" in trade and trade[trade.index("--vec-env") + 1] == "subproc"
    assert "--resume" not in trade and "--resume" in band
    assert trade[trade.index("--train-seed") + 1] == band[band.index("--train-seed") + 1]
    assert str(tmp_path / "results" / "sac_trade") in " ".join(trade)
    assert str(tmp_path / "results" / "sac_band") in " ".join(band)
