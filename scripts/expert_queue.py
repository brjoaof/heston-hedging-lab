#!/usr/bin/env python3
"""Prepare Heston-Q regime splits once; run independent resumable SAC expert queues.

Usage (repository root):
    python scripts/expert_queue.py prepare
    python scripts/expert_queue.py run --mode trade      # dry run, no training
    python scripts/expert_queue.py run --mode band       # dry run, no training
    python scripts/expert_queue.py run --mode trade --execute
    python scripts/expert_queue.py run --mode band --execute
    python scripts/expert_queue.py status

Only stdlib is imported by the orchestrator. Data preparation and SAC training
are launched as separate Python processes, with frozen per-run configuration.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "configs" / "expert_schedule.json"
MODES = ("trade", "band")
SPLITS = ("train", "validation", "test")


def _full_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _xi(value: str) -> tuple[str, str]:
    try:
        dec = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid xi: {value}") from exc
    if not dec.is_finite() or dec < 0 or dec != dec.quantize(Decimal("0.01")):
        raise ValueError(f"xi must be nonnegative with at most two decimal places: {value}")
    normalized = f"{dec:.2f}"
    return normalized, "xi_" + normalized.replace(".", "p")


def load_schedule(path: Path) -> dict[str, Any]:
    schedule = json.loads(path.read_text(encoding="utf-8"))
    if schedule.get("schema_version") != 1:
        raise ValueError("Unsupported specialist schedule schema")
    schedule["regimes"] = [_xi(x)[0] for x in schedule["regimes"]]
    reserved = [_xi(x)[0] for x in schedule["reserved_operator_regimes"]]
    schedule["reserved_operator_regimes"] = reserved
    if (not schedule["regimes"] or len(set(schedule["regimes"])) != len(schedule["regimes"])
            or len(set(reserved)) != len(reserved) or set(reserved) & set(schedule["regimes"])):
        raise ValueError("Regime lists must be disjoint, nonempty for training, and unique")
    seeds = schedule["seeds"]
    if not seeds or any(type(x) is not int or x < 0 for x in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonnegative, unique integers")
    sizes = schedule["split_sizes"]
    split_seeds = schedule["split_seeds"]
    if (set(sizes) != set(SPLITS) or set(split_seeds) != set(SPLITS) or
            any(type(sizes[x]) is not int or sizes[x] < 1 for x in SPLITS) or
            any(type(split_seeds[x]) is not int or split_seeds[x] < 0 for x in SPLITS) or
            len(set(split_seeds.values())) != 3):
        raise ValueError("Invalid or overlapping data splits")
    t = schedule["train"]
    for item in ("timesteps", "n_envs", "train_chunk_paths", "eval_every", "resume_every",
                 "progress_every", "learning_starts", "buffer_size", "batch_size", "train_seed_base"):
        if type(t[item]) is not int or t[item] < (0 if item == "learning_starts" else 1):
            raise ValueError(f"Invalid {item}")
    if t["learning_starts"] >= t["timesteps"] or t["vec_env"] not in ("subproc", "dummy"):
        raise ValueError("Invalid training steps or vector environment")
    for item in ("gamma", "tau", "learning_rate", "penalty", "max_trade_increment", "max_band_width"):
        if not isinstance(t[item], (int, float)) or not 0 < t[item]:
            raise ValueError(f"Invalid {item}")
    for x in ("base_config", "data_root"):
        if not isinstance(schedule[x], str) or not schedule[x]:
            raise ValueError(f"Missing {x}")
    if "result_root" in schedule:
        path = _full_path(schedule["result_root"])
        expected_parent = (ROOT / "results").resolve()
        if not path.is_relative_to(expected_parent) or path == expected_parent:
            raise ValueError("result_root must be inside results/ and not equal to it")
    return schedule


def dataset_root(schedule: dict) -> Path:
    return _full_path(schedule["data_root"])


def config_paths(schedule: dict, xi: str) -> tuple[Path, Path]:
    label = _xi(xi)[1]
    root = dataset_root(schedule) / label
    return root / "config.json", root / "splits"


def train_seed(schedule: dict, xi: str, seed: int) -> int:
    """Identical training-stream seeds for trade/band paired experiments."""
    xi_integer = int(Decimal(xi) * 100)
    return schedule["train"]["train_seed_base"] + seed * 100003 + xi_integer * 1009


def result_dir(schedule: dict, mode: str, xi: str, seed: int) -> Path:
    t = schedule["train"]["timesteps"]
    budget = f"{t // 1_000_000}M" if t % 1_000_000 == 0 else str(t)
    root = _full_path(schedule["result_root"]) if "result_root" in schedule else ROOT / "results"
    return root / f"sac_{mode}" / f"{_xi(xi)[1]}_seed{seed}_{budget}"


def _run_matches(run: dict, schedule: dict, mode: str, xi: str, seed: int,
                 config: Path, splits: Path) -> bool:
    t = schedule["train"]
    return (
        run.get("action_mode") == mode
        and run.get("measure") == "Q"
        and run.get("selected_by") == "validation_q95_loss"
        and abs(float(run.get("regime_xi", -1)) - float(xi)) < 1e-12
        and run.get("seed") == seed
        and run.get("train_seed") == train_seed(schedule, xi, seed)
        and run.get("requested_timesteps") == t["timesteps"]
        and run.get("n_envs") == t["n_envs"]
        and run.get("vec_env") == t["vec_env"]
        and run.get("paths_per_chunk") == t["train_chunk_paths"]
        and run.get("eval_every") == t["eval_every"]
        and run.get("resume_every") == t["resume_every"]
        and run.get("progress_every") == t["progress_every"]
        and run.get("learning_starts") == t["learning_starts"]
        and run.get("buffer_size") == t["buffer_size"]
        and run.get("batch_size") == t["batch_size"]
        and abs(float(run.get("gamma", -1)) - t["gamma"]) < 1e-12
        and abs(float(run.get("tau", -1)) - t["tau"]) < 1e-12
        and abs(float(run.get("learning_rate", -1)) - t["learning_rate"]) < 1e-12
        and abs(float(run.get("penalty", -1)) - t["penalty"]) < 1e-12
        and (abs(float(run.get("max_band_width", -1)) - t["max_band_width"]) < 1e-12
             if mode == "band" else
             abs(float(run.get("max_trade_increment", -1)) - t["max_trade_increment"]) < 1e-12)
        and (run.get("band_execution") == "delta_center" if mode == "band" else True)
        and run.get("config_file") == str(config.resolve())
        and run.get("splits_dir") == str(splits.resolve())
    )


def job_state(schedule: dict, mode: str, xi: str, seed: int) -> str:
    out = result_dir(schedule, mode, xi, seed)
    run_path = out / "run.json"
    if not out.exists() or not any(out.iterdir()):
        return "pending"
    if not run_path.is_file():
        return "invalid: result directory contains no run.json"
    config, splits = config_paths(schedule, xi)
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return "invalid: cannot read run.json"
    if not _run_matches(run, schedule, mode, xi, seed, config, splits):
        return "invalid: completed or partial run does not match manifest"
    if (out / "summary.json").is_file():
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        if (summary.get("selection_metric") != "validation_q95_loss" or
                int(summary.get("final_timesteps", -1)) != int(run["timesteps"])):
            return "invalid: summary does not match frozen run"
        return "completed"
    if (out / "recovery" / "latest.json").is_file():
        return "resume"
    return "invalid: partial run without recovery/latest.json"


def _check_prepared(schedule: dict, xi: str) -> bool:
    """Inspect immutable metadata; the trainer checks the actual validation SHA."""
    conf_path, splits = config_paths(schedule, xi)
    if not conf_path.is_file() or not (splits / "manifest.json").is_file():
        return False
    base = json.loads(_full_path(schedule["base_config"]).read_text(encoding="utf-8"))
    expected = json.loads(json.dumps(base))
    expected["forward_experiment"]["heston"]["xi"] = float(xi)
    if json.loads(conf_path.read_text(encoding="utf-8")) != expected:
        raise ValueError(f"Financial config drift in {conf_path}")
    manifest = json.loads((splits / "manifest.json").read_text(encoding="utf-8"))
    for name in SPLITS:
        metadata = manifest["splits"][name]
        if (metadata["seed"] != schedule["split_seeds"][name]
                or metadata["n_paths"] != schedule["split_sizes"][name]
                or not (splits / f"{name}.npz").is_file()
                or not (splits / f"{name}.json").is_file()):
            raise ValueError(f"Invalid frozen dataset for {xi}, {name}")
        saved = json.loads((splits / f"{name}.json").read_text(encoding="utf-8"))
        if saved != metadata:
            raise ValueError(f"Split/manifest mismatch: {xi}, {name}")
    return True


def _prepare(schedule: dict, dry_run: bool) -> None:
    for xi in schedule["regimes"]:
        label = _xi(xi)[1]
        root = dataset_root(schedule) / label
        if _check_prepared(schedule, xi):
            print(f"[prepare] {label}: already prepared; keeping immutable datasets", flush=True)
            continue
        if root.exists():
            raise RuntimeError(f"Incomplete dataset: {root}. Inspect before manually recovering; nothing removed")
        t = schedule["split_sizes"]
        cmd = [sys.executable, "-u", str(ROOT / "regime_experiments.py"), "prepare",
               "--base-config", str(_full_path(schedule["base_config"])),
               "--out-root", str(dataset_root(schedule)), "--xi", xi,
               "--train-paths", str(t["train"]),
               "--validation-paths", str(t["validation"]),
               "--test-paths", str(t["test"])]
        print("[prepare]", " ".join(cmd), flush=True)
        if not dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)
            if not _check_prepared(schedule, xi):
                raise RuntimeError(f"Incomplete preparation after successful command: {label}")


def _train_command(schedule: dict, mode: str, xi: str, seed: int, *, resume: bool) -> list[str]:
    t = schedule["train"]
    config, splits = config_paths(schedule, xi)
    cmd = [sys.executable, "-u", str(ROOT / "train_sac_scale.py"), "train",
           "--action-mode", mode, "--config", str(config), "--splits-dir", str(splits),
           "--out-dir", str(result_dir(schedule, mode, xi, seed))]
    keys = ("timesteps", "n_envs", "vec_env", "train_chunk_paths", "eval_every",
            "resume_every", "progress_every", "learning_starts", "buffer_size",
            "batch_size", "gamma", "tau", "learning_rate", "penalty",
            "max_trade_increment", "max_band_width")
    for key in keys:
        cmd += ["--" + key.replace("_", "-"), str(t[key])]
    cmd += ["--seed", str(seed), "--train-seed", str(train_seed(schedule, xi, seed))]
    if resume:
        cmd.append("--resume")
    return cmd


@contextmanager
def _queue_lock(mode: str) -> Iterator[None]:
    location = ROOT / "results" / "queue_logs" / f"{mode}.lock"
    location.parent.mkdir(parents=True, exist_ok=True)
    with location.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another {mode} queue is already running") from exc
        yield


def _execute_job(schedule: dict, mode: str, xi: str, seed: int, resume: bool) -> None:
    cmd = _train_command(schedule, mode, xi, seed, resume=resume)
    label = f"{_xi(xi)[1]}_seed{seed}"
    log_root = _full_path(schedule["result_root"]) if "result_root" in schedule else ROOT / "results"
    log_dir = log_root / "queue_logs" / mode
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{label}.log"
    print(f"[queue] {mode} {label}: running; log={log_file}", flush=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"\n# command: {' '.join(cmd)}\n")
        f.flush()
        environment = dict(os.environ)
        # Avoid BLAS oversubscription when two independent 30-worker queues run.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            environment[variable] = "1"
        process = subprocess.Popen(cmd, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"{mode} {label} failed (exit={code}); queue stopped. Log: {log_file}")
    state = job_state(schedule, mode, xi, seed)
    if state != "completed":
        raise RuntimeError(f"{mode} {label} ended without complete summary ({state}); queue stopped")
    print(f"[queue] {mode} {label}: completed", flush=True)


def _run_queue(schedule: dict, mode: str, *, execute: bool) -> None:
    jobs = [(xi, seed) for xi in schedule["regimes"] for seed in schedule["seeds"]]
    print(f"[queue] mode={mode}; {len(jobs)} jobs; "
          f"budget={schedule['train']['timesteps']:,} transitions/job; "
          f"workers/job={schedule['train']['n_envs']}; execute={execute}", flush=True)
    for xi, seed in jobs:
        state = job_state(schedule, mode, xi, seed)
        print(f"[queue] {_xi(xi)[1]} seed={seed}: {state}", flush=True)
        if state.startswith("invalid"):
            raise RuntimeError(f"Aborting queue: {state} in {result_dir(schedule, mode, xi, seed)}")
    if not execute:
        print("[queue] Preview only. Use --execute to train; no models were started.")
        return
    # Preflight ALL datasets before a single expensive training job starts.
    for xi in schedule["regimes"]:
        if not _check_prepared(schedule, xi):
            raise RuntimeError(f"Prepare shared dataset first: {_xi(xi)[1]}")
    with _queue_lock(mode):
        for xi, seed in jobs:
            state = job_state(schedule, mode, xi, seed)
            if state == "completed":
                print(f"[queue] {_xi(xi)[1]} seed={seed}: already completed; skip", flush=True)
                continue
            _execute_job(schedule, mode, xi, seed, resume=(state == "resume"))
    print(f"[queue] {mode}: all scheduled jobs complete", flush=True)


def _status(schedule: dict) -> None:
    for xi in schedule["regimes"]:
        ready = _check_prepared(schedule, xi)
        print(f"{_xi(xi)[1]}: dataset {'ready' if ready else 'missing'}")
        for mode in MODES:
            states = [job_state(schedule, mode, xi, seed) for seed in schedule["seeds"]]
            print(f"  {mode}: " + ", ".join(f"s{seed}={state}" for seed, state in zip(schedule["seeds"], states)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                   help="JSON schedule: expert regimes, seeds, compute budget")
    sub = p.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare", help="Prepare shared frozen validation/test data, once")
    prepare.add_argument("--dry-run", action="store_true")
    run = sub.add_parser("run", help="Run one independent SAC queue")
    run.add_argument("--mode", choices=MODES, required=True)
    run.add_argument("--execute", action="store_true", help="Actually launch expensive jobs")
    sub.add_parser("status", help="Show dataset preparation and job statuses")
    args = p.parse_args()
    schedule = load_schedule(_full_path(str(args.manifest)))
    if args.action == "prepare":
        _prepare(schedule, dry_run=args.dry_run)
    elif args.action == "run":
        _run_queue(schedule, args.mode, execute=args.execute)
    else:
        _status(schedule)


if __name__ == "__main__":
    main()
