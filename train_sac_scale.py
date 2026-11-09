"""Long-horizon Heston-Q SAC trainer: Q95 selection, vectorized Q streams and resume.

Usage (from repository root):
    python train_sac_scale.py train --action-mode trade --config config.json \
        --splits-dir results/scale_splits/xi_0p20 --out-dir results/sac_trade/scale_xi_0p20 \
        --timesteps 50000000 --n-envs 30 --vec-env subproc
    python train_sac_scale.py train --action-mode trade --config config.json \
        --splits-dir results/scale_splits/xi_0p20 --out-dir results/sac_trade/scale_xi_0p20 \
        --timesteps 50000000 --n-envs 30 --vec-env subproc --resume
    python train_sac_scale.py evaluate --run-dir results/sac_trade/scale_xi_0p20

The Q95 selector only sees validation paths. Training uses independently seeded,
on-demand Heston-Q streams. Recovery snapshots include SAC optimizer/model, replay
buffer, VecNormalize, RNG, and each stream's current marked Q path and episode
position. Recovery is statistically faithful, NOT guaranteed bit-for-bit: SB3
can reset internal rollout state and the process scheduling may differ.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import shutil
import time
from typing import Any

import numpy as np

from forward_experiments import _read_config, iter_train_chunks, load_split
from train_sac_trade import (
    StreamingQForwardGymEnv, StreamingQBandGymEnv,
    _episode_factory, _action_bounds, _sb3_policy, _save_checkpoint,
    _checkpoint_stem, _progress_line, _format_duration,
    _write_json, evaluate_actions, experiment_config,
)
from band_policy import BAND_EXECUTION

SELECTION_METRIC = "validation_q95_loss"
SCHEMA_VERSION = 1
SOURCE_FILES = (
    "train_sac_scale.py", "train_sac_trade.py", "rl_env.py", "band_policy.py",
    "hedging.py", "forward_experiments.py", "heston.py", "pricing.py", "pricing_cos.py",
    "volatility.py", "portfolio.py", "simulate.py",
)


def code_fingerprint(base: Path | None = None) -> str:
    """Protect resume from accidental changes to the financial/training code."""
    root = Path(__file__).resolve().parent if base is None else Path(base)
    sha = hashlib.sha256()
    for file in SOURCE_FILES:
        sha.update(file.encode("utf-8") + b"\x00")
        sha.update((root / file).read_bytes())
    return sha.hexdigest()


def q95_is_better(metrics: dict, previous: float) -> bool:
    """Strict minimize-Q95 rule from ENIAC; ties keep the earlier checkpoint."""
    q95 = float(metrics["q95_loss"])
    if not math.isfinite(q95):
        raise ValueError("Nonfinite validation Q95; checkpoint cannot be selected")
    return q95 < previous


def worker_stream_seed(base_seed: int, worker: int) -> int:
    if base_seed < 0 or worker < 0:
        raise ValueError("Nonnegative stream seeds and worker indices required")
    return int(np.random.SeedSequence([base_seed, worker, 0x5343414C])
               .generate_state(1, dtype=np.uint64)[0])


class _ResumableStream:
    """Mixin for persistent Q stream state; no validation/test arrays are involved."""

    def initialize_scale(self, config: str, seed: int, paths_per_chunk: int,
                         n_chunks: int):
        self._scale_config_path = config
        self._scale_seed = seed
        self._scale_chunk_paths = paths_per_chunk
        self._scale_n_chunks = n_chunks
        # Resume restarts each worker on its next, unused Q episode.  Keeping
        # a stale SAC observation while exposing an old in-flight episode is
        # unsafe under VecEnv's automatic reset-on-done protocol.

    def export_training_state(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "seed": self._scale_seed,
            "n_chunks": self._scale_n_chunks,
            "paths_per_chunk": self._scale_chunk_paths,
            "core": self.core,   # includes current Q path and cash/position/day
            "order": self._order.copy(), "cursor": int(self._cursor),
            "chunk_index": int(self.chunk_index),
            "generated_chunks": int(self.generated_chunks),
        }

    def import_training_state(self, state: dict) -> None:
        if (state["schema_version"] != SCHEMA_VERSION or
                state["seed"] != self._scale_seed or
                state["n_chunks"] != self._scale_n_chunks or
                state["paths_per_chunk"] != self._scale_chunk_paths or
                state["core"].config != self.core.config):
            raise ValueError("Restored Heston training stream is incompatible")
        self.core = state["core"]
        self._order = state["order"].copy()
        self._cursor = state["cursor"]
        self.chunk_index = state["chunk_index"]
        self.generated_chunks = state["generated_chunks"]
        # Do not regenerate previously consumed Q pools. The *next* chunk is
        # indexed deterministically, independent of worker execution order.
        self._chunks = iter_train_chunks(
            self._scale_config_path, base_seed=self._scale_seed,
            chunk_size=self._scale_chunk_paths, n_chunks=self._scale_n_chunks,
            start_chunk=self.chunk_index + 1,
        )
        # Do NOT resume an in-flight episode with a cached SB3 observation.
        # SAC.load(force_reset=True) makes _setup_learn call VecEnv.reset(),
        # which selects the next unused path from this saved stream.  The
        # current partially consumed path is discarded; its past transitions
        # remain valid in the restored replay buffer.  This is a *statistical*
        # resume, not a bitwise-identical continuation.
        self.core.path_index = None
        self.core.position = None
        self.core.cash = None
        self.core.day = 0


class ScaleTradeStream(_ResumableStream, StreamingQForwardGymEnv):
    def __init__(self, config_path: str, *, train_seed: int, paths_per_chunk: int,
                 n_chunks: int, penalty: float, max_trade_increment: float,
                 include_bucket_deltas: bool):
        super().__init__(config_path, train_seed=train_seed,
                         paths_per_chunk=paths_per_chunk, n_chunks=n_chunks,
                         penalty=penalty, max_trade_increment=max_trade_increment,
                         include_bucket_deltas=include_bucket_deltas)
        self.initialize_scale(config_path, train_seed, paths_per_chunk, n_chunks)


class ScaleBandStream(_ResumableStream, StreamingQBandGymEnv):
    def __init__(self, config_path: str, *, train_seed: int, paths_per_chunk: int,
                 n_chunks: int, penalty: float, max_trade_increment: float,
                 include_bucket_deltas: bool, max_band_width: float):
        super().__init__(config_path, train_seed=train_seed,
                         paths_per_chunk=paths_per_chunk, n_chunks=n_chunks,
                         penalty=penalty, max_trade_increment=max_trade_increment,
                         include_bucket_deltas=include_bucket_deltas,
                         max_band_width=max_band_width)
        self.initialize_scale(config_path, train_seed, paths_per_chunk, n_chunks)


def make_worker(index: int, *, action_mode: str, config_path: str, train_seed: int,
                paths_per_chunk: int, n_chunks: int, penalty: float,
                max_trade_increment: float, max_band_width: float):
    seed = worker_stream_seed(train_seed, index)
    cls = ScaleBandStream if action_mode == "band" else ScaleTradeStream
    extra = {"max_band_width": max_band_width} if action_mode == "band" else {}
    return cls(config_path, train_seed=seed,
               paths_per_chunk=paths_per_chunk, n_chunks=n_chunks,
               penalty=penalty, max_trade_increment=max_trade_increment,
               include_bucket_deltas=True, **extra)


def set_eniac_cpu_threads(torch_module) -> None:
    """Match the ENIAC training scripts' CPU threading convention."""
    torch_module.set_num_threads(1)


def load_sb3():
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    import torch
    # ENIAC CPU protocol: avoid PyTorch BLAS over-subscription while
    # 30 independent SubprocVecEnv workers simulate Heston paths.
    set_eniac_cpu_threads(torch)
    return SAC, DummyVecEnv, SubprocVecEnv, VecNormalize, BaseCallback, torch


def validation_selection_record(step: int, summary: dict) -> dict:
    return {"timesteps": int(step), **summary, "selection_metric": SELECTION_METRIC}


def scale_plan(args, *, cfg, config_sha: str, val_sha: str) -> dict:
    if args.timesteps <= 0 or args.eval_every <= 0 or args.resume_every <= 0:
        raise ValueError("Timesteps and evaluation/recovery intervals must be positive")
    if args.progress_every <= 0 or args.n_envs <= 0 or args.train_chunk_paths <= 0:
        raise ValueError("Progress, n_envs and chunk paths must be positive")
    if args.learning_starts < 0 or args.learning_starts >= args.timesteps:
        raise ValueError("learning_starts must be below timesteps")
    if args.batch_size <= 0 or args.buffer_size < args.batch_size:
        raise ValueError("buffer_size must be >= batch_size > 0")
    if args.seed < 0 or args.train_seed < 0:
        raise ValueError("Seeds must be nonnegative")
    if args.vec_env not in ("dummy", "subproc"):
        raise ValueError("vec_env must be dummy or subproc")
    if not (0 < args.gamma <= 1 and 0 < args.tau <= 1 and args.learning_rate > 0):
        raise ValueError("Invalid SAC hyperparameters")
    if not math.isfinite(args.penalty) or args.penalty < 0:
        raise ValueError("Invalid reward penalty")
    if args.action_mode == "band":
        if not 0 < args.max_band_width <= cfg.max_position:
            raise ValueError("Invalid maximum band width")
        increment = cfg.max_position
    else:
        increment = args.max_trade_increment
        if not 0 < increment <= cfg.max_position:
            raise ValueError("Invalid maximum trade increment")
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm": "SAC-Band" if args.action_mode == "band" else "SAC-Trade",
        "action_mode": args.action_mode,
        "measure": "Q", "selected_by": SELECTION_METRIC,
        "financial_config_sha256": config_sha, "validation_data_sha256": val_sha,
        "code_sha256": code_fingerprint(), "regime_xi": args.regime_xi,
        "config_file": str(Path(args.config).resolve()),
        "splits_dir": str(Path(args.splits_dir).resolve()),
        "timesteps": math.ceil(args.timesteps / args.n_envs) * args.n_envs,
        "requested_timesteps": args.timesteps, "n_envs": args.n_envs,
        "vec_env": args.vec_env, "seed": args.seed, "train_seed": args.train_seed,
        "paths_per_chunk": args.train_chunk_paths,
        "penalty": args.penalty, "max_trade_increment": increment,
        "include_bucket_deltas": True,
        **({"band_execution": BAND_EXECUTION, "max_band_width": args.max_band_width}
           if args.action_mode == "band" else {}),
        "eval_every": args.eval_every, "resume_every": args.resume_every,
        "progress_every": args.progress_every, "learning_starts": args.learning_starts,
        "batch_size": args.batch_size, "buffer_size": args.buffer_size,
        "gamma": args.gamma, "tau": args.tau, "learning_rate": args.learning_rate,
        "train_freq_steps": 1, "gradient_steps": 1, "torch_num_threads": 1,
        "network": [128, 128], "save_eval_checkpoints": True,
        "config": asdict(experiment_config(cfg, penalty=args.penalty,
                    max_trade_increment=increment, include_bucket_deltas=True)),
        "note": "Best minimizes validation Q95; test untouched. Resume not guaranteed bitwise identical.",
    }


def code_version_compatible(saved_sha: str, current_sha: str) -> bool:
    """Permit only exact, reviewed source-version migrations for legacy snapshots.

    Each manifest entry authorizes one specific *complete* source fingerprint
    to run the fresh-episode recovery fix.  The manifest is not a SOURCE_FILE
    (avoiding a self-referential hash), and future code changes fail closed.
    Scientific configuration and recovery-run signatures are checked separately.
    """
    if saved_sha == current_sha:
        return True
    manifest = Path(__file__).with_name("scale_resume_compat.json")
    if not manifest.is_file():
        return False
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (record.get("schema_version") == 2
            and record.get("reason") == "fresh_episode_resume_after_vecenv_reset"
            and record.get("fixed_code_sha256") == current_sha
            and saved_sha in record.get("compatible_previous_sha256", []))


# ENIAC/SB3 defaults were implicit in the first Heston-scale pilot.
# When reading those old snapshots, supply *only* these frozen, audited defaults;
# all other missing keys remain fatal and all explicitly changed keys are rejected.
LEGACY_IMPLICIT_ENIAC_DEFAULTS = {
    "train_freq_steps": 1,
    "gradient_steps": 1,
    "torch_num_threads": 1,
}


def assert_same_plan(saved: dict, requested: dict) -> None:
    """Freeze all training controls, comparing nested JSON canonically.

    Dataclass tuple fields become JSON arrays in persisted run.json. Compare
    their JSON-serializable representations instead of Python tuple/list types.
    Only the audited code-fingerprint transition can bypass exact hash match.
    """
    keys = set(requested) - {"package_versions", "created_at_utc"}
    for key in keys:
        if key == "code_sha256":
            matches = code_version_compatible(saved.get(key), requested[key])
        elif key not in saved and key in LEGACY_IMPLICIT_ENIAC_DEFAULTS:
            matches = requested[key] == LEGACY_IMPLICIT_ENIAC_DEFAULTS[key]
        else:
            matches = (json.dumps(saved.get(key), sort_keys=True, allow_nan=False)
                       == json.dumps(requested[key], sort_keys=True, allow_nan=False))
        if not matches:
            raise ValueError(f"Resume configuration mismatch for {key}; use the original flags")


class ScaleProgress:
    """Elapsed wall-time + rolling ETA, including preexisting work when resuming."""
    def __init__(self, total: int, interval: int, completed: int = 0,
                 previous_wall: float = 0.0):
        self.total, self.interval = total, interval
        self.started = self.previous = time.perf_counter()
        self.previous_steps = completed
        self.previous_wall = previous_wall
        self.rolling_rate = None
        self.next_report = (completed // interval + 1) * interval

    def snapshot(self, completed: int, force=False) -> dict | None:
        if not self.previous_steps <= completed <= self.total:
            raise ValueError("Invalid completed step count")
        if completed == self.previous_steps or (not force and completed < self.next_report):
            return None
        now = time.perf_counter()
        interval_seconds = now - self.previous
        if interval_seconds > 0:
            rate = (completed - self.previous_steps) / interval_seconds
            self.rolling_rate = (rate if self.rolling_rate is None else
                                 0.65 * self.rolling_rate + 0.35 * rate)
        elapsed = self.previous_wall + (now - self.started)
        eta = ((self.total - completed) / self.rolling_rate
               if self.rolling_rate and self.rolling_rate > 0 else None)
        record = {
            "timesteps": completed, "total_timesteps": self.total,
            "progress_percent": 100 * completed / self.total,
            "elapsed_seconds": elapsed,
            "average_steps_per_second": completed / elapsed if elapsed > 0 else None,
            "rolling_steps_per_second": self.rolling_rate,
            "eta_seconds": eta,
            "estimated_finish_utc": (
                datetime.now(timezone.utc) + timedelta(seconds=eta)
            ).isoformat(timespec="seconds") if eta is not None else None,
        }
        self.previous = now
        self.previous_steps = completed
        self.next_report = (completed // self.interval + 1) * self.interval
        return record


def _save_recovery(model, norm, output: Path, *, step: int, callback_state: dict,
                   run_sha: str, torch_module, keep: int = 2) -> Path:
    """Commit a recovery snapshot only after every required file is written."""
    recovery_root = output / "recovery"
    recovery_root.mkdir(exist_ok=True)
    stem = _checkpoint_stem(step)
    staging = recovery_root / (stem + ".incomplete")
    target = recovery_root / stem
    if target.exists():
        return target
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        model.save(str(staging / "model"))
        model.save_replay_buffer(str(staging / "replay_buffer.pkl"))
        norm.save(str(staging / "vecnormalize.pkl"))
        streams = norm.venv.env_method("export_training_state")
        with (staging / "streams.pkl").open("wb") as file:
            pickle.dump(streams, file, protocol=pickle.HIGHEST_PROTOCOL)
        rng_state = {
            "python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch_module.get_rng_state(),
        }
        with (staging / "rng.pkl").open("wb") as file:
            pickle.dump(rng_state, file, protocol=pickle.HIGHEST_PROTOCOL)
        _write_json(staging / "metadata.json", {
            "schema_version": SCHEMA_VERSION, "run_sha256": run_sha,
            "timesteps": step, "callback": callback_state,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "resume_bitwise_guaranteed": False,
        })
        os.replace(staging, target)
        # The latest pointer is updated atomically *after* the complete checkpoint.
        ptr = recovery_root / "latest.json"
        temp = recovery_root / "latest.json.incomplete"
        _write_json(temp, {"name": stem, "timesteps": step})
        os.replace(temp, ptr)
        all_snapshots = sorted(p for p in recovery_root.glob("step_[0-9]*") if p.is_dir())
        for old in all_snapshots[:-keep]:
            shutil.rmtree(old)
        return target
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _load_recovery(output: Path, run_sha: str):
    folder = output / "recovery"
    ptr = json.loads((folder / "latest.json").read_text(encoding="utf-8"))
    stem = ptr["name"]
    if not isinstance(stem, str) or not stem.startswith("step_") or len(stem) != 17:
        raise ValueError("Invalid recovery pointer")
    root = folder / stem
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata["run_sha256"] != run_sha or metadata["timesteps"] != ptr["timesteps"]:
        raise ValueError("Recovery files do not match the frozen run")
    for required in ("model.zip", "replay_buffer.pkl", "vecnormalize.pkl",
                     "streams.pkl", "rng.pkl"):
        if not (root / required).is_file():
            raise FileNotFoundError(f"Incomplete recovery snapshot: {required}")
    return root, metadata


def _run_sha(run: dict) -> str:
    return hashlib.sha256(json.dumps(run, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def train(args) -> None:
    SAC, DummyVecEnv, SubprocVecEnv, VecNormalize, BaseCallback, torch = load_sb3()
    pool_val, base, meta = load_split(args.splits_dir, "validation", args.config)
    config_data, config_sha = _read_config(args.config)
    args.regime_xi = config_data["heston"]["xi"]
    run = scale_plan(args, cfg=base, config_sha=config_sha, val_sha=meta["data_sha256"])
    if args.train_seed in {int(meta["seed"])}:
        raise ValueError("Training base seed collides with a validation seed")
    manifest = Path(args.splits_dir) / "manifest.json"
    if manifest.exists():
        split_seeds = {int(item["seed"]) for item in
                       json.loads(manifest.read_text(encoding="utf-8"))["splits"].values()}
        if args.train_seed in split_seeds:
            raise ValueError("Training base seed collides with prepared data split")
    output = Path(args.out_dir).resolve()
    if args.stop_after is not None and not (0 < args.stop_after < args.timesteps):
        raise ValueError("--stop-after must be between zero and the total budget")
    if args.resume:
        saved_run = json.loads((output / "run.json").read_text(encoding="utf-8"))
        assert_same_plan(saved_run, run)
        run = saved_run
        if (output / "summary.json").exists():
            raise ValueError("This run was already finalized; never silently extend it")
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Training output directory is nonempty")
        output.mkdir(parents=True, exist_ok=True)
        from importlib.metadata import version
        run["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        run["package_versions"] = {
            key: version(key) for key in
            ("stable-baselines3", "gymnasium", "torch", "numpy")
        }
        _write_json(output / "run.json", run)
    run_sha = _run_sha(run)
    target_steps = run["timesteps"]
    init_step = 0
    old_wall = 0.0
    resume_root, resume_meta = (None, None)
    if args.resume:
        resume_root, resume_meta = _load_recovery(output, run_sha)
        init_step = int(resume_meta["timesteps"])
        old_wall = float(resume_meta["callback"].get("wall_seconds", 0))
    # Independent Q-Heston streams: one per worker. Bound each worker's chunk
    # generator by an ample per-worker upper bound; no path cache is shared.
    n_chunks = math.ceil(target_steps / (args.n_envs * base.expiries_days[-1] *
                                         args.train_chunk_paths)) + 16
    stream_args = dict(
        action_mode=args.action_mode, config_path=str(Path(args.config).resolve()),
        train_seed=args.train_seed, paths_per_chunk=args.train_chunk_paths,
        n_chunks=n_chunks, penalty=args.penalty,
        max_trade_increment=run["max_trade_increment"],
        max_band_width=args.max_band_width,
    )
    factories = [partial(make_worker, i, **stream_args) for i in range(args.n_envs)]
    raw_env = (DummyVecEnv(factories) if args.vec_env == "dummy" else
               SubprocVecEnv(factories, start_method="forkserver"))
    norm = None
    try:
        if args.resume:
            norm = VecNormalize.load(str(resume_root / "vecnormalize.pkl"), raw_env)
            norm.training, norm.norm_reward = True, True
            with (resume_root / "streams.pkl").open("rb") as file:
                stream_states = pickle.load(file)
            if len(stream_states) != args.n_envs:
                raise ValueError("Saved stream count differs from n_envs")
            for i, state in enumerate(stream_states):
                raw_env.env_method("import_training_state", state, indices=[i])
            # SB3 must reset the vector environment at the start of learn().
            # With force_reset=False, a cached _last_obs from the checkpoint
            # skips VecEnv.reset(); a saved restoration hook can then return
            # a *terminated* observation on DummyVecEnv's later auto-reset.
            model = SAC.load(str(resume_root / "model.zip"), env=norm,
                             device="cpu", force_reset=True)
            if model._last_obs is not None:
                raise RuntimeError("SAC resume must reset its vector environment")
            model.load_replay_buffer(str(resume_root / "replay_buffer.pkl"))
            if model.num_timesteps != init_step:
                raise ValueError("Model steps do not match recovery metadata")
            with (resume_root / "rng.pkl").open("rb") as file:
                rng = pickle.load(file)
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_cpu"])
        else:
            norm = VecNormalize(raw_env, norm_obs=True, norm_reward=True,
                                clip_obs=10.0, gamma=args.gamma)
            model = SAC(
                "MlpPolicy", norm, seed=args.seed, device="cpu", verbose=0,
                learning_rate=args.learning_rate, gamma=args.gamma, tau=args.tau,
                learning_starts=args.learning_starts, buffer_size=args.buffer_size,
                batch_size=args.batch_size, ent_coef="auto",
                train_freq=(1, "step"), gradient_steps=1,
                policy_kwargs={"net_arch": [128, 128]},
            )
        progress = ScaleProgress(target_steps, args.progress_every,
                                 completed=init_step, previous_wall=old_wall)

        class ScaleCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                old = resume_meta["callback"] if resume_meta else {}
                self.best_q95 = float(old.get("best_q95_loss", "inf"))
                self.n_evals = int(old.get("n_evaluations", 0))
                self.next_eval = ((init_step // args.eval_every) + 1) * args.eval_every
                self.next_recovery = ((init_step // args.resume_every) + 1) * args.resume_every
                self.last_progress = None
                self.last_wall = old_wall
                self.pending_recovery = False
                self.stop_when_saved = False
                self.stopped_early = False

            def _write_progress(self, force=False):
                record = progress.snapshot(self.num_timesteps, force=force)
                if record is None:
                    return
                self.last_progress = record
                self.last_wall = record["elapsed_seconds"]
                with (output / "training_progress.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                print(_progress_line(record), flush=True)

            def _evaluate(self):
                summary, _ = evaluate_actions(
                    pool_val, run_cfg, _sb3_policy(self.model, norm),
                    episode_factory=_episode_factory(args.action_mode, args.max_band_width),
                    action_bounds=_action_bounds(args.action_mode, run_cfg,
                                                 args.max_band_width),
                )
                record = validation_selection_record(self.num_timesteps, summary)
                step_name = _checkpoint_stem(int(self.num_timesteps))
                _save_checkpoint(self.model, norm, output, step_name)
                record["checkpoint"] = step_name
                with (output / "validation_history.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                self.n_evals += 1
                if q95_is_better(summary, self.best_q95):
                    self.best_q95 = summary["q95_loss"]
                    _save_checkpoint(self.model, norm, output, "best")
                print(f"[{self.num_timesteps:>10,d}] validation: "
                      f"Q95={summary['q95_loss']:.4f} "
                      f"CVaR95={summary['cvar95_loss']:.4f} "
                      f"cost={summary['mean_transaction_cost']:.4f} "
                      f"best_Q95={self.best_q95:.4f}", flush=True)

            def _callback_state(self):
                return {"best_q95_loss": self.best_q95,
                        "n_evaluations": self.n_evals,
                        "wall_seconds": self.last_wall}

            def _on_step(self):
                if self.stop_when_saved:
                    self.stopped_early = True
                    return False
                if self.num_timesteps >= self.next_eval:
                    self._evaluate()
                    self.next_eval = ((self.num_timesteps // args.eval_every) + 1) * args.eval_every
                self._write_progress()
                if self.num_timesteps >= self.next_recovery:
                    self.pending_recovery = True
                if args.stop_after is not None and self.num_timesteps >= args.stop_after:
                    self.pending_recovery = True
                    self.stop_when_saved = True
                return True

            def _on_rollout_end(self):
                # SAC has inserted the rollout's transitions into replay buffer
                # when this hook is invoked (unlike _on_step). Gradient update
                # for this rollout happens afterward. Resume is statistical,
                # and slight update/order differences are documented.
                if self.pending_recovery:
                    _save_recovery(self.model, norm, output,
                                   step=int(self.num_timesteps),
                                   callback_state=self._callback_state(),
                                   run_sha=run_sha, torch_module=torch)
                    self.pending_recovery = False
                    self.next_recovery = (
                        (self.num_timesteps // args.resume_every) + 1
                    ) * args.resume_every
                    print(f"[recovery] saved at {self.num_timesteps:,} steps", flush=True)

        run_cfg = experiment_config(base, penalty=args.penalty,
                                    max_trade_increment=run["max_trade_increment"],
                                    include_bucket_deltas=True)
        callback = ScaleCallback()
        remaining = target_steps - init_step
        if remaining <= 0:
            raise ValueError("Recovery checkpoint has already reached the full budget")
        print(f"[scale] {args.action_mode} | {args.n_envs} envs ({args.vec_env}) | "
              f"{init_step:,}/{target_steps:,} transitions | selection: min validation Q95",
              flush=True)
        model.learn(total_timesteps=remaining, callback=callback,
                    reset_num_timesteps=not args.resume)
        if callback.stopped_early:
            print("[scale] Operational stop completed; resume using the same flags "
                  "plus --resume (omit --stop-after).", flush=True)
            return
        callback._write_progress(force=True)
        history_path = output / "validation_history.jsonl"
        history_steps = ([json.loads(line)["timesteps"] for line in
                          history_path.read_text(encoding="utf-8").splitlines()]
                         if history_path.exists() else [])
        if not history_steps or history_steps[-1] != model.num_timesteps:
            callback.num_timesteps = model.num_timesteps
            callback._evaluate()
        _save_checkpoint(model, norm, output, "final")
        _write_json(output / "summary.json", {
            "selection_metric": SELECTION_METRIC,
            "best_validation_q95_loss": callback.best_q95,
            "n_evaluations": callback.n_evals,
            "final_timesteps": model.num_timesteps,
            "training_wall_seconds": callback.last_wall,
        })
        print("[scale] Training completed:", output, flush=True)
    finally:
        if norm is not None:
            norm.close()
        else:
            raw_env.close()


def evaluate(args):
    """Reuse the audited financial evaluator with original config+split paths."""
    from train_sac_trade import evaluate as common_evaluate
    run_dir = Path(args.run_dir).resolve()
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if (run["selected_by"] != SELECTION_METRIC or
            not code_version_compatible(run["code_sha256"], code_fingerprint())):
        raise ValueError("Incompatible or changed scale training code")
    arguments = argparse.Namespace(
        config=run["config_file"], splits_dir=run["splits_dir"],
        run_dir=str(run_dir), model=args.model,
        split=args.split, allow_test=args.allow_test, max_paths=args.max_paths,
    )
    common_evaluate(arguments, action_mode=run["action_mode"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Train or resume SAC with min-Q95 selection")
    t.add_argument("--action-mode", choices=("trade", "band"), required=True)
    t.add_argument("--config", default="config.json")
    t.add_argument("--splits-dir", required=True)
    t.add_argument("--out-dir", required=True)
    t.add_argument("--timesteps", type=int, default=50_000_000)
    t.add_argument("--n-envs", type=int, default=30)
    t.add_argument("--vec-env", choices=("dummy", "subproc"), default="subproc")
    t.add_argument("--train-chunk-paths", type=int, default=16)
    t.add_argument("--eval-every", type=int, default=2_500_000)
    t.add_argument("--resume-every", type=int, default=500_000)
    t.add_argument("--progress-every", type=int, default=50_000)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--stop-after", type=int, default=None,
                   help="Operational pause after N transitions; saves full recovery snapshot")
    t.add_argument("--seed", type=int, default=123)
    t.add_argument("--train-seed", type=int, default=551177)
    t.add_argument("--penalty", type=float, default=2.0)
    t.add_argument("--max-trade-increment", type=float, default=1.0)
    t.add_argument("--max-band-width", type=float, default=1.5)
    t.add_argument("--learning-starts", type=int, default=50_000)
    t.add_argument("--buffer-size", type=int, default=500_000)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--gamma", type=float, default=0.995)
    t.add_argument("--tau", type=float, default=0.002)
    t.add_argument("--learning-rate", type=float, default=1e-4)
    e = sub.add_parser("evaluate", help="Evaluate with the original frozen dataset")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--model", default="best")
    e.add_argument("--split", choices=("validation", "test"), default="validation")
    e.add_argument("--allow-test", action="store_true")
    e.add_argument("--max-paths", type=int)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
