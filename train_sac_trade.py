"""Small, auditable SAC-Trade pilot for the fixed ENIAC book under Heston-Q.

The trading and accounting equations remain in hedging.py / rl_env.py.
This script only prepares *training* episode chunks, trains SAC, and evaluates on
an independent frozen validation split. It never chooses a model from test data.

Examples (after running forward_experiments.py prepare):
  python train_sac_trade.py train --out-dir results/sac_trade/pilot_c2_limit1 \
      --timesteps 3000 --eval-every 1500 --penalty 2 --max-trade-increment 1
  python train_sac_trade.py evaluate --run-dir results/sac_trade/pilot_c2_limit1

Dependencies for training/evaluation: stable-baselines3, gymnasium, torch.
The pure rollout and statistics helpers can be tested without those packages.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from dataclasses import asdict, replace
import json
import math
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np

from forward_experiments import _read_config, iter_train_chunks, load_split
from hedging import summarize_hedge
from rl_env import ForwardEnvConfig, ForwardEpisode, ForwardEpisodePool, HestonForwardGymEnv
from band_policy import (BAND_EXECUTION, ForwardBandEpisode, HestonForwardBandGymEnv,
                         require_current_band_execution)


def experiment_config(base: ForwardEnvConfig, *, penalty: float, max_trade_increment: float,
                      include_bucket_deltas: bool = True) -> ForwardEnvConfig:
    """Two *experimental* overrides; financial dataset configuration stays frozen."""
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    if not math.isfinite(max_trade_increment) or not 0 < max_trade_increment <= base.max_position:
        raise ValueError("max_trade_increment must be in (0, max_position]")
    return replace(base, quadratic_reward_penalty=penalty,
                   max_trade_increment=max_trade_increment,
                   include_bucket_deltas=include_bucket_deltas)


def evaluate_actions(
    pool: ForwardEpisodePool, config: ForwardEnvConfig,
    policy: Callable[[np.ndarray], np.ndarray],
    *, max_paths: int | None = None,
    episode_factory: Callable[[ForwardEpisodePool, ForwardEnvConfig], ForwardEpisode] = ForwardEpisode,
    action_bounds: tuple[float, float] | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Deterministic, sequential held-out rollouts using exactly the live RL transition.

    `policy` receives current observations only. No path future or latent
    scenario arrays are passed to the policy. For Band, `actions` are widths;
    `executed_change` records the financially executed increments.
    """
    n_available = len(pool.paths.spot)
    if max_paths is None:
        n = n_available
    elif type(max_paths) is int and 1 <= max_paths <= n_available:
        n = max_paths
    else:
        raise ValueError("max_paths must lie between 1 and the number of trajectories")
    episode = episode_factory(pool, config)
    low, high = ((-config.max_trade_increment, config.max_trade_increment)
                 if action_bounds is None else action_bounds)
    terminals = np.empty(n, dtype=float)
    costs = np.empty(n, dtype=float)
    trades = np.empty(n, dtype=int)
    turnover = np.empty(n, dtype=float)
    rewards = np.empty(n, dtype=float)
    actions = np.empty((n, pool.paths.spot.shape[1]-1, len(config.expiries_days)), dtype=float)
    executed = np.empty_like(actions)
    for i in range(n):
        obs, _ = episode.reset(path_index=i)
        done = False
        reward_sum = 0.0
        day = 0
        while not done:
            action = np.asarray(policy(obs.copy()), dtype=float)
            if action.shape != (len(config.expiries_days),) or not np.all(np.isfinite(action)):
                raise ValueError("policy must return one finite action per hedge instrument")
            if np.any(action < low - 1e-5) or np.any(action > high + 1e-5):
                raise ValueError("policy action exceeds declared action limits")
            actions[i, day] = action
            obs, reward, done, truncated, info = episode.step(action)
            executed[i, day] = info["executed_change"]
            if truncated:
                raise RuntimeError("Pilot episodes must end through scheduled maturity")
            reward_sum += reward
            day += 1
        terminals[i] = info["terminal_pnl"]
        costs[i] = info["total_transaction_cost"]
        trades[i] = info["trade_count"]
        turnover[i] = info["turnover"]
        rewards[i] = reward_sum
    # Shared RU VaR/CVaR definition from the established financial baseline.
    from types import SimpleNamespace
    result = SimpleNamespace(terminal_pnl=terminals, transaction_cost=costs,
                             trades=trades, turnover=turnover)
    summary = summarize_hedge(result)
    summary.update(n_paths=n, mean_episode_reward=float(rewards.mean()),
                   std_terminal_pnl=float(terminals.std(ddof=1)) if n > 1 else 0.0)
    return summary, dict(terminal_pnl=terminals, transaction_cost=costs,
                         trades=trades, turnover=turnover,
                         episode_reward=rewards, actions=actions,
                         executed_change=executed)


def _load_deps():
    try:
        import gymnasium  # noqa: F401
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise RuntimeError("Install dependencies with python -m pip install -r requirements.txt") from exc
    return SAC, DummyVecEnv, VecNormalize, BaseCallback


class StreamingQForwardGymEnv(HestonForwardGymEnv):
    """Single-env deterministic stream: each generated Q path used once per chunk.

    This is *training only*. Never accesses validation/test arrays. Whole Heston
    paths are pre-marked by the existing pricer; the policy sees only today's
    observation. Chunks are replaced at episode boundaries.
    """
    def __init__(self, config_path: str | Path, *, train_seed: int,
                 paths_per_chunk: int, n_chunks: int,
                 penalty: float, max_trade_increment: float,
                 include_bucket_deltas: bool):
        self._chunks = iter_train_chunks(config_path, base_seed=train_seed,
                                         chunk_size=paths_per_chunk, n_chunks=n_chunks)
        self._order = np.empty(0, dtype=int)
        self._cursor = 0
        manifest, pool, base = next(self._chunks)
        config = experiment_config(base, penalty=penalty,
                                   max_trade_increment=max_trade_increment,
                                   include_bucket_deltas=include_bucket_deltas)
        super().__init__(pool, config, seed=train_seed)
        self._new_order(pool, manifest["seed"])
        self.chunk_index = 0
        self.generated_chunks = 1

    def _new_order(self, pool: ForwardEpisodePool, seed: int) -> None:
        self._order = np.random.default_rng(seed).permutation(len(pool.paths.spot))
        self._cursor = 0

    def _replace_core(self, pool: ForwardEpisodePool, *, seed: int) -> None:
        self.core = ForwardEpisode(pool, self.core.config, seed=seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if options and "path_index" in options:
            raise ValueError("Training path selection is controlled by the stream")
        if self._cursor >= len(self._order):
            manifest, pool, base = next(self._chunks)
            if base != replace(self.core.config, quadratic_reward_penalty=base.quadratic_reward_penalty,
                               max_trade_increment=base.max_trade_increment,
                               include_bucket_deltas=base.include_bucket_deltas):
                raise ValueError("Training financial configuration changed across chunks")
            self._replace_core(pool, seed=int(manifest["seed"]))
            self._new_order(pool, manifest["seed"])
            self.chunk_index = manifest["chunk_index"]
            self.generated_chunks += 1
        path_index = int(self._order[self._cursor])
        self._cursor += 1
        return super().reset(seed=seed, options={"path_index": path_index})


class StreamingQBandGymEnv(StreamingQForwardGymEnv):
    """Stream fresh Q-Heston paths while applying width actions, never trade actions."""

    def __init__(self, config_path: str | Path, *, max_band_width: float, **kwargs):
        self.max_band_width = max_band_width
        super().__init__(config_path, **kwargs)
        self._replace_core(self.core.pool, seed=kwargs["train_seed"])
        from gymnasium import spaces
        self.action_space = spaces.Box(
            low=0.0, high=max_band_width,
            shape=(len(self.core.config.expiries_days),), dtype=np.float32,
        )

    def _replace_core(self, pool: ForwardEpisodePool, *, seed: int) -> None:
        self.core = ForwardBandEpisode(pool, self.core.config,
                                       max_band_width=self.max_band_width, seed=seed)


def _episode_factory(action_mode: str, max_band_width: float):
    if action_mode == "band":
        return lambda pool, cfg: ForwardBandEpisode(pool, cfg, max_band_width=max_band_width)
    return ForwardEpisode


def _action_bounds(action_mode: str, config: ForwardEnvConfig, max_band_width: float):
    return ((0.0, max_band_width) if action_mode == "band"
            else (-config.max_trade_increment, config.max_trade_increment))


def _sb3_policy(model, vec_norm):
    def act(obs: np.ndarray) -> np.ndarray:
        normalized = vec_norm.normalize_obs(obs[None, :])
        prediction, _ = model.predict(normalized, deterministic=True)
        return np.asarray(prediction[0], dtype=float)
    return act


def _save_checkpoint(model, vec_norm, folder: Path, name: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.save(str(folder / name))
    vec_norm.save(str(folder / f"{name}.vecnormalize.pkl"))


def _checkpoint_stem(step: int) -> str:
    """Unique, sortable model/VecNormalize basename for an evaluated step."""
    if not isinstance(step, int) or step <= 0:
        raise ValueError("checkpoint timestep must be a positive integer")
    return f"step_{step:012d}"


def _validate_model_stem(name: str) -> str:
    """Accepted model names; do not allow arbitrary paths in --model."""
    if name not in ("best", "final") and not re.fullmatch(r"step_[0-9]{12}", name):
        raise ValueError("--model must be best, final, or step_<12-digit-timestep>")
    return name


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _format_duration(seconds: float | None) -> str:
    """Human-readable duration (including multi-day runs); no fake precision."""
    if seconds is None or not math.isfinite(seconds):
        return "--"
    whole = max(0, int(round(seconds)))
    days, remainder = divmod(whole, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    time_part = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}d {time_part}" if days else time_part


class TrainingProgress:
    """Wall-clock training progress, with a rolling ETA including evaluations.

    Per-step callback checks are cheap; the clock is only consulted on reports.
    ETA is an extrapolation, not a deadline: new Heston chunks, SAC updates,
    validation and I/O may change the future throughput.
    """

    def __init__(self, total_steps: int, every_steps: int,
                 *, clock: Callable[[], float] = time.perf_counter):
        if total_steps <= 0 or every_steps <= 0:
            raise ValueError("total_steps and every_steps must be positive")
        self.total_steps = total_steps
        self.every_steps = every_steps
        self._clock = clock
        self._started = self._last_time = clock()
        self._last_steps = 0
        self._next_report = every_steps
        self._rolling_rate: float | None = None

    def snapshot(self, steps: int, *, force: bool = False) -> dict | None:
        if not 0 <= steps <= self.total_steps:
            raise ValueError("completed timesteps outside training range")
        if steps == self._last_steps:
            return None  # Never log a duplicate at the final step.
        if not force and steps < self._next_report:
            return None
        now = self._clock()
        elapsed = max(0.0, now - self._started)
        interval = now - self._last_time
        if interval > 0:
            latest_rate = (steps - self._last_steps) / interval
            self._rolling_rate = (latest_rate if self._rolling_rate is None else
                                  0.65 * self._rolling_rate + 0.35 * latest_rate)
        average_rate = steps / elapsed if elapsed > 0 else None
        rate = self._rolling_rate if self._rolling_rate is not None else average_rate
        eta = ((self.total_steps - steps) / rate
               if rate is not None and rate > 0 else None)
        end_at = (datetime.now(timezone.utc) + timedelta(seconds=eta)).isoformat(
            timespec="seconds") if eta is not None else None
        self._last_steps = steps
        self._last_time = now
        self._next_report = (steps // self.every_steps + 1) * self.every_steps
        return {
            "timesteps": steps,
            "total_timesteps": self.total_steps,
            "progress_percent": 100.0 * steps / self.total_steps,
            "elapsed_seconds": elapsed,
            "average_steps_per_second": average_rate,
            "rolling_steps_per_second": rate,
            "eta_seconds": eta,
            "estimated_finish_utc": end_at,
        }


def _progress_line(snapshot: dict) -> str:
    step = snapshot["timesteps"]
    total = snapshot["total_timesteps"]
    rate = snapshot["rolling_steps_per_second"]
    speed = f"{rate:.1f} steps/s" if rate is not None else "calculating speed"
    return (f"[progress] {step:,}/{total:,} ({snapshot['progress_percent']:.1f}%) "
            f"| elapsed {_format_duration(snapshot['elapsed_seconds'])} "
            f"| {speed} | ETA ~{_format_duration(snapshot['eta_seconds'])}")


def train(args: argparse.Namespace, *, action_mode: str = "trade") -> None:
    SAC, DummyVecEnv, VecNormalize, BaseCallback = _load_deps()
    if args.progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if args.timesteps <= 0 or args.eval_every <= 0 or args.train_chunk_paths <= 0:
        raise ValueError("timesteps, eval_every and train_chunk_paths must be positive")
    if args.learning_starts < 0 or args.learning_starts >= args.timesteps:
        raise ValueError("learning_starts must be nonnegative and smaller than timesteps")
    if args.batch_size < 1 or args.buffer_size < args.batch_size:
        raise ValueError("buffer_size must be >= batch_size >= 1")
    if args.seed < 0 or args.train_seed < 0:
        raise ValueError("seeds must be nonnegative")
    if not (0 < args.gamma <= 1 and 0 < args.tau <= 1 and args.learning_rate > 0):
        raise ValueError("invalid SAC discount, target update, or learning rate")
    if action_mode not in ("trade", "band"):
        raise ValueError("invalid action mode")
    pool_val, base, val_metadata = load_split(args.splits_dir, "validation", args.config)
    if action_mode == "band":
        if (not math.isfinite(args.max_band_width) or
                not 0 < args.max_band_width <= base.max_position):
            raise ValueError("max_band_width must lie in (0, max_position]")
        args.max_trade_increment = base.max_position  # width action permits full rebalance
        args.include_bucket_deltas = True
    manifest_path = Path(args.splits_dir) / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        split_seeds = {int(item["seed"]) for item in manifest["splits"].values()}
    else:
        split_seeds = {int(val_metadata["seed"])}
    if args.train_seed in split_seeds:
        raise ValueError("train_seed must differ from every prepared split seed")
    _, fingerprint = _read_config(args.config)
    cfg = experiment_config(base, penalty=args.penalty,
                            max_trade_increment=args.max_trade_increment,
                            include_bucket_deltas=args.include_bucket_deltas)
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is not empty. Use a new run directory to avoid overwriting results")
    output.mkdir(parents=True, exist_ok=True)
    from importlib.metadata import version
    package_versions = {name: version(name) for name in
                        ("stable-baselines3", "gymnasium", "torch", "numpy")}
    run_info = {
        "algorithm": "SAC-Band" if action_mode == "band" else "SAC-Trade",
        "action_mode": action_mode, "measure": "Q", "regime_xi": _read_config(args.config)[0]["heston"]["xi"],
        "financial_config_sha256": fingerprint,
        "validation_data_sha256": val_metadata["data_sha256"],
        "seed": args.seed, "train_seed": args.train_seed,
        "timesteps": args.timesteps, "eval_every": args.eval_every,
        "progress_every": args.progress_every,
        "paths_per_chunk": args.train_chunk_paths,
        "penalty": args.penalty, "max_trade_increment": args.max_trade_increment,
        **({"max_band_width": args.max_band_width,
            "band_execution": BAND_EXECUTION} if action_mode == "band" else {}),
        "include_bucket_deltas": args.include_bucket_deltas,
        "learning_starts": args.learning_starts, "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "gamma": args.gamma, "learning_rate": args.learning_rate,
        "tau": args.tau, "network": [128, 128],
        "package_versions": package_versions,
        "save_eval_checkpoints": args.save_eval_checkpoints,
        "config": asdict(cfg), "selected_by": "validation_mean_episode_reward",
        "note": "Validation split is for model selection; test split is not touched by training.",
    }
    _write_json(output / "run.json", run_info)
    # Each complete episode has exactly last_expiry steps. Retain a few extra
    # chunks for resets after a timestep boundary; do not preload all of them.
    n_episodes = math.ceil(args.timesteps / cfg.expiries_days[-1]) + 3
    n_chunks = math.ceil(n_episodes / args.train_chunk_paths) + 1
    stream_class = StreamingQBandGymEnv if action_mode == "band" else StreamingQForwardGymEnv
    extra_env = {"max_band_width": args.max_band_width} if action_mode == "band" else {}
    env_factory = lambda: stream_class(
        args.config, train_seed=args.train_seed, paths_per_chunk=args.train_chunk_paths,
        n_chunks=n_chunks, penalty=args.penalty,
        max_trade_increment=args.max_trade_increment,
        include_bucket_deltas=args.include_bucket_deltas, **extra_env)
    base_env = DummyVecEnv([env_factory])
    env = VecNormalize(base_env, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, gamma=args.gamma)
    model = SAC(
        "MlpPolicy", env, seed=args.seed, device="cpu", verbose=0,
        learning_rate=args.learning_rate, gamma=args.gamma, tau=args.tau,
        learning_starts=args.learning_starts, buffer_size=args.buffer_size,
        batch_size=args.batch_size, ent_coef="auto",
        policy_kwargs={"net_arch": [128, 128]},
    )

    progress = TrainingProgress(args.timesteps, args.progress_every)

    class ValidationCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.best = -float("inf")
            self.history: list[dict] = []
            self.next_eval = args.eval_every
            self.last_progress: dict | None = None

        def _report_progress(self, *, force: bool = False) -> None:
            record = progress.snapshot(int(self.num_timesteps), force=force)
            if record is None:
                return
            self.last_progress = record
            with (output / "training_progress.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(_progress_line(record), flush=True)

        def _evaluate(self) -> dict:
            summary, _ = evaluate_actions(
                pool_val, cfg, _sb3_policy(self.model, env),
                episode_factory=_episode_factory(action_mode, getattr(args, "max_band_width", 1.5)),
                action_bounds=_action_bounds(action_mode, cfg, getattr(args, "max_band_width", 1.5)),
            )
            record = {"timesteps": self.num_timesteps, **summary}
            if args.save_eval_checkpoints:
                stem = _checkpoint_stem(int(self.num_timesteps))
                _save_checkpoint(self.model, env, output, stem)
                record["checkpoint"] = stem
            self.history.append(record)
            with (output / "validation_history.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{self.num_timesteps:>8d}] val: mean_reward={summary['mean_episode_reward']:.3f} "
                  f"q95={summary['q95_loss']:.3f} CVaR95={summary['cvar95_loss']:.3f} "
                  f"cost={summary['mean_transaction_cost']:.3f}", flush=True)
            if summary["mean_episode_reward"] > self.best:
                self.best = summary["mean_episode_reward"]
                _save_checkpoint(self.model, env, output, "best")
            return record

        def _on_step(self) -> bool:
            if self.num_timesteps >= self.next_eval:
                self._evaluate()
                self.next_eval += args.eval_every
            # Deliberately after validation: wall-clock rate includes evaluation,
            # checkpointing, and fresh training-path generation overhead.
            self._report_progress()
            return True

    callback = ValidationCallback()
    print(f"[progress] starting {args.timesteps:,} steps; reports every "
          f"{args.progress_every:,} steps (ETA estimates adjust during training)", flush=True)
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback,
                    reset_num_timesteps=True)
        callback._report_progress(force=True)
        _save_checkpoint(model, env, output, "final")
        # Even short smoke runs must contain a selected checkpoint and metrics.
        if not callback.history or callback.history[-1]["timesteps"] != model.num_timesteps:
            callback.num_timesteps = model.num_timesteps
            callback._evaluate()
        _write_json(output / "summary.json", {
            "best_validation_mean_episode_reward": callback.best,
            "n_evaluations": len(callback.history),
            "final_timesteps": model.num_timesteps,
            "training_wall_seconds": (callback.last_progress or {}).get("elapsed_seconds"),
            "selection_metric": "validation_mean_episode_reward",
        })
        print("Models and logs saved to:", output)
    finally:
        env.close()


def evaluate(args: argparse.Namespace, *, action_mode: str = "trade") -> None:
    SAC, DummyVecEnv, VecNormalize, _ = _load_deps()
    root = Path(args.run_dir)
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if run.get("action_mode", "trade") != action_mode:
        raise ValueError("Model action semantics differ from this evaluator; use the matching script")
    if action_mode == "band":
        require_current_band_execution(run)
    _, fingerprint = _read_config(args.config)
    if fingerprint != run["financial_config_sha256"]:
        raise ValueError("Financial config differs from training. Restore the original config")
    if args.split == "test" and not args.allow_test:
        raise ValueError("Test split is held out: use --allow-test only after fixing all choices")
    pool, base, meta = load_split(args.splits_dir, args.split, args.config)
    if args.split == "validation" and meta["data_sha256"] != run["validation_data_sha256"]:
        raise ValueError("Validation dataset differs from the one used during training")
    cfg = experiment_config(base, penalty=run["penalty"],
                            max_trade_increment=run["max_trade_increment"],
                            include_bucket_deltas=run["include_bucket_deltas"])
    model_name = _validate_model_stem(args.model)
    model_file = root / f"{model_name}.zip"
    norm_file = root / f"{model_name}.vecnormalize.pkl"
    if not model_file.exists() or not norm_file.exists():
        raise FileNotFoundError(f"Missing model or normalization snapshot: {args.model}")
    eval_env_class = HestonForwardBandGymEnv if action_mode == "band" else HestonForwardGymEnv
    extra_env = {"max_band_width": run["max_band_width"]} if action_mode == "band" else {}
    env = DummyVecEnv([lambda: eval_env_class(pool, cfg, seed=run["seed"], **extra_env)])
    try:
        norm = VecNormalize.load(str(norm_file), env)
        norm.training = False
        norm.norm_reward = False
        model = SAC.load(str(model_file), device="cpu")
        if model.action_space != env.action_space:
            raise ValueError("Saved model action space is not the requested financial action space")
        summary, traces = evaluate_actions(
            pool, cfg, _sb3_policy(model, norm), max_paths=args.max_paths,
            episode_factory=_episode_factory(action_mode, run.get("max_band_width", 1.5)),
            action_bounds=_action_bounds(action_mode, cfg, run.get("max_band_width", 1.5)),
        )
        target = root / f"{args.split}_{args.model}_evaluation"
        _write_json(target.with_suffix(".json"), {
            "split": args.split, "model": args.model, "measure": "Q",
            "action_mode": action_mode,
            **({"band_execution": BAND_EXECUTION} if action_mode == "band" else {}),
            "financial_config_sha256": fingerprint, "metrics": summary,
        })
        np.savez_compressed(target.with_suffix(".npz"), **traces)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("Evaluation saved:", target.with_suffix(".json").resolve())
        print("Evaluation saved:", target.with_suffix(".npz").resolve())
    finally:
        env.close()


def main(*, action_mode: str = "trade") -> None:
    if action_mode not in ("trade", "band"):
        raise ValueError("invalid action mode")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help=f"SAC-{action_mode} pilot on independent Q-Heston chunks")
    train_parser.add_argument("--config", default="config.json")
    train_parser.add_argument("--splits-dir", default="results/forward_experiments/pilot_xi_0p20")
    train_parser.add_argument("--out-dir", required=True)
    train_parser.add_argument("--timesteps", type=int, default=3000)
    train_parser.add_argument("--eval-every", type=int, default=1500)
    train_parser.add_argument("--progress-every", type=int, default=5000,
                              help="Print progress/speed/ETA and log it every N steps (default: 5000)")
    train_parser.add_argument("--train-chunk-paths", type=int, default=16)
    train_parser.add_argument("--seed", type=int, default=123)
    train_parser.add_argument("--train-seed", type=int, default=551177)
    train_parser.add_argument("--penalty", type=float, default=2.0)
    if action_mode == "band":
        train_parser.add_argument("--max-band-width", type=float, default=1.5,
                                  help="SAC-Band action interval is [0, this half-width]")
    else:
        train_parser.add_argument("--max-trade-increment", type=float, default=1.0)
    train_parser.add_argument("--save-eval-checkpoints", action=argparse.BooleanOptionalAction,
                              default=True, help="Save model + VecNormalize at every validation (default: yes)")
    if action_mode == "trade":
        train_parser.add_argument("--include-bucket-deltas", action=argparse.BooleanOptionalAction,
                                  default=True, help="Expose current Heston bucket deltas to Trade, as Band uses")
    train_parser.add_argument("--learning-starts", type=int, default=256)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--buffer-size", type=int, default=50000)
    train_parser.add_argument("--gamma", type=float, default=0.995)
    train_parser.add_argument("--tau", type=float, default=0.002)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    eval_parser = sub.add_parser("evaluate", help="Evaluate a saved deterministic policy")
    eval_parser.add_argument("--config", default="config.json")
    eval_parser.add_argument("--splits-dir", default="results/forward_experiments/pilot_xi_0p20")
    eval_parser.add_argument("--run-dir", required=True)
    eval_parser.add_argument("--split", choices=("validation", "test"), default="validation")
    eval_parser.add_argument("--allow-test", action="store_true")
    eval_parser.add_argument("--model", default="best",
                             help="best, final, or step_000000005000 (saved evaluation checkpoint)")
    eval_parser.add_argument("--max-paths", type=int)
    args = parser.parse_args()
    if args.command == "train":
        train(args, action_mode=action_mode)
    else:
        evaluate(args, action_mode=action_mode)


if __name__ == "__main__":
    main()
