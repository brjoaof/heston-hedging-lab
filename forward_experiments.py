"""Reproducible Q-Heston episode splits and transparent forward-hedging baselines.

No RL optimizer is trained here. Validation selects the scale of candidate
risk penalties; test data must remain untouched until choices are frozen.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from hedging import (
    BookPaths, ForwardHedgeResults, execute_forward_trades,
    forward_prices, simulate_forward_hedge, summarize_hedge,
)
from heston import HestonPaths, MarketDynamics
from rl_env import ForwardEpisodePool, ForwardEnvConfig, prepare_forward_episodes

SCHEMA_VERSION = 1
SPLITS = ("train", "validation", "test")
BASELINES = ("no_hedge", "hold_initial", "delta_daily", "delta_periodic_21", "delta_band_0p5")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_config(config_path: str | Path) -> tuple[dict, str]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as file:
        cfg = json.load(file)["forward_experiment"]
    # Canonical scientific configuration fingerprint, unaffected by whitespace or
    # unrelated legacy blocks in config.json.
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if cfg["measure"] != "Q":
        raise ValueError("Only Q experiments are supported")
    return cfg, fingerprint


def _paths_for_split(folder: str | Path, split: str) -> tuple[Path, Path]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    root = Path(folder)
    return root / f"{split}.npz", root / f"{split}.json"


def save_split(
    folder: str | Path, split: str, pool: ForwardEpisodePool,
    config_path: str | Path, *, seed: int,
) -> dict:
    """Persist a fully marked Q pool, with byte-level integrity and config ID."""
    _, fingerprint = _read_config(config_path)
    data_path, meta_path = _paths_for_split(folder, split)
    if data_path.exists() or meta_path.exists():
        raise FileExistsError(f"Refusing to overwrite {split} data; use a fresh directory")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if pool.paths.measure != "Q" or pool.market_q.measure != "Q":
        raise ValueError("Only Q paths can be saved")
    b = pool.book_paths
    if b.contract_prices is None or b.bucket_delta is None:
        raise ValueError("Need contract prices and bucket deltas for a reproducible split")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        time=pool.paths.time_years, spot=pool.paths.spot,
        variance=pool.paths.variance, raw_variance=pool.paths.raw_variance,
        option_value=b.value, option_delta=b.delta,
        option_gamma=b.gamma, option_variance_sensitivity=b.variance_sensitivity,
        settlements=b.settlements, contract_prices=b.contract_prices,
        bucket_delta=b.bucket_delta, initial_positions=pool.initial_positions,
    )
    np.savez_compressed(data_path, **arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "measure": "Q",
        "seed": seed,
        "n_paths": int(len(pool.paths.spot)),
        "n_steps": int(pool.paths.spot.shape[1] - 1),
        "config_sha256": fingerprint,
        "data_sha256": _sha256(data_path),
        "heston_q": _read_config(config_path)[0]["heston"],
        "market": asdict(pool.market_q),
        "fields": sorted(arrays),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def load_split(
    folder: str | Path, split: str, config_path: str | Path,
) -> tuple[ForwardEpisodePool, ForwardEnvConfig, dict]:
    """Fail closed if contents or financial configuration have changed."""
    cfg, fingerprint = _read_config(config_path)
    data_path, meta_path = _paths_for_split(folder, split)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if (metadata["schema_version"] != SCHEMA_VERSION or metadata["split"] != split or
            metadata["measure"] != "Q" or metadata["config_sha256"] != fingerprint):
        raise ValueError("Split metadata, Q measure or financial configuration mismatch")
    if metadata["data_sha256"] != _sha256(data_path):
        raise ValueError("Split NPZ checksum mismatch")
    market = MarketDynamics(**metadata["market"])
    expected_market = MarketDynamics(
        measure=cfg["measure"], spot0=cfg["spot0"],
        risk_free_rate=cfg["risk_free_rate"], dividend_yield=cfg["dividend_yield"],
    )
    if market != expected_market or market.measure != "Q":
        raise ValueError("Saved market is inconsistent with current configuration")
    with np.load(data_path, allow_pickle=False) as f:
        if set(f.files) != set(metadata["fields"]):
            raise ValueError("Split fields do not match manifest")
        arrays = {name: f[name].copy() for name in f.files}
    paths = HestonPaths(arrays["time"], arrays["spot"], arrays["variance"],
                        arrays["raw_variance"], "Q")
    b = BookPaths(arrays["option_value"], arrays["option_delta"],
                  arrays["option_gamma"], arrays["option_variance_sensitivity"],
                  arrays["settlements"], arrays["contract_prices"], arrays["bucket_delta"])
    pool = ForwardEpisodePool(paths, b, arrays["initial_positions"], market)
    spec = ForwardEnvConfig.from_mapping(cfg)
    if (paths.spot.shape != (metadata["n_paths"], metadata["n_steps"] + 1) or
            paths.spot.shape != paths.variance.shape or
            b.bucket_delta.shape != (*paths.spot.shape, len(spec.expiries_days)) or
            pool.initial_positions.shape != (metadata["n_paths"], len(spec.expiries_days))):
        raise ValueError("Saved episode arrays have inconsistent shapes")
    # The existing episode validates more detailed constraints and accounting inputs.
    from rl_env import ForwardEpisode
    ForwardEpisode(pool, spec)
    return pool, spec, metadata


def prepare_splits(
    config_path: str | Path, folder: str | Path, *,
    sizes: tuple[int, int, int] = (64, 32, 32),
    seeds: tuple[int, int, int] = (1101, 2202, 3303),
) -> dict:
    """Disjoint named pseudo-random streams, with no silent overwrite."""
    if len(sizes) != 3 or any(type(v) is not int or v <= 0 for v in sizes):
        raise ValueError("Three positive split sizes are required")
    if (len(seeds) != 3 or len(set(seeds)) != 3 or
            any(type(v) is not int or v < 0 for v in seeds)):
        raise ValueError("Train/validation/test seeds must be distinct nonnegative integers")
    _read_config(config_path)
    if any(_paths_for_split(folder, name)[i].exists() for name in SPLITS for i in (0, 1)):
        raise FileExistsError("Output contains an existing split; refuse overwrite")
    manifest = {"schema_version": SCHEMA_VERSION, "splits": {}}
    for name, n, seed in zip(SPLITS, sizes, seeds):
        pool, _ = prepare_forward_episodes(config_path, n_paths=n, seed=seed)
        manifest["splits"][name] = save_split(folder, name, pool, config_path, seed=seed)
    root = Path(folder)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def iter_train_chunks(
    config_path: str | Path, *, base_seed: int, chunk_size: int,
    n_chunks: int, start_chunk: int = 0,
):
    """Regenerate Q training pools in bounded memory, one chunk at a time.

    Each chunk has a reproducible seed derived from its index, rather than
    cycling a small frozen pool. Validation/test pools are never accessed.
    This generator is an infrastructure hook, not an SB3 training wrapper.
    """
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if type(n_chunks) is not int or n_chunks < 1:
        raise ValueError("n_chunks must be positive")
    if type(start_chunk) is not int or not 0 <= start_chunk <= n_chunks:
        raise ValueError("start_chunk must be an index in [0, n_chunks]")
    _read_config(config_path)
    for chunk in range(start_chunk, n_chunks):
        # Domain separation: these seeds are separate from split manifest seeds.
        seed = int(np.random.SeedSequence([base_seed, chunk, 0x48455354])
                   .generate_state(1, dtype=np.uint64)[0])
        pool, cfg = prepare_forward_episodes(config_path, n_paths=chunk_size, seed=seed)
        yield {"chunk_index": chunk, "seed": seed}, pool, cfg


def _target_forward_positions(pool: ForwardEpisodePool, cfg: ForwardEnvConfig, day: int) -> np.ndarray:
    """Current Q-delta per maturity, without reading day+1 market variables."""
    bucket = pool.book_paths.bucket_delta
    if bucket is None:
        raise ValueError("Delta baselines require bucket_delta from current Q option marks")
    remaining = np.maximum(np.asarray(cfg.expiries_days, dtype=float) - day, 0.0)
    df_dspot = np.exp((pool.market_q.risk_free_rate - pool.market_q.dividend_yield)
                       * remaining / cfg.trading_days_per_year)
    target = np.clip(-bucket[:, day, :] / df_dspot[None, :], 0.0, cfg.max_position)
    target[:, np.asarray(cfg.expiries_days) <= day] = 0.0
    return target


def baseline_actions(
    pool: ForwardEpisodePool, cfg: ForwardEnvConfig, strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct causal proposed actions. The financial engine executes all trades.

    The action preview updates positions using the SAME bounded execution rule;
    forced closing only changes the position after the relevant expiry.
    """
    if strategy not in BASELINES:
        raise ValueError(f"Unknown baseline: {strategy}")
    n_paths, n_times = pool.paths.spot.shape
    n_buckets = len(cfg.expiries_days)
    initial = (np.zeros_like(pool.initial_positions) if strategy == "no_hedge"
               else pool.initial_positions.copy())
    pos = initial.copy()
    actions = np.zeros((n_paths, n_times - 1, n_buckets), dtype=float)
    for day in range(n_times - 1):
        if strategy in ("no_hedge", "hold_initial"):
            continue
        target = _target_forward_positions(pool, cfg, day)
        if strategy == "delta_periodic_21" and day % 21 != 0:
            target = pos.copy()
        elif strategy == "delta_band_0p5":
            gap = pos - target
            target = np.where(np.abs(gap) > 0.5, target + np.sign(gap) * 0.5, pos)
        requested = target - pos
        actions[:, day, :] = np.clip(requested, -cfg.max_trade_increment, cfg.max_trade_increment)
        fwd = forward_prices(pool.paths.spot[:, day], day, cfg.expiries_days,
                             pool.market_q.risk_free_rate,
                             dividend_yield=pool.market_q.dividend_yield,
                             trading_days_per_year=cfg.trading_days_per_year)
        pos, _, _, _ = execute_forward_trades(
            pos, actions[:, day, :], fwd, np.asarray(cfg.expiries_days) > day,
            spread_half_pcts=cfg.spread_half_pcts, fixed_cost=cfg.fixed_cost,
            min_trade=cfg.min_trade, max_trade_increment=cfg.max_trade_increment,
            max_position=cfg.max_position,
        )
        pos[:, np.asarray(cfg.expiries_days) == day + 1] = 0.0
    return initial, actions


def evaluate_baseline(
    pool: ForwardEpisodePool, cfg: ForwardEnvConfig, strategy: str,
) -> ForwardHedgeResults:
    init, actions = baseline_actions(pool, cfg, strategy)
    return simulate_forward_hedge(
        pool.paths, pool.book_paths, pool.market_q,
        initial_positions=init, actions=actions,
        expiries_days=cfg.expiries_days,
        spread_half_pcts=cfg.spread_half_pcts,
        fixed_cost=cfg.fixed_cost, min_trade=cfg.min_trade,
        max_trade_increment=cfg.max_trade_increment,
        max_position=cfg.max_position,
        trading_days_per_year=cfg.trading_days_per_year,
    )


def reward_increments(result: ForwardHedgeResults) -> np.ndarray:
    """Match ForwardEpisode.step: initial hedge cost is in step zero."""
    steps = np.diff(result.wealth, axis=1)
    steps[:, 0] = result.wealth[:, 1]  # from zero before hedge opening
    if not np.allclose(steps.sum(axis=1), result.terminal_pnl, rtol=1e-11, atol=1e-8):
        raise AssertionError("Step rewards must telescope to terminal P&L")
    return steps


def calibrate_reward_candidates(
    pool: ForwardEpisodePool, cfg: ForwardEnvConfig,
    *, fraction: tuple[float, ...] = (0.0, 0.05, 0.1, 0.2),
) -> dict:
    """Validation-only penalty grid, NOT a trained or selected RL objective.

    At the 90th percentile of absolute daily net P&L, a term c*p**2 has
    magnitude fraction*|p|. This scale rule prevents arbitrary reuse of the
    ENIAC coefficient in a different financial regime.
    """
    if any(not math.isfinite(x) or x < 0 for x in fraction):
        raise ValueError("Penalty fractions must be finite and nonnegative")
    arr = np.concatenate([
        np.abs(reward_increments(evaluate_baseline(pool, cfg, name))).ravel()
        for name in ("hold_initial", "delta_daily")
    ])
    scale = float(np.quantile(arr, 0.90))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot calibrate penalty from degenerate reference P&L")
    return {
        "reference_policies": ["hold_initial", "delta_daily"],
        "p90_absolute_step_pnl": scale,
        "candidate_coefficients": [{"relative_penalty_at_p90": f, "coefficient": f / scale}
                                   for f in fraction],
        "interpretation": "Candidate values only; choose using validation, never test",
    }


def report_baselines(
    pool: ForwardEpisodePool, cfg: ForwardEnvConfig, folder: str | Path,
    *, split: str, config_sha256: str,
) -> list[dict]:
    """Export finite-sample Q results; no policy ranking or test tuning."""
    if split not in ("validation", "test"):
        raise ValueError("Report only on validation or held-out test")
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    summary = []
    traces = {}
    for name in BASELINES:
        result = evaluate_baseline(pool, cfg, name)
        summary.append({"strategy": name, **summarize_hedge(result)})
        traces[f"{name}_terminal_pnl"] = result.terminal_pnl
        traces[f"{name}_transaction_cost"] = result.transaction_cost
        traces[f"{name}_trades"] = result.trades
        traces[f"{name}_turnover"] = result.turnover
    with (root / f"{split}_baselines.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    np.savez_compressed(root / f"{split}_baseline_per_path.npz", **traces)
    metadata = {
        "split": split, "measure": "Q", "n_paths": int(len(pool.paths.spot)),
        "config_sha256": config_sha256, "metrics": summary,
        "note": "Small-sample q95/CVaR95 may be unstable. Q losses are not P probabilities.",
    }
    (root / f"{split}_baselines.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "baseline", "calibrate"))
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--out-dir", default="results/forward_experiments/pilot_xi_0p20")
    parser.add_argument("--train-paths", type=int, default=64)
    parser.add_argument("--validation-paths", type=int, default=32)
    parser.add_argument("--test-paths", type=int, default=32)
    parser.add_argument("--train-seed", type=int, default=1101)
    parser.add_argument("--validation-seed", type=int, default=2202)
    parser.add_argument("--test-seed", type=int, default=3303)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare_splits(
            args.config, args.out_dir,
            sizes=(args.train_paths, args.validation_paths, args.test_paths),
            seeds=(args.train_seed, args.validation_seed, args.test_seed),
        )
        print("Prepared Q splits:", {s: v["n_paths"] for s, v in manifest["splits"].items()})
        return
    if args.command == "calibrate":
        pool, cfg, _ = load_split(args.out_dir, "validation", args.config)
        candidates = calibrate_reward_candidates(pool, cfg)
        target = Path(args.out_dir) / "validation_reward_candidates.json"
        target.write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(candidates, indent=2))
        return
    pool, cfg, meta = load_split(args.out_dir, args.split, args.config)
    results = report_baselines(pool, cfg, args.out_dir, split=args.split,
                               config_sha256=meta["config_sha256"])
    for r in results:
        print(f"{r['strategy']:20s} mean={r['mean_pnl']:10.3f} "
              f"q95={r['q95_loss']:10.3f} CVaR95={r['cvar95_loss']:10.3f} "
              f"cost={r['mean_transaction_cost']:9.3f}")


if __name__ == "__main__":
    main()
