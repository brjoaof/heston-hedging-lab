"""Small pretraining controls for Heston-Q SAC specialists.

Commands
--------
control: evaluate a constant-width band through the *same* ForwardBandEpisode
         as SAC-Band, optionally compare paired paths with its saved evaluation.
prepare: generate separate immutable configuration/splits for Heston xi regimes.

No reinforcement learning occurs here. No test split is opened for model choice.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path

import numpy as np

from band_policy import (BAND_EXECUTION, ForwardBandEpisode,
                         require_current_band_execution)
from forward_experiments import _read_config, load_split, prepare_splits
from heston import HestonParams
from train_sac_trade import evaluate_actions, experiment_config


def _save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def regime_label(xi: float | str) -> str:
    """Unambiguous directory label for the 0.01-resolution exploratory grid."""
    try:
        value = Decimal(str(xi))
    except InvalidOperation as exc:
        raise ValueError("xi must be a finite nonnegative decimal") from exc
    if not value.is_finite() or value < 0 or value != value.quantize(Decimal("0.01")):
        raise ValueError("xi must be finite, nonnegative and have at most two decimals")
    return "xi_" + f"{value:.2f}".replace(".", "p")


def regime_config(base: dict, xi: float | str) -> dict:
    """Only update the forward Heston vol-of-vol; leave legacy blocks intact."""
    cfg = deepcopy(base)
    if cfg["forward_experiment"]["measure"] != "Q":
        raise ValueError("Regime experiments require the Q forward_experiment")
    regime_label(xi)
    cfg["forward_experiment"]["heston"]["xi"] = float(xi)
    HestonParams(**cfg["forward_experiment"]["heston"])
    return cfg


def prepare_regimes(base_config: str | Path, out_root: str | Path, *,
                    xis: tuple[float, ...], sizes: tuple[int, int, int],
                    seeds: tuple[int, int, int]) -> list[dict]:
    """One distinct, fingerprinted dataset per regime; fail on existing outputs."""
    if not xis:
        raise ValueError("At least one xi is required")
    labels = [regime_label(xi) for xi in xis]
    if len(set(labels)) != len(labels):
        raise ValueError("Repeated xi regime")
    if len(sizes) != 3 or any(type(x) is not int or x < 1 for x in sizes):
        raise ValueError("Train, validation and test sizes must all be positive integers")
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(x) is not int or x < 0 for x in seeds):
        raise ValueError("Split seeds must be three distinct nonnegative integers")
    base_path = Path(base_config)
    _, _ = _read_config(base_path)
    base = json.loads(base_path.read_text(encoding="utf-8"))
    output = Path(out_root)
    for label in labels:
        if (output / label).exists():
            raise FileExistsError(f"Regime output already exists: {output / label}")
    prepared: list[dict] = []
    for xi, label in zip(xis, labels):
        destination = output / label
        destination.mkdir(parents=True, exist_ok=False)
        config_file = destination / "config.json"
        _save_json(config_file, regime_config(base, xi))
        splits = destination / "splits"
        manifest = prepare_splits(config_file, splits, sizes=sizes, seeds=seeds)
        _, fingerprint = _read_config(config_file)
        item = {
            "xi": float(xi), "config": str(config_file),
            "splits_dir": str(splits),
            "financial_config_sha256": fingerprint,
            "sizes": {key: record["n_paths"] for key, record in manifest["splits"].items()},
            "seeds": {key: record["seed"] for key, record in manifest["splits"].items()},
        }
        _save_json(destination / "regime.json", item)
        prepared.append(item)
    return prepared


def fixed_band_control(pool, base_config, *, width: float = 0.75,
                       max_band_width: float = 1.5, penalty: float = 2.0):
    """Roll out a *constant* band using identical SAC-Band trading/accounting.

    It is intentionally not a separately coded baseline with subtly different
    rounding, min trade, expiry handling or initial hedge costs.
    """
    if not math.isfinite(max_band_width) or not 0 < max_band_width <= base_config.max_position:
        raise ValueError("max_band_width outside (0, max_position]")
    if not math.isfinite(width) or not 0 <= width <= max_band_width:
        raise ValueError("constant band width outside [0, max_band_width]")
    cfg = experiment_config(base_config, penalty=penalty,
                            max_trade_increment=base_config.max_position,
                            include_bucket_deltas=True)
    constant = np.full(len(cfg.expiries_days), width, dtype=float)
    return evaluate_actions(
        pool, cfg, lambda _observation: constant.copy(),
        episode_factory=lambda p, c: ForwardBandEpisode(p, c, max_band_width=max_band_width),
        action_bounds=(0.0, max_band_width),
    )


def paired_comparison(control: dict[str, np.ndarray], learned: dict[str, np.ndarray]) -> dict:
    """Same-path differences; no significance claim from a 32-path pilot."""
    fields = ("terminal_pnl", "transaction_cost", "trades", "turnover")
    for field in fields:
        a, b = np.asarray(control[field]), np.asarray(learned[field])
        if a.ndim != 1 or a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            raise ValueError(f"Paired {field} must have matching finite 1D trajectories")
    d = learned["terminal_pnl"] - control["terminal_pnl"]
    return {
        "n_paths": int(len(d)),
        "mean_pnl_advantage_learned": float(np.mean(d)),
        "median_pnl_advantage_learned": float(np.median(d)),
        "learned_better_paths": int(np.count_nonzero(d > 0)),
        "equal_pnl_paths": int(np.count_nonzero(np.isclose(d, 0, atol=1e-10))),
        "mean_cost_difference_learned_minus_constant": float(np.mean(
            learned["transaction_cost"] - control["transaction_cost"])),
        "mean_trade_difference_learned_minus_constant": float(np.mean(
            learned["trades"] - control["trades"])),
        "warning": "Exploratory same-path comparison, not a test of superiority."
    }


def run_control(*, config: str | Path, splits_dir: str | Path, out_dir: str | Path,
                width: float = 0.75, max_band_width: float = 1.5,
                compare_evaluation: str | Path | None = None) -> dict:
    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output is not empty; use a fresh directory")
    pool, base, meta = load_split(splits_dir, "validation", config)
    # Fixed c=2 for comparability with existing SAC-Band; no reward optimization.
    summary, traces = fixed_band_control(pool, base, width=width,
                                         max_band_width=max_band_width, penalty=2.0)
    record = {
        "method": "constant_no_transaction_band", "measure": "Q", "split": "validation",
        "band_execution": BAND_EXECUTION,
        "regime_xi": _read_config(config)[0]["heston"]["xi"],
        "width_per_bucket": [width] * len(base.expiries_days),
        "max_band_width": max_band_width, "reward_penalty": 2.0,
        "financial_config_sha256": meta["config_sha256"],
        "validation_data_sha256": meta["data_sha256"],
        "metrics": summary,
        "caution": "Finite-sample validation diagnostics; do not tune width on the held-out test.",
    }
    if compare_evaluation is not None:
        eval_npz = Path(compare_evaluation)
        eval_json = eval_npz.with_suffix(".json")
        run_json = eval_npz.parent / "run.json"
        if not all(f.exists() for f in (eval_npz, eval_json, run_json)):
            raise FileNotFoundError("Expected evaluation .npz/.json and run.json in the same run directory")
        learned_meta = json.loads(eval_json.read_text(encoding="utf-8"))
        run_meta = json.loads(run_json.read_text(encoding="utf-8"))
        require_current_band_execution(run_meta)
        require_current_band_execution(learned_meta)
        if (learned_meta.get("split") != "validation" or
                learned_meta.get("action_mode") != "band" or
                run_meta.get("action_mode") != "band" or
                learned_meta.get("financial_config_sha256") != meta["config_sha256"] or
                run_meta.get("validation_data_sha256") != meta["data_sha256"] or
                run_meta.get("financial_config_sha256") != meta["config_sha256"] or
                run_meta.get("penalty") != 2.0 or
                run_meta.get("max_band_width") != max_band_width):
            raise ValueError("Compared SAC-Band model must match Q validation, regime, c=2 and action range")
        with np.load(eval_npz, allow_pickle=False) as f:
            learned = {name: f[name] for name in
                       ("terminal_pnl", "transaction_cost", "trades", "turnover")}
        record["compared_model"] = learned_meta.get("model")
        record["learned_metrics"] = learned_meta.get("metrics")
        record["paired_comparison"] = paired_comparison(traces, learned)
    output.mkdir(parents=True, exist_ok=True)
    _save_json(output / "validation_constant_band.json", record)
    np.savez_compressed(output / "validation_constant_band.npz", **traces)
    print("[control] Saved:", (output / "validation_constant_band.json").resolve())
    print("[control] Saved:", (output / "validation_constant_band.npz").resolve())
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    ctl = sub.add_parser("control", help="constant band on existing validation split")
    ctl.add_argument("--config", default="config.json")
    ctl.add_argument("--splits-dir", default="results/forward_experiments/pilot_xi_0p20")
    ctl.add_argument("--out-dir", required=True)
    ctl.add_argument("--width", type=float, default=0.75)
    ctl.add_argument("--max-band-width", type=float, default=1.5)
    ctl.add_argument("--compare-evaluation", help="SAC-Band validation_best_evaluation.npz")
    prep = sub.add_parser("prepare", help="immutable configs and Q train/validation/test splits")
    prep.add_argument("--base-config", default="config.json")
    prep.add_argument("--out-root", default="results/forward_experiments/regime_pilots")
    prep.add_argument("--xi", action="append", type=float, required=True,
                      help="Repeat: --xi 0 --xi 0.20 --xi 0.40")
    prep.add_argument("--train-paths", type=int, default=64)
    prep.add_argument("--validation-paths", type=int, default=128)
    prep.add_argument("--test-paths", type=int, default=32)
    args = p.parse_args()
    if args.command == "control":
        result = run_control(config=args.config, splits_dir=args.splits_dir,
                             out_dir=args.out_dir, width=args.width,
                             max_band_width=args.max_band_width,
                             compare_evaluation=args.compare_evaluation)
        print(json.dumps({"metrics": result["metrics"],
                          "paired_comparison": result.get("paired_comparison")}, indent=2))
    else:
        records = prepare_regimes(
            args.base_config, args.out_root, xis=tuple(args.xi),
            sizes=(args.train_paths, args.validation_paths, args.test_paths),
            seeds=(1101, 2202, 3303),
        )
        for record in records:
            print(f"[OK] {record['xi']:.2f}: config={record['config']} splits={record['splits_dir']}")


if __name__ == "__main__":
    main()
