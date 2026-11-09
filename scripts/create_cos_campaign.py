#!/usr/bin/env python3
"""Create isolated, frozen COS Heston config/schedule; do not edit ongoing runs.

Run on the *server* in the repository root, then inspect generated files before
running a short smoke. This script does not generate paths or train SAC.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def create_campaign(*, base_config: Path, base_schedule: Path, out_config: Path,
                    out_schedule: Path, data_root: str, result_root: str,
                    n_terms: int = 512, L: float = 24.0) -> None:
    if type(n_terms) is not int or n_terms < 16 or not math.isfinite(L) or L <= 0:
        raise ValueError('Invalid COS N/L')
    if out_config.resolve() == base_config.resolve() or out_schedule.resolve() == base_schedule.resolve():
        raise ValueError('Cannot overwrite original financial configuration or schedule')
    if out_config.exists() or out_schedule.exists():
        raise FileExistsError('Campaign config or schedule already exists; refuse overwrite')
    config = json.loads(base_config.read_text(encoding='utf-8'))
    schedule = json.loads(base_schedule.read_text(encoding='utf-8'))
    if 'forward_experiment' not in config:
        raise ValueError('Missing forward_experiment financial configuration')
    # The settings are part of the immutable dataset config fingerprint.
    config['forward_experiment']['pricing'] = {
        'method': 'cos', 'n_terms': n_terms, 'L': L,
    }
    schedule['base_config'] = str(out_config.resolve().relative_to(ROOT.resolve())) if out_config.resolve().is_relative_to(ROOT.resolve()) else str(out_config.resolve())
    schedule['data_root'] = data_root
    schedule['result_root'] = result_root
    # Do not overwrite historical datasets or checkpoints.
    def full(value: str) -> Path:
        p = Path(value)
        return (p if p.is_absolute() else ROOT / p).resolve()
    if full(data_root).exists() or full(result_root).exists():
        raise FileExistsError('Dataset or checkpoint root already exists; choose fresh directories')
    if not full(data_root).is_relative_to((ROOT/'results').resolve()):
        raise ValueError('data_root must be inside results/')
    if not full(result_root).is_relative_to((ROOT/'results').resolve()):
        raise ValueError('result_root must be inside results/')
    out_config.parent.mkdir(parents=True, exist_ok=True)
    out_schedule.parent.mkdir(parents=True, exist_ok=True)
    out_config.write_text(json.dumps(config, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    out_schedule.write_text(json.dumps(schedule, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'[OK] COS financial configuration: {out_config}')
    print(f'[OK] Independent COS campaign manifest: {out_schedule}')
    print(f'[OK] COS N={n_terms} L={L}, data={data_root}, results={result_root}')
    print('No data generated and no RL job started.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config', type=Path, default=ROOT/'config.json')
    p.add_argument('--base-schedule', type=Path, default=ROOT/'configs/expert_schedule.json')
    p.add_argument('--out-config', type=Path, default=ROOT/'configs/heston_cos_campaign.json')
    p.add_argument('--out-schedule', type=Path, default=ROOT/'configs/expert_schedule_cos.json')
    p.add_argument('--data-root', default='results/forward_experiments/scale_regimes_heston_cos_N512_L24')
    p.add_argument('--result-root', default='results/cos_experts')
    p.add_argument('--n-terms', type=int, default=512)
    p.add_argument('--L', type=float, default=24.0)
    a = p.parse_args()
    create_campaign(base_config=a.base_config, base_schedule=a.base_schedule,
                    out_config=a.out_config, out_schedule=a.out_schedule,
                    data_root=a.data_root, result_root=a.result_root,
                    n_terms=a.n_terms, L=a.L)


if __name__ == '__main__':
    main()
