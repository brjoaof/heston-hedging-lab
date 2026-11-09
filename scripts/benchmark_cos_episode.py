#!/usr/bin/env python3
"""End-to-end Heston-Q episode benchmark: same paths, COS vs Fourier marks.

No saved train/validation/test split is read or changed. The output is a small
JSON report for controlled numerical evaluation before committing to an oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rl_env import prepare_forward_episodes

FIELDS = ('value', 'delta', 'gamma', 'variance_sensitivity', 'contract_prices', 'bucket_delta')


def compare_episode(base: dict, *, xi: float, n_paths: int, seed: int,
                    cos_n_terms: int, cos_L: float) -> dict:
    config = json.loads(json.dumps(base))
    config['forward_experiment']['heston']['xi'] = xi
    row = {'xi': xi, 'n_paths': n_paths, 'seed': seed,
           'cos_n_terms': cos_n_terms, 'cos_L': cos_L}
    pools = {}
    with tempfile.TemporaryDirectory() as work:
        for method in ('fourier', 'cos'):
            config['forward_experiment']['pricing'] = (
                {'method': 'fourier'} if method == 'fourier' else
                {'method': 'cos', 'n_terms': cos_n_terms, 'L': cos_L})
            path = Path(work)/f'{method}.json'
            path.write_text(json.dumps(config), encoding='utf-8')
            start = time.perf_counter()
            try:
                pools[method], _ = prepare_forward_episodes(path,n_paths=n_paths,seed=seed)
                row[f'{method}_time_s'] = time.perf_counter()-start
            except (FloatingPointError, ValueError, OverflowError) as exc:
                row[f'{method}_error'] = f'{type(exc).__name__}: {exc}'
                return row
    a,b = pools['fourier'],pools['cos']
    # Ensure underlying paths were not changed by pricing choice.
    assert np.array_equal(a.paths.spot,b.paths.spot)
    assert np.array_equal(a.paths.variance,b.paths.variance)
    row['speedup_fourier_over_cos'] = row['fourier_time_s']/row['cos_time_s']
    for name in FIELDS:
        x = getattr(a.book_paths,name)
        y = getattr(b.book_paths,name)
        difference = np.abs(x-y)
        index = np.unravel_index(np.argmax(difference),difference.shape)
        row[name] = {
            'max_abs': float(np.max(difference)),
            'p95_abs': float(np.quantile(difference,.95)),
            'worst_index': list(map(int,index)),
            'fourier_at_worst': float(x[index]),
            'cos_at_worst': float(y[index]),
        }
    positions = np.abs(a.initial_positions - b.initial_positions)
    row['max_abs_initial_positions'] = float(positions.max())
    row['status'] = 'ok'
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config',type=Path,default=ROOT/'config.json')
    p.add_argument('--xis',type=float,nargs='+',default=[0.,.2,.4])
    p.add_argument('--n-paths',type=int,default=16)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--terms',type=int,default=512)
    p.add_argument('--L',type=float,default=24.)
    p.add_argument('--output',type=Path,default=ROOT/'results/diagnostics/cos_episode_screen.json')
    args = p.parse_args()
    if args.n_paths<1 or any(x<0 or x>.4 for x in args.xis):
        p.error('n-paths must be positive and xi must be between 0 and 0.40')
    base=json.loads(args.base_config.read_text(encoding='utf-8'))
    rows=[]
    for xi in args.xis:
        row=compare_episode(base,xi=xi,n_paths=args.n_paths,seed=args.seed,
                            cos_n_terms=args.terms,cos_L=args.L)
        rows.append(row)
        print(f'xi={xi:.2f} COS={row.get("cos_time_s",float("nan")):.2f}s '
              f'Fourier={row.get("fourier_time_s",float("nan")):.2f}s '
              f'max|dOptionValue|={row.get("value",{}).get("max_abs",float("nan")):.4g} '
              f'{row.get("status",row.get("cos_error",row.get("fourier_error","failed")))}',
              flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'parameters': vars(args) | {'base_config':str(args.base_config),'output':str(args.output)},
                                        'rows': rows},indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('[OK] Report:',args.output)


if __name__ == '__main__':
    main()
