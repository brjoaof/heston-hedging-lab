#!/usr/bin/env python3
"""Run comparable financial baselines on frozen Q-Heston validation paths.

'--split test' requires --allow-test and should be used only after freezing all
methods, conventions and choices. Runs no RL training; never changes datasets.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from financial_baselines import DEFAULT_STRATEGIES, STRATEGIES, evaluate_financial_baseline
from forward_experiments import load_split, _read_config
from hedging import summarize_hedge
from heston import HestonParams
from scripts.expert_queue import load_schedule, config_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',default='configs/expert_schedule_cos.json')
    parser.add_argument('--split',choices=('validation','test'),default='validation')
    parser.add_argument('--allow-test',action='store_true')
    parser.add_argument('--regimes',nargs='+',help='Subset of regimes, e.g. 0.00 0.20 0.40')
    parser.add_argument('--strategies',nargs='+',choices=STRATEGIES,default=DEFAULT_STRATEGIES)
    parser.add_argument('--max-paths',type=int,help='First N paths for infrastructure smoke only')
    parser.add_argument('--out-dir',default='results/financial_baselines_cos')
    args = parser.parse_args()
    if args.split == 'test' and not args.allow_test:
        parser.error('Held-out test requires explicit --allow-test after protocol is frozen')
    schedule = load_schedule(Path(args.manifest))
    regimes = schedule['regimes'] if args.regimes is None else args.regimes
    if len(set(regimes))!=len(regimes) or any(x not in schedule['regimes'] for x in regimes):
        parser.error('Invalid or duplicate regimes; use the manifest strings, e.g. 0.20')
    if len(set(args.strategies))!=len(args.strategies):
        parser.error('Duplicate strategy')
    if args.max_paths is not None and args.max_paths <= 0:
        parser.error('--max-paths must be positive')
    out_dir = Path(args.out_dir)
    # Fail before any expensive COS evaluation if the destination is occupied.
    summary_file=out_dir/f'{args.split}_summary.csv'
    intended=[out_dir / f'{args.split}_xi_{xi.replace(".","p")}_per_path.npz'
              for xi in regimes]
    intended += [out_dir / f'{args.split}_xi_{xi.replace(".","p")}_provenance.json'
                 for xi in regimes]
    intended.append(summary_file)
    occupied=[p for p in intended if p.exists()]
    if occupied:
        raise FileExistsError(f'Refusing to overwrite existing reports: {occupied[:3]}')
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for xi in regimes:
        config, splits = config_paths(schedule, xi)
        pool, env_cfg, meta = load_split(splits,args.split,config)
        financial_cfg, config_sha = _read_config(config)
        pc = financial_cfg.get('pricing',{})
        if pc.get('method')!='cos':
            raise RuntimeError(f'COS campaign expected: {config}')
        n_available = len(pool.paths.spot)
        n = n_available if args.max_paths is None else min(args.max_paths,n_available)
        if n != n_available:
            from rl_env import ForwardEpisodePool
            from heston import HestonPaths
            from hedging import BookPaths
            paths,b = pool.paths,pool.book_paths
            reduced_paths=HestonPaths(paths.time_years,paths.spot[:n],paths.variance[:n],
                                      paths.raw_variance[:n],paths.measure)
            reduced_book=BookPaths(b.value[:n],b.delta[:n],b.gamma[:n],
                                  b.variance_sensitivity[:n],b.settlements[:n],
                                  b.contract_prices[:n] if b.contract_prices is not None else None,
                                  b.bucket_delta[:n] if b.bucket_delta is not None else None)
            pool=ForwardEpisodePool(reduced_paths,reduced_book,pool.initial_positions[:n],pool.market_q)
        params=HestonParams(**financial_cfg['heston'])
        cache={}
        records={}
        for strategy in args.strategies:
            result=evaluate_financial_baseline(pool,env_cfg,strategy,heston=params,
                                                cos_n_terms=pc['n_terms'],cos_L=pc['L'],
                                                cached_deltas=cache)
            summary=summarize_hedge(result)
            all_rows.append({'xi':xi,'strategy':strategy,'n_paths':n,'split':args.split,
                             **summary})
            records[strategy+'_terminal_pnl']=result.terminal_pnl
            records[strategy+'_cost']=result.transaction_cost
            records[strategy+'_trades']=result.trades
            records[strategy+'_turnover']=result.turnover
            print(f"xi={xi} {strategy:23s} Q95={summary['q95_loss']:.4f} "
                  f"CVaR95={summary['cvar95_loss']:.4f} cost={summary['mean_transaction_cost']:.4f}",
                  flush=True)
        tag='xi_'+xi.replace('.','p')
        file=out_dir / f'{args.split}_{tag}_per_path.npz'
        if file.exists():
            raise FileExistsError(f'Refusing to overwrite: {file}')
        np.savez_compressed(file,**records)
        data = {
            'split':args.split,'xi':xi,'n_paths':n,'source_manifest':str(args.manifest),
            'config_sha256':config_sha,'data_sha256':meta['data_sha256'],
            'pricing':pc,'strategies':list(args.strategies),
            'note':'Q validation only; test is held out. Same execution/costs as SAC; '
                   'not independent market-measure P performance.',
        }
        (out_dir/f'{args.split}_{tag}_provenance.json').write_text(json.dumps(data,indent=2)+'\n')
    with summary_file.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print('[OK]',summary_file,'rows=',len(all_rows),flush=True)

if __name__=='__main__':
    main()
