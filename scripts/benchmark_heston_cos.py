#!/usr/bin/env python3
"""Reproducible COS vs existing Fourier/Heston benchmark, no RL side effects.

Examples:
 python scripts/benchmark_heston_cos.py --n-states 32 --terms 128 256 512 \
   --xis 0 0.2 0.4 --maturities .004 .168 .504 \
   --csv results/diagnostics/cos_regular.csv
 python scripts/benchmark_heston_cos.py --scenario boundary --n-states 8 \
   --xis .25 .4 --maturities .004 .04 --terms 256 512 \
   --csv results/diagnostics/cos_boundary.csv
 python scripts/benchmark_heston_cos.py --npz-file \
   results/forward_experiments/.../xi_0p20/splits/validation.npz \
   --xis .2 --maturities .004 .168 --n-states 32

The reference Fourier implementation can itself fail or be inaccurate near
v=0. The CSV records disagreement and ref failures rather than conflating
agreement with accuracy. Run after expensive dataset preparation for fair timing.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heston import HestonParams, MarketDynamics
from pricing import heston_greeks_strikes_batch, _adaptive_heston_greeks_near_boundary
from pricing_cos import heston_cos_greeks_strikes_batch


FIELDS = ('price','delta','gamma','variance_sensitivity')


def make_states(args):
    rng=np.random.default_rng(args.seed)
    n=args.n_states
    if args.npz_file:
        with np.load(args.npz_file,allow_pickle=False) as data:
            spot=data['spot']
            variance=data['variance']
            if spot.shape != variance.shape or spot.ndim != 2:
                raise ValueError('npz requires matching 2-D spot/variance arrays')
            # Randomly pick path and day from saved Q-Heston paths.
            rows=rng.integers(0,spot.shape[0],size=n)
            days=rng.integers(0,spot.shape[1],size=n)
            return np.asarray(spot[rows,days]),np.asarray(variance[rows,days])
    spot=rng.uniform(90.,110.,size=n)
    if args.scenario=='regular':
        variance=rng.uniform(.008,.12,size=n)
    else:
        variance=rng.choice([0.,1e-5,1e-4,.0003,.001,.003],size=n)
        spot[0]=106.24767
        variance[0]=.0001084851
        if n>=2:
            spot[1]=94.258753
            variance[1]=0.
    return spot,variance


def elapsed(fn, repeats):
    times=[]
    result=None
    for i in range(repeats):
        start=time.perf_counter()
        result=fn()
        times.append(time.perf_counter()-start)
    return result,float(np.median(times))


def summarize(args):
    states=make_states(args)
    market=MarketDynamics(measure='Q',spot0=100.,risk_free_rate=math.log(1.1))
    strikes=np.array([96.,100.,104.])
    result_rows=[]
    for xi in args.xis:
        params=HestonParams(kappa=1.,theta=.04,xi=float(xi),rho=.7,v0=.04)
        for maturity in args.maturities:
            base=dict(params=params,market=market,strikes=strikes,maturity=float(maturity),
                      spots=states[0],variances=states[1])
            reference=None
            ref_elapsed=float('nan')
            ref_error=''
            try:
                reference,ref_elapsed=elapsed(lambda:heston_greeks_strikes_batch(**base),args.repeats)
            except (FloatingPointError,ValueError,OverflowError) as exc:
                ref_error=f'{type(exc).__name__}: {str(exc)}'[:240]
            for terms in args.terms:
                row=dict(
                    scenario='saved_paths' if args.npz_file else args.scenario,
                    n_states=args.n_states, seed=args.seed,
                    xi=xi, maturity=maturity, terms=terms, L=args.L,
                    reference_time_s=ref_elapsed,
                    ref_error=ref_error,
                )
                try:
                    cos,sec=elapsed(lambda:heston_cos_greeks_strikes_batch(
                        **base,n_terms=terms,L=args.L),args.repeats)
                    row['cos_time_s']=sec
                    row['reference_speedup']=ref_elapsed/sec if np.isfinite(ref_elapsed) else float('nan')
                    if reference is not None:
                        for field in FIELDS:
                            diff=np.abs(getattr(cos,field)-getattr(reference,field))
                            row[f'max_abs_{field}']=float(np.max(diff))
                            row[f'p95_abs_{field}']=float(np.quantile(diff,.95))
                    if terms==max(args.terms):
                        refined=heston_cos_greeks_strikes_batch(
                            **base,n_terms=terms*2,L=args.L)
                        wide=heston_cos_greeks_strikes_batch(
                            **base,n_terms=terms*2,L=args.L*1.5)
                        for field in FIELDS:
                            row[f'max_n_convergence_{field}']=float(np.max(
                                np.abs(getattr(cos,field)-getattr(refined,field))))
                            row[f'max_L_convergence_{field}']=float(np.max(
                                np.abs(getattr(refined,field)-getattr(wide,field))))
                    # Optional costly triangulation at the state where the
                    # two vectorized pricers disagree the most. This uses an
                    # independent damped Fourier inversion, not the batch
                    # Fourier code that shares its approximation errors.
                    if args.audit_worst and terms==max(args.terms) and float(xi)>0:
                        if reference is not None:
                            worst=np.unravel_index(
                                np.argmax(np.abs(cos.price-reference.price)),
                                cos.price.shape)
                            path_idx,strike_idx=worst
                        else:
                            path_idx,strike_idx=0,1
                        row['audit_spot']=float(states[0][path_idx])
                        row['audit_variance']=float(states[1][path_idx])
                        row['audit_strike']=float(strikes[strike_idx])
                        try:
                            audited=_adaptive_heston_greeks_near_boundary(
                                params,market,spot=row['audit_spot'],
                                variance=row['audit_variance'],
                                strike=row['audit_strike'],maturity=float(maturity))
                            for field in FIELDS:
                                row[f'audit_{field}_damped']=float(getattr(audited,field))
                                row[f'audit_{field}_cos']=float(getattr(cos,field)[worst])
                                if reference is not None:
                                    row[f'audit_{field}_fourier']=float(getattr(reference,field)[worst])
                        except (FloatingPointError,ValueError,OverflowError) as exc:
                            row['audit_error']=f'{type(exc).__name__}: {exc}'[:240]
                    row['status']='ok'
                except (FloatingPointError,ValueError,OverflowError) as exc:
                    row['status']=f'{type(exc).__name__}: {str(exc)}'[:240]
                result_rows.append(row)
                print(f'xi={xi:.2f} T={maturity:.4f} N={terms:4d} '
                      f'COS={row.get("cos_time_s",float("nan")):.4f}s '
                      f'Fourier={ref_elapsed:.4f}s '
                      f'max|dPrice|={row.get("max_abs_price",float("nan")):.3g} '
                      f'{row["status"]}',flush=True)
    return result_rows


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n-states',type=int,default=32)
    ap.add_argument('--seed',type=int,default=2026)
    ap.add_argument('--scenario',choices=('regular','boundary'),default='regular')
    ap.add_argument('--npz-file',type=Path,help='Optional existing saved train/validation NPZ; read-only')
    ap.add_argument('--terms',type=int,nargs='+',default=[128,256,512])
    ap.add_argument('--L',type=float,default=24.)
    ap.add_argument('--xis',type=float,nargs='+',default=[0.,.05,.2,.4])
    ap.add_argument('--maturities',type=float,nargs='+',default=[.004,.04,.168,.504])
    ap.add_argument('--repeats',type=int,default=1)
    ap.add_argument('--strict',action='store_true',help='Exit nonzero if any COS configuration fails numerically')
    ap.add_argument('--audit-worst',action='store_true',help='Damped Fourier triangulation for one maximum-disagreement state per case (slow)')
    ap.add_argument('--csv',type=Path,default=Path('results/diagnostics/heston_cos_benchmark.csv'))
    args=ap.parse_args()
    if args.npz_file and args.npz_file.name == 'test.npz':
        ap.error('Refuse to use held-out test.npz for method development; use train or validation')
    if args.n_states<1 or args.repeats<1 or any(n<16 for n in args.terms):
        ap.error('n-states, repeats must be positive and terms >=16')
    if any(x<0 or x>.4 for x in args.xis) or any(t<=0 for t in args.maturities):
        ap.error('xi must lie in [0,0.4], maturities must be positive')
    rows=summarize(args)
    args.csv.parent.mkdir(parents=True,exist_ok=True)
    headers=list(dict.fromkeys(k for row in rows for k in row))
    with args.csv.open('w',newline='',encoding='utf-8') as output:
        writer=csv.DictWriter(output,fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    config=args.csv.with_suffix('.json')
    config.write_text(json.dumps({
        'n_states':args.n_states,'seed':args.seed,'scenario':args.scenario,
        'npz_file':str(args.npz_file) if args.npz_file else None,
        'xis':args.xis,'maturities':args.maturities,'terms':args.terms,
        'L':args.L,'repeats':args.repeats,'audit_worst':args.audit_worst,'reference':'pricing.heston_greeks_strikes_batch',
        'note':'Fourier is a comparison, not validated ground truth at boundary states'
    },indent=2)+'\n',encoding='utf-8')
    print(f'Saved {args.csv} and {config}')
    failed=sum(row['status']!='ok' for row in rows)
    if failed:
        print(f'WARNING: {failed} COS configurations failed numerical checks; see CSV.')
        if args.strict:
            raise SystemExit(2)


if __name__=='__main__':
    main()
