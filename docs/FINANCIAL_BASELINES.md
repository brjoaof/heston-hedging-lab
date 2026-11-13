# Financial baselines on the frozen Heston-COS campaign

This module evaluates **financial hedging policies**, not the trained SAC
weights. It intentionally reuses the exact `simulate_forward_hedge` execution
engine used by the SAC environment, including forward hedge buckets, bounded
position/increment, minimum trade, half spreads, fixed fees, financing and
mandatory closing at expiry.

## Strategy definitions

| Strategy | Target exposure | Rebalance frequency |
|---|---|---|
| `no_hedge` | zero | never |
| `hold_initial` | frozen COS Heston Delta position from inception | never (except mandatory maturity close) |
| `heston_daily` | negative Heston spot Delta of the short-option bucket, converted to forward units | every trading day |
| `heston_periodic_21` | same Heston Delta target | every 21 trading days, starting at day zero |
| `heston_band_0p5` | Heston Delta with fixed no-transaction band 0.5 | each day if outside band |
| `black20_daily` | negative Black-Scholes Delta at **constant 20% volatility** | daily |
| `black20_periodic_21` | same fixed-volatility Delta | every 21 trading days |
| `heston_mv_daily` | Heston **minimum-variance** Delta | daily |
| `heston_mv_periodic_21` | same adjusted Delta | every 21 trading days |
| `black_iv_daily` | Black Delta using Heston-COS price-implied volatility (optional) | daily |
| `black_iv_periodic_21` | same IV-based Delta (optional) | every 21 trading days |

The usual Heston Delta is `∂C_H/∂S` holding current variance fixed. Under the
stated Heston SDEs, the instantaneous variance-minimizing hedge Delta in units
of the underlying is `∂C_H/∂S + ρ ξ/S ∂C_H/∂v`. The adjustment depends on the
**current variance sensitivity** (not Black implied-volatility vega). This
baseline is not generally optimal for the dissertation's Q95/CVaR objective
under transaction costs.

For a liability of short calls, the *signed* book sensitivities are negative.
Positions in forwards of maturity T are therefore the clipped quantity
`-Delta_book / exp((r-q)*(T-t))`. Each strategy's **opening positions are its
own** (not forcibly the Heston Delta positions), and opening costs are charged
on precisely those holdings. Later changes obey the same max-trade/min-trade
and costs as SAC. Trades are decided with current-day information only.

The COS Heston prices and Greeks saved in the immutable split remain the P&L
marking reference, **even when a Black Delta strategy hedges those options**.
Black is not used to reprice the liabilities. `black_iv` inverts each saved
Heston call price for a constant-volatility input; its Delta holds that IV
fixed at the decision state. This is distinct from Heston's total spot Delta;
near expiry/zero vega the inversion is ill-conditioned, so it is optional and
should be explicitly audited before interpretation.

## Validation only: first small operational run

```bash
python scripts/evaluate_financial_baselines.py \
  --manifest configs/expert_schedule_cos.json \
  --regimes 0.00 0.20 0.40 \
  --max-paths 32 \
  --out-dir results/diagnostics/financial_baselines_smoke
```

This **reads only validation** splits, validates SHA-256 and never changes the
training configuration or saved RL policies. The small subset is not a
financial performance experiment.

## Full validation baseline comparison

```bash
python scripts/evaluate_financial_baselines.py \
  --manifest configs/expert_schedule_cos.json \
  --out-dir results/financial_baselines_cos_seed1_validation
```

Writes a consolidated `validation_summary.csv`, per-regime `validation_xi_..._per_path.npz`
(for paired path-level analysis), and provenance JSON containing the frozen
config/data hashes. It refuses to overwrite existing outputs. Black-IV is
intentionally excluded by default; to evaluate it, pass `--strategies
black_iv_daily black_iv_periodic_21` in a **new output directory**.

To add Black-IV baselines to existing results, write to a new output directory,
not an existing result. Do not merge separate summary files without documenting
the exact same split, N and dataset SHA.

## Interpretation and test discipline

The historical SAC best-checkpoint selection used the **same full validation
split** (5,000 paths per regime). We may compare validation Q95/CVaR and costs,
but this is descriptive model-selection evidence, **not independent test evidence**.
Freeze the strategies/parameters using validation alone; evaluate the held-out
20,000-path test split **once** later using `--split test --allow-test`.

Q-Heston pricing and dynamics define risk-neutral scenarios; Q risk metrics
are not automatically physical-measure (P) risk metrics. Periodic-21 is a
predefined ENIAC comparator, **not** a period optimized on the validation set.
Do not add multiple rebalance frequencies or tune widths on the held-out test.

The existing code-version fingerprint covers the live financial/RL engine;
these new **independent baseline files** are not included, and adding them
alone does not invalidate SAC model checkpoints.
