# COS (Fang--Oosterlee, 2008): independent Heston-Q numerical experiment

**Status:** experimental pricer + reproducible benchmark, **not** the SAC
pricing backend. This change only adds files; it does not edit `pricing.py`,
`hedging.py`, the queue manifest, saved datasets, or trainer/checkpoint logic.
Do **not** change a running Heston preparation or 50M-step SAC run to COS.

## Reference

F. Fang and C. W. Oosterlee (2008). *A Novel Pricing Method for European
Options Based on Fourier-Cosine Series Expansions*. **SIAM Journal on Scientific
Computing** 31(2), 826--848. https://doi.org/10.1137/080718061

COS approximates the conditional log-return density over [a,b] by its cosine
series with coefficients obtained from the Heston characteristic function.
See paper Eqs. (8)--(11), the analytic payoff coefficients in Sec. 3 and Heston
applications in Sec. 5.

This implementation uses `X = log(S_T/F_t)` with
`F_t=S_t exp((r-q)T)`. The characteristic function is independent of the strike
for a fixed market state/maturity, allowing reuse for K=96,100,104. We integrate
the truncated cosine density against 1 and `exp(X)` analytically to obtain
OTM call/put price and Delta; Gamma is recovered from the density at
`log(K/F_t)`, and `dPrice/dv` from the derivative of the affine characteristic
function with respect to *current instantaneous variance*. ITM options are
recovered by put-call parity. Black with constant integrated variance is an
**independent analytic benchmark only when xi=0**.

**Truncation differs from the most common paper convention.** We use exact
conditional first and second cumulants of `X` to set
`[a,b]=[c1-L*sqrt(c2), c1+L*sqrt(c2)]` rather than a c4-based range. Neither
`N` (number of cosine terms) nor `L` (range width) comes with a blanket error
guarantee, especially for extremely low variance, short expiry or a violated
Feller condition. The reported accuracy must be established by varying **both**.
Defaults `N=512,L=24` are screening values, not calibrated error tolerances.
A converged benchmark is a prerequisite for any switch of the RL oracle.

## Quick tests and benchmarks

Run at the project root after applying the patch:

```bash
python -m pytest -q tests/test_pricing_cos.py

# Light numerical screen: run now if useful; *timing is not representative*
# while the dataset-preparation job is using the same CPUs.
python scripts/benchmark_heston_cos.py \
    --n-states 16 --terms 256 512 --xis 0 0.20 0.40 \
    --maturities 0.004 0.168 \
    --csv results/diagnostics/cos_screen.csv

# When the preparation job is finished: systematic time/accuracy experiment.
python scripts/benchmark_heston_cos.py \
    --n-states 256 --seed 2026 --repeats 3 \
    --terms 128 256 512 1024 --L 24 \
    --xis 0 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 \
    --maturities 0.004 0.040 0.168 0.336 0.504 \
    --csv results/diagnostics/cos_regular_256.csv

# Boundary stress, with explicit third-method audit of the largest discrepancy:
python scripts/benchmark_heston_cos.py \
    --scenario boundary --n-states 8 --terms 256 512 1024 --L 24 \
    --xis 0.25 0.40 --maturities 0.004 0.040 \
    --audit-worst --csv results/diagnostics/cos_boundary.csv
```

The output is a CSV and companion JSON containing the command parameters.
`--strict` makes the benchmark exit nonzero on any COS numerical failure.
Without `--strict`, a nonconvergent parameter choice is recorded as a failed
row and the complete scan finishes. Do **not** discard failed rows from the
analysis; they reveal method limitations.

Optional use of saved, **pre-existing training or validation** states:

```bash
python scripts/benchmark_heston_cos.py \
    --npz-file results/forward_experiments/<campaign>/xi_0p20/splits/validation.npz \
    --xis 0.20 --n-states 128 --terms 256 512 1024 --L 24 \
    --maturities 0.004 0.040 0.168 0.336 0.504 \
    --csv results/diagnostics/cos_xi_0p20_validation.csv
```

Replace `<campaign>` with your actual folder; `--npz-file` is read-only and
only spot/variance arrays are loaded. The script **rejects** `test.npz` to
preserve the held-out test set. In particular, never tune N, L or model
parameters using the 20k-path held-out evaluation split.

## Reading the results

- `cos_time_s`, `reference_time_s`, `reference_speedup`: median wall-clock time
  to price the **same batch** of three strikes, with all Greeks. Run on a quiet
  machine and warm up to interpret speedups. Report batch size and hardware.
- `max_abs_price`, `p95_abs_price`, and analogous Greek fields: disagreement
  between COS and the *current production Fourier implementation*. **It is
  not independently certified ground truth** in all boundary regimes.
- `max_n_convergence_*`: difference from `N` to `2N`, at fixed `L`, for the
  largest requested N; `max_L_convergence_*`: difference when additionally
  widening L to 1.5L at fixed 2N. Small N differences alone do not guarantee
  adequate truncation width.
- `--audit-worst`: evaluates the state with largest price disagreement using
  the independent Carr--Madan OTM damped integral fallback. Three-way
  comparison helps identify which pricer is unreliable. It can be slow.
- Check call/put no-arbitrage bounds, positive Gamma and put-call parity; the
  implementation flags material numerical violations rather than silently
  replacing output by Black or intrinsic value. Tiny numerical roundoff is
  clipped inside explicit `1e-5` tolerances.

### Important reproducibility boundary

No rollout, reward, SAC policy, or existing dataset is modified. Merely adding
this module does not demonstrate COS is faster or more accurate on the server.
If, after validation, you choose COS as the financial oracle, that is a **new
financial experiment version**: generate a new, fingerprinted dataset and
retrain/re-evaluate experts; do not silently mix pre-existing v3 Fourier
trajectory marks with newly computed COS values.
