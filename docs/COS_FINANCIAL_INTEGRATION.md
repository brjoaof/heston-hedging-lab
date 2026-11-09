# COS as an opt-in pricing backend for Heston-Q episodes

**Stage:** experimental integration, NOT clearance for a 50M-step SAC campaign.

`forward_experiment.pricing` accepts either `{"method":"fourier"}` (the historical
behavior; also the default if absent) or `{"method":"cos", "n_terms":512, "L":24.0}`.
The **same method** marks all nine calls and computes their initial forward hedge
positions. The selection is included in `config_sha256` for persisted datasets
and the COS source is added to the trainer's code fingerprint.

**Do not modify `config.json` or `configs/expert_schedule.json` in place to test COS.**
Generate independent files and a separate dataset/model root instead:

```bash
python scripts/create_cos_campaign.py
python scripts/expert_queue.py --manifest configs/expert_schedule_cos.json prepare --dry-run
```

The generator refuses to overwrite existing config, schedule, dataset root or
checkpoint root. Check `configs/heston_cos_campaign.json` retains the intended
advisor parameters (`kappa=1`, `theta=v0=.04`, `rho=+.7`), and the new schedule
retains all nine xi values and one seed. **No training is started.**

## One-episode functional check (before 225k trajectories)

```bash
python -m pytest -q tests/test_cos_financial_integration.py
python - <<'PY'
from rl_env import prepare_forward_episodes
pool,spec=prepare_forward_episodes('configs/heston_cos_campaign.json',n_paths=8,seed=2026)
print('shape:',pool.book_paths.contract_prices.shape)
print('initial positions:',pool.initial_positions.mean(axis=0))
PY
```

This confirms wiring, not numerical accuracy or training readiness. COS
accuracy remains conditioned on independently checking **both** N and L,
prices, bucket deltas, gamma and dPrice/dv in actual sampled trajectories.
The observed Fourier-vs-COS discrepancies in small-variance states are not
resolved by looking only at prices. Audit those states before selecting the
canonical oracle.

## Consequence of changing backend

A COS dataset is a **different financial marking dataset** from the Fourier-v3
campaign, although the underlying stochastic paths and seeds can be identical.
Never combine splits across the two campaigns, reuse a previous SAC checkpoint
or claim bitwise equivalence. Use `--manifest configs/expert_schedule_cos.json`
consistently for COS `prepare`, `status`, and future `run` commands.

**Important:** the existing `train_sac_scale.py` resume safety hash now includes
`pricing_cos.py`, preventing accidental resume across version drift. The
`result_root` field in the independent COS manifest isolates SAC checkpoints
from historical Fourier runs.

## End-to-end benchmarks on identical simulated trajectories

The following uses *fresh deterministic* pilot paths with the same seed for
Fourier and COS; it does **not** read a held-out test split or start RL.

```bash
python scripts/benchmark_cos_episode.py \
  --xis 0 0.20 0.40 --n-paths 16 --terms 512 --L 24 \
  --output results/diagnostics/cos_episode_screen.json
```

Run more paths/xi only after the short test passes. The report includes
end-to-end marking times, maximum and p95 differences in book value, book
Delta, Gamma, dPrice/dv, individual contract prices, bucket Delta and initial
forward positions. The report quantifies **disagreement**, not true error.
The candidate COS oracle still requires cross-checking worst Greek states with
independent numerical integration and N/L convergence before data production.
