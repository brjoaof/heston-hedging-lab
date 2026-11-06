"""Train/evaluate SAC-Band under the SAME Heston-Q, costs and validation splits.

Only the action semantics differ from train_sac_trade.py. The common trainer
and checkpoint/evaluation implementation is intentionally reused.

Usage:
    python train_sac_band.py train --out-dir results/sac_band/pilot_c2_30k \
        --timesteps 30000 --eval-every 5000 --penalty 2 --max-band-width 1.5
    python train_sac_band.py evaluate --run-dir results/sac_band/pilot_c2_30k
"""
from train_sac_trade import main

if __name__ == "__main__":
    main(action_mode="band")
