#!/usr/bin/env bash
set -euo pipefail

python src/features/featurize_rdf.py \
  --labels data/meta/labels.sample.csv \
  --out data/processed/features.csv \
  --rmax 3.0 --nbins 60

python src/models/train_baseline.py \
  --features data/processed/features.csv \
  --target target_visc_ln \
  --group_by chain_len \
  --outdir outputs/smoke
test -f outputs/smoke/metrics_*json
test -f outputs/smoke/parity_log_*png
test -f outputs/smoke/parity_real_*png

echo "Smoke test passed."
