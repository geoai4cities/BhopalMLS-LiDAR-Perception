#!/usr/bin/env bash
# Evaluate a Concerto model trained on one dataset on another (paper Table 5 / supplementary cross-region table).
# Usage:  run_cross_test.sh <source>_on_<target> --weight PATH   (pairs: bhopal_on_kerala, kerala_on_bhopal, bhopal_on_tr, tr_on_bhopal, bhopal_on_pl, pl_on_bhopal, bhopal_on_sk, sk_on_bhopal)
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/scripts/launch.sh" cross concerto "$@"
