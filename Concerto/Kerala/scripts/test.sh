#!/usr/bin/env bash
# Evaluate a run of Concerto on Ke-MLS in the unified 12-class taxonomy (cross-region source model).
# Usage:  scripts/test.sh [lp|dp|lora|ft|insseg_lp|insseg_dp|insseg_ft] [4ch|9ch] [outdoor|indoor] [--batch B --accum A] [key=value ...]   (no protocol = ft 9ch, the paper's cross-region source model) [--weight PATH]
# All work is done by scripts/launch.sh at the repository root; this file only names the model and the dataset.
case "${1:-}" in ""|-*|*=*|4ch|9ch|12c|outdoor|indoor) set -- ft "$@";; esac
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/launch.sh" test concerto "$1" --dataset Kerala "${@:2}"
