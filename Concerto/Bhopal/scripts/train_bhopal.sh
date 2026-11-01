#!/usr/bin/env bash
# Train Concerto on BhopalMLS.
# Usage mirrors the paper runs:  train_bhopal.sh <lp|dp|lora|ft|insseg_lp|insseg_dp|insseg_ft> [4ch|9ch] [12c|21c] [v10] [outdoor|indoor]
# All work is done by scripts/launch.sh at the repository root; this file only names the model.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/launch.sh" train concerto "$@"
