#!/usr/bin/env bash
# Evaluate a trained Utonia run on the BhopalMLS test split.
# Usage mirrors the paper runs:  test_bhopal.sh <lp|dp|lora|ft|insseg_lp|insseg_dp|insseg_ft> [4ch|9ch] [12c|21c] [v10] [outdoor|indoor] [--weight PATH]
# All work is done by scripts/launch.sh at the repository root; this file only names the model.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/launch.sh" test utonia "$@"
