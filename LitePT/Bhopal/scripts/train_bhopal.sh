#!/usr/bin/env bash
# Train LitePT on BhopalMLS.
# Usage mirrors the paper runs:  train_bhopal.sh <train|insseg> [4ch|9ch] [12c|21c]
# All work is done by scripts/launch.sh at the repository root; this file only names the model.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/launch.sh" train litept "$@"
