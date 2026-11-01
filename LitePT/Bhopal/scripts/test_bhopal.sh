#!/usr/bin/env bash
# Evaluate a trained LitePT run on the BhopalMLS test split.
# Usage mirrors the paper runs:  test_bhopal.sh <train|insseg> [4ch|9ch] [12c|21c] [--weight PATH]
# All work is done by scripts/launch.sh at the repository root; this file only names the model.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/launch.sh" test litept "$@"
