#!/usr/bin/env bash
# Fine-tune Utonia on SemanticKITTI in the unified 12-class taxonomy (cross-region source model).
# Usage:  scripts/train.sh [ft] [9ch] [outdoor|indoor] [--batch B --accum A] [key=value ...]   (only the FT-9ch config exists here)
# All work is done by scripts/launch.sh at the repository root; this file only names the model and the dataset.
case "${1:-}" in ""|-*|*=*|4ch|9ch|12c|outdoor|indoor) set -- ft "$@";; esac
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/launch.sh" train utonia "$1" --dataset SemanticKITTI "${@:2}"
