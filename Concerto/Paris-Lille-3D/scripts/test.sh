#!/usr/bin/env bash
# Evaluate a run of Concerto on Paris-Lille-3D in the unified 12-class taxonomy (cross-region source model).
# Usage:  scripts/test.sh [ft] [9ch] [outdoor|indoor] [--batch B --accum A] [key=value ...]   (only the FT-9ch config exists here) [--weight PATH]
# All work is done by scripts/launch.sh at the repository root; this file only names the model and the dataset.
case "${1:-}" in ""|-*|*=*|4ch|9ch|12c|outdoor|indoor) set -- ft "$@";; esac
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/scripts/launch.sh" test concerto "$1" --dataset Paris-Lille-3D "${@:2}"
