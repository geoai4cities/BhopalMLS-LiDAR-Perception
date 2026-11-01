#!/usr/bin/env bash
# MAINTAINERS ONLY. Re-create the vendored frameworks (Pointcept/, LitePT/litept/) from the upstream
# repositories at the pinned commits plus our changes. Users do not need this: the trees are in git.
# Needs network. Afterwards compare with `git status`.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$HERE")"
POINTCEPT_SHA=cca1802c9083759f40a71ff740b7522c47dc112a   # Pointcept/Pointcept, 2026-02-04
LITEPT_SHA=fb44846dcdb33deeb0739c53a0b338f1091aecbb      # prs-eth/LitePT, 2026-02-12
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

rebuild() {  # name url sha dst
  local fw=$1 url=$2 sha=$3 dst=$4 src="$WORK/$fw"
  echo "== $fw @ $sha -> $dst"
  git clone -q --filter=blob:none "$url" "$src"
  git -C "$src" checkout -q --detach "$sha"
  while IFS= read -r f; do [[ -n "$f" ]] || continue; mkdir -p "$src/$(dirname "$f")"; cp "$dst/$f" "$src/$f"; done < "$HERE/${fw}_new_files.txt"
  git -C "$src" apply --whitespace=nowarn "$HERE/$fw.patch"
  rm -rf "$src/.git" "$src/.gitignore" "$src/.github"
  if [[ "$fw" == litept ]]; then find "$src/libs/sparsehash" -mindepth 1 -maxdepth 1 ! -name src ! -name COPYING ! -name README ! -name AUTHORS -exec rm -rf {} +; fi
  find "$src" \( -name '__pycache__' -o -name '*.pyc' -o -name '*.so' -o -name '*.o' \) -exec rm -rf {} + 2>/dev/null || true
  cp "$dst/UPSTREAM_COMMIT" "$src/UPSTREAM_COMMIT"
  rm -rf "$dst"; mv "$src" "$dst"
  echo "   $(find "$dst" -type f | wc -l) files"
}
rebuild pointcept https://github.com/Pointcept/Pointcept.git "$POINTCEPT_SHA" "$REPO/Pointcept"
rebuild litept https://github.com/prs-eth/LitePT.git "$LITEPT_SHA" "$REPO/LitePT/litept"
echo "done; review with: git status Pointcept LitePT/litept"
