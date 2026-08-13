#!/usr/bin/env bash
set -euo pipefail

REMOTE="overleaf"
OVERLEAF_BRANCH="master"

while true; do
  git fetch "$REMOTE" "$OVERLEAF_BRANCH" || true

  if ! git diff --quiet || ! git diff --cached --quiet; then
    git add --all
    git commit -m "agent: update $(date -u +'%Y-%m-%dT%H:%M:%SZ')" || true
    git push "$REMOTE" HEAD:"$OVERLEAF_BRANCH"
  fi

  sleep 5
done