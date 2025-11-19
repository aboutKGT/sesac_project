#!/bin/bash
# dj 브랜치로 전환하는 스크립트 (dj 폴더에만 있음)
cd "$(dirname "$0")"
git checkout dj 2>/dev/null || echo "Already on dj branch or dj branch does not exist"
