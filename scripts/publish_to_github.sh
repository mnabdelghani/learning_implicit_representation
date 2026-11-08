#!/usr/bin/env bash
# Requires a local GitHub-authenticated git client. Does not force-push.
set -euo pipefail
REPO='https://github.com/mnabdelghani/learning_implicit_representation.git'
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
python - "$ROOT" <<'PY'
import pathlib,sys
root=pathlib.Path(sys.argv[1])
for p in root.rglob('*'):
 if p.is_file():
  if 'manuscript' in p.name.lower() or p.parts[len(root.parts)] == 'paper':
   raise SystemExit(f'Forbidden main-paper artifact: {p}')
  if p.stat().st_size >= 100_000_000:
   raise SystemExit(f'File too large for ordinary Git: {p}')
print('Validated manuscript exclusion and GitHub file-size limits.')
PY
git clone "$REPO" "$WORK/repo"
python - "$ROOT" "$WORK/repo" <<'PY'
from pathlib import Path
import shutil,sys
source,dest=map(Path,sys.argv[1:])
for p in source.rglob('*'):
 if not p.is_file():continue
 if any(x in ('__pycache__','.git','dist','runs') for x in p.relative_to(source).parts):continue
 target=dest/p.relative_to(source)
 target.parent.mkdir(parents=True,exist_ok=True)
 shutil.copy2(p,target)
PY
cd "$WORK/repo"
git add -A
if git diff --cached --quiet; then echo 'Repository is already current.'; exit 0; fi
printf '\nFiles to publish:\n'
git diff --cached --stat
printf '\nPublish these changes to main? [y/N] '
read -r answer
case "$answer" in y|Y|yes|YES) ;; *) echo 'Cancelled; no remote changes.'; exit 1;; esac
git commit -m 'Publish reproducible implicit-representation supplement and code'
git push origin HEAD:main
printf '\nPublished commit: '; git rev-parse HEAD
