#!/usr/bin/env bash
# One-command pod setup. Idempotent.
#   bash bootstrap.sh              # full setup
#   bash bootstrap.sh --save-auth  # persist Claude Code auth to volume
set -euo pipefail
REPO_DIR="/workspace/caee-revision"
AUTH_BACKUP="/workspace/.claude_backup"

if [[ "${1:-}" == "--save-auth" ]]; then
  mkdir -p "$AUTH_BACKUP"
  cp -r ~/.claude "$AUTH_BACKUP/dot_claude" 2>/dev/null || true
  cp ~/.claude.json "$AUTH_BACKUP/" 2>/dev/null || true
  echo "[bootstrap] Claude auth saved."
  exit 0
fi

apt-get update -qq && apt-get install -y -qq tmux git rsync htop unzip > /dev/null
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"; mkdir -p "$HF_HOME"
grep -q HF_HOME ~/.bashrc || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc

git config --global user.name  "Hector Borobia"
git config --global user.email "hecboar@doctor.upv.es"
if [[ -d "$REPO_DIR/.git" && -n "${GITHUB_TOKEN:-}" ]]; then
  cd "$REPO_DIR"
  git remote set-url origin "https://${GITHUB_TOKEN}@github.com/hecboar/caee-revision.git"
  git pull --ff-only || true
fi

if ! command -v claude >/dev/null 2>&1; then
  curl -fsSL https://claude.ai/install.sh | bash
  export PATH="$HOME/.local/bin:$PATH"
  grep -q '.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi
if [[ -d "$AUTH_BACKUP" ]]; then
  cp -r "$AUTH_BACKUP/dot_claude" ~/.claude 2>/dev/null || true
  cp "$AUTH_BACKUP/.claude.json" ~/ 2>/dev/null || true
fi
[[ -f "$REPO_DIR/revision/requirements_revision.txt" ]] && \
  pip install -q -r "$REPO_DIR/revision/requirements_revision.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "[bootstrap] DONE — next: tmux new -s claude && cd $REPO_DIR && claude"
