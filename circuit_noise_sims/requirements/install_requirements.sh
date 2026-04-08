\#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_REQ_FILE="$SCRIPT_DIR/python.txt"
SYS_REQ_FILE="$SCRIPT_DIR/system-ubuntu.txt"

WITH_LATEX=0

for arg in "$@"; do
  case "$arg" in
    --with-latex)
      WITH_LATEX=1
      ;;
    *)
      echo "Unknown option: $arg"
      echo "Usage: bash requirements/install_requirements.sh [--with-latex]"
      exit 1
      ;;
  esac
done

echo "Installing Python requirements from $PY_REQ_FILE"
python3 -m pip install -r "$PY_REQ_FILE"

if [[ "$WITH_LATEX" -eq 1 ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "Installing system LaTeX requirements from $SYS_REQ_FILE"
    sudo apt-get update
    sudo xargs -a "$SYS_REQ_FILE" apt-get install -y
  else
    echo "apt-get not found. Install packages listed in $SYS_REQ_FILE manually."
    exit 1
  fi
fi

echo "Done."
