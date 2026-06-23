#!/usr/bin/env bash
# Runs INSIDE the Lightning job image.
# Assumes: git present, repo already cloned + checked out at $GITHUB_SHA,
# and cwd is the repo root (handled by the dispatch command).
set -euo pipefail

: "${PY_VERSION:?PY_VERSION must be set}"

echo "Installing uv"
pip install uv

echo "Setting up Python ${PY_VERSION} via uv"
uv python install "${PY_VERSION}"
uv venv --python "${PY_VERSION}" /workspace/venv
# shellcheck disable=SC1091
source /workspace/venv/bin/activate

echo "Installing project + coverage tooling"
uv pip install -e .
uv pip install coverage coveralls

echo "Env ready ($(python --version))"