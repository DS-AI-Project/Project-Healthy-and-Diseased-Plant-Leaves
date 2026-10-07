#!/usr/bin/env bash
# Creates a Python 3.11 virtual environment (.venv) in the folder you run this from
# and installs requirements.txt into it.
# Usage:  bash setup_venv.sh        (run it from your project folder)

set -euo pipefail

VENV_DIR="$PWD/.venv"
REQ_FILE="$PWD/requirements.txt"
PY_VER="3.11"

echo "==> Project folder: $PWD"

if [[ ! -f "$REQ_FILE" ]]; then
    echo "ERROR: requirements.txt not found in $PWD"
    exit 1
fi

if [[ -d "$VENV_DIR" ]]; then
    echo "WARNING: $VENV_DIR already exists."
    read -rp "Delete and recreate it? [y/N] " ans
    if [[ "${ans,,}" == "y" ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Aborted."
        exit 1
    fi
fi

USE_UV=0

# --- 1. Get a Python 3.11 interpreter --------------------------------------
if command -v python3.11 >/dev/null 2>&1 && python3.11 -c "import venv, ensurepip" >/dev/null 2>&1; then
    echo "==> Found python3.11 at $(command -v python3.11)"
else
    echo "==> python3.11 not found (Ubuntu 26.04 ships a newer Python). Trying deadsnakes PPA..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common curl ca-certificates
    if sudo add-apt-repository -y ppa:deadsnakes/ppa \
        && sudo apt-get update \
        && sudo apt-get install -y python3.11 python3.11-venv python3.11-dev; then
        echo "==> Installed python3.11 from deadsnakes"
    else
        echo "==> deadsnakes not available for this Ubuntu release, falling back to uv..."
        # Remove the PPA again so it doesn't break future apt updates
        sudo add-apt-repository -y --remove ppa:deadsnakes/ppa >/dev/null 2>&1 || true
        sudo apt-get update >/dev/null 2>&1 || true
        USE_UV=1
    fi
fi

# --- 2. Create the venv -----------------------------------------------------
if [[ "$USE_UV" -eq 1 ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "==> Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    fi
    uv python install "$PY_VER"
    # --seed puts pip/setuptools/wheel into the venv so it behaves like a normal venv
    uv venv --seed --python "$PY_VER" "$VENV_DIR"
else
    python3.11 -m venv "$VENV_DIR"
fi

# --- 3. Install requirements ------------------------------------------------
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "==> Using: $(python --version) at $(command -v python)"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$REQ_FILE"

echo
echo "==> Done. Virtual environment is in: $VENV_DIR"
echo "    Activate it with:   source .venv/bin/activate"
echo "    Deactivate with:    deactivate"
