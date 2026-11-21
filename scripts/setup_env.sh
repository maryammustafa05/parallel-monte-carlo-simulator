#!/usr/bin/env bash
# setup_env.sh — check (and optionally install) the toolchain on Ubuntu / WSL2.
#
# Usage:
#   bash scripts/setup_env.sh            # check only; prints install commands for anything missing
#   bash scripts/setup_env.sh --install  # also apt-install missing system packages (asks for sudo)
#
# Always creates/updates the Python virtual environment at .venv.

set -euo pipefail
cd "$(dirname "$0")/.."

INSTALL=0
[[ "${1:-}" == "--install" ]] && INSTALL=1

missing=()
check() {  # check <command> <apt package>
    if command -v "$1" >/dev/null 2>&1; then
        printf "  [ok]      %-8s %s\n" "$1" "$("$1" --version 2>&1 | head -n1)"
    else
        printf "  [MISSING] %-8s (apt package: %s)\n" "$1" "$2"
        missing+=("$2")
    fi
}

echo "== System tools =="
check g++     build-essential
check make    build-essential
check mpicxx  libopenmpi-dev
check mpirun  openmpi-bin
check python3 python3
check git     git
python3 -c "import venv, ensurepip" 2>/dev/null || missing+=("python3-venv")

if ((${#missing[@]})); then
    pkgs=$(printf "%s\n" "${missing[@]}" | sort -u | tr '\n' ' ')
    if ((INSTALL)); then
        sudo apt update && sudo apt install -y $pkgs
    else
        echo
        echo "Missing packages. Install them with:"
        echo "  sudo apt update && sudo apt install -y $pkgs"
        exit 1
    fi
fi

echo
echo "== CPU =="
echo "  logical cores (nproc): $(nproc)"
if (( $(nproc) <= 2 )); then
    echo "  WARNING: only $(nproc) core(s). In VirtualBox raise Settings > System > Processor (VM off)."
fi

echo
echo "== Python virtual environment (.venv) =="
[[ -d .venv ]] || python3 -m venv .venv
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt
.venv/bin/python -c "import numpy, pandas, matplotlib, streamlit, pytest; print('  [ok]      numpy', numpy.__version__, '| pandas', pandas.__version__, '| matplotlib', matplotlib.__version__, '| streamlit', streamlit.__version__, '| pytest', pytest.__version__)"

echo
echo "Environment ready. Activate Python with:  source .venv/bin/activate"
