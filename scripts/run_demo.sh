#!/usr/bin/env bash
# One complete demo: build, run the parallel simulator with default settings,
# and draw the charts.
#
# Usage: bash scripts/run_demo.sh [extra mc_hybrid flags, e.g. --trials 5000000]
# Output: results/demo/{fan_chart.csv, final_hist.csv, summary.json}
#         docs/images/{fan_chart.png, distribution.png}

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=.venv/bin/python
if [ ! -x "$PYTHON" ]; then
    echo "error: .venv not found. Create it with:"
    echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Use up to 4 processes, but never more than the machine's cores.
CORES=$(nproc)
PROCESSES=$(( CORES < 4 ? CORES : 4 ))
OUT=results/demo

# hwloc prints a harmless display warning on VMs; this hides only that (see README).
export HWLOC_COMPONENTS=-gl

echo "== Building =="
make --no-print-directory serial hybrid

echo
echo "== Simulating on $PROCESSES MPI processes x 1 thread =="
mpirun --bind-to none -np "$PROCESSES" ./build/mc_hybrid --out "$OUT" "$@"

echo
echo "== Drawing charts =="
"$PYTHON" scripts/plot_fan_chart.py --input "$OUT" --output docs/images/fan_chart.png
"$PYTHON" scripts/plot_distribution.py --input "$OUT" --output docs/images/distribution.png

echo
echo "Done. Open docs/images/fan_chart.png and docs/images/distribution.png,"
echo "or start the dashboard:  .venv/bin/streamlit run dashboard/app.py"
