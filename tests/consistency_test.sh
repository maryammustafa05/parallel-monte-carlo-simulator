#!/usr/bin/env bash
# Consistency test (design decision D2): the parallel program must give the
# SAME result as the serial one, whatever the number of processes and threads.
#
# What "same" means:
#   - histogram checksum (all integer counts)      -> must be identical
#   - percentiles, VaR, prob_loss, min, max         -> must be identical
#     (they come only from the integer counts and exact comparisons)
#   - means, std                                    -> equal to ~1e-9 relative
#     (floating-point sums added in a different order round slightly differently)
#
# Usage: tests/consistency_test.sh            (run from anywhere)

set -u
cd "$(dirname "$0")/.."

# Open MPI's hardware detection tries to reach the VM's display and prints a
# harmless "Authorization required" line; this hides only that (see README).
export HWLOC_COMPONENTS=-gl

WORK=build/consistency
rm -rf "$WORK"
mkdir -p "$WORK"
failed=0

# compare_runs SERIAL_DIR PARALLEL_DIR LABEL -> exit 0 if they match
compare_runs() {
    python3 - "$1" "$2" "$3" <<'EOF'
import csv, json, sys

serial_dir, parallel_dir, label = sys.argv[1], sys.argv[2], sys.argv[3]
REL_TOL = 1e-9
problems = []

def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)

s = json.load(open(f"{serial_dir}/summary.json"))
p = json.load(open(f"{parallel_dir}/summary.json"))
if s["histogram"] != p["histogram"]:
    problems.append(f"histogram differs: {s['histogram']} vs {p['histogram']}")
for key in ("median", "p05", "p95", "min", "max"):
    if s["final_value"][key] != p["final_value"][key]:
        problems.append(f"final_value.{key}: {s['final_value'][key]} vs {p['final_value'][key]}")
for key in ("mean", "std"):
    if rel_diff(s["final_value"][key], p["final_value"][key]) > REL_TOL:
        problems.append(f"final_value.{key}: {s['final_value'][key]} vs {p['final_value'][key]}")
if s["risk"] != p["risk"]:
    problems.append(f"risk differs: {s['risk']} vs {p['risk']}")

s_rows = list(csv.DictReader(open(f"{serial_dir}/fan_chart.csv")))
p_rows = list(csv.DictReader(open(f"{parallel_dir}/fan_chart.csv")))
if len(s_rows) != len(p_rows):
    problems.append("fan_chart.csv row counts differ")
for sr, pr in zip(s_rows, p_rows):
    for col in ("p05", "p25", "median", "p75", "p95"):
        if sr[col] != pr[col]:
            problems.append(f"month {sr['month']} {col}: {sr[col]} vs {pr[col]}")
    if rel_diff(float(sr["mean"]), float(pr["mean"])) > REL_TOL:
        problems.append(f"month {sr['month']} mean: {sr['mean']} vs {pr['mean']}")

if problems:
    print(f"[consistency] {label}: DIFFERENT from serial -> FAIL")
    for line in problems[:10]:
        print("    " + line)
    sys.exit(1)
print(f"[consistency] {label}: checksum {p['histogram']['checksum']} == serial -> PASS")
EOF
}

# check_case NAME "ARGS" "PROCESS COUNTS" "THREAD COUNTS"
check_case() {
    local name=$1 args=$2 process_counts=$3 thread_counts=$4
    local serial_dir="$WORK/$name/serial"
    # shellcheck disable=SC2086  # $args must be split into separate flags
    if ! ./build/mc_serial $args --out "$serial_dir" > /dev/null; then
        echo "[consistency] $name: mc_serial failed -> FAIL"
        failed=1
        return
    fi
    for np in $process_counts; do
        for threads in $thread_counts; do
            local out="$WORK/$name/np${np}_t${threads}"
            local label
            label=$(printf "%-6s np=%d threads=%d" "$name" "$np" "$threads")
            # shellcheck disable=SC2086
            if ! mpirun --bind-to none -np "$np" ./build/mc_hybrid $args --threads "$threads" \
                    --out "$out" > /dev/null; then
                echo "[consistency] $label: mc_hybrid failed -> FAIL"
                failed=1
                continue
            fi
            compare_runs "$serial_dir" "$out" "$label" || failed=1
        done
    done
}

# Main case: 50 000 trials = 12 full chunks + 1 partial chunk (50000 = 12*4096 + 848).
# 13 chunks don't divide evenly by 2, 3 or 4 processes, and 50 000 is not
# divisible by 3, so uneven splits are tested on purpose.
check_case "main" "--trials 50000 --months 60 --seed 2026" "1 2 3 4" "1 2 4"

# Tiny case: 5 000 trials = only 2 chunks, so with 3 or 4 processes some
# processes get no work at all. They must not spoil min/max or the sums.
check_case "tiny" "--trials 5000 --months 12 --seed 7" "3 4" "2"

exit $failed
