#!/usr/bin/env bash
# Consistency test (design decision D2): the parallel program must give the
# SAME result as the serial one, whatever the number of threads.
#
# What "same" means:
#   - histogram checksum (all integer counts)      -> must be identical
#   - fan-chart percentiles (p05..p95)              -> must be identical
#     (they are computed only from the counts)
#   - means, std                                    -> equal to ~1e-9 relative
#     (floating-point sums added in a different order round slightly differently)
#
# Usage: tests/consistency_test.sh            (run from anywhere)

set -u
cd "$(dirname "$0")/.."

# 50 000 trials = 12 full chunks + 1 partial chunk (50000 = 12*4096 + 848),
# so threads get uneven amounts of work on purpose.
ARGS="--trials 50000 --months 60 --seed 2026"
WORK=build/consistency
THREAD_COUNTS="1 2 3 4"

rm -rf "$WORK"
mkdir -p "$WORK"

if ! ./build/mc_serial $ARGS --out "$WORK/serial" > /dev/null; then
    echo "[consistency] mc_serial failed -> FAIL"
    exit 1
fi

failed=0
for threads in $THREAD_COUNTS; do
    out="$WORK/hybrid_t$threads"
    if ! ./build/mc_hybrid $ARGS --threads "$threads" --out "$out" > /dev/null; then
        echo "[consistency] mc_hybrid --threads $threads failed -> FAIL"
        failed=1
        continue
    fi
    # Compare with Python's standard library only (no numpy needed).
    if ! python3 - "$WORK/serial" "$out" "threads=$threads" <<'EOF'
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
    then
        failed=1
    fi
done

exit $failed
