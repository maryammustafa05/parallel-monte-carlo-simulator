#!/usr/bin/env bash
# Edge-case tests (Phase 7): run the real programs on unusual inputs and check
# the answers against what must be true.
#
#   1 trial / fewer trials than workers  -> parallel == serial, std = 0
#   sigma = 0                            -> every future is the exact formula value
#   months = 1                           -> one-step mean matches the formula
#   huge volatility                      -> portfolios wiped out: median 0, no inf/nan
#   bad input to mc_hybrid under mpirun  -> exit code != 0, ONE error message (rank 0)
#
# Usage: tests/edge_cases.sh            (run from anywhere, after `make`)

set -u
cd "$(dirname "$0")/.."
export HWLOC_COMPONENTS=-gl   # hides a harmless hwloc display warning on VMs (see README)

WORK=build/edge_cases
rm -rf "$WORK"
mkdir -p "$WORK"
failed=0

# run NAME COMMAND... : runs a simulation writing to $WORK/NAME; fails the test on error.
run() {
    local name=$1; shift
    if ! "$@" --out "$WORK/$name" > "$WORK/$name.log" 2>&1; then
        echo "[edge] $name: program failed -> FAIL"
        sed 's/^/    /' "$WORK/$name.log" | head -5
        failed=1
        return 1
    fi
}

# check LABEL PYTHON-EXPRESSIONS... : evaluates assertions with s = summary.json of the
# first named run, plus helpers; prints PASS/FAIL.
check() {
    local label=$1 code=$2
    if python3 - "$WORK" "$label" "$code" <<'EOF'
import csv, json, math, sys
work, label, code = sys.argv[1], sys.argv[2], sys.argv[3]

def load(name):
    text = open(f"{work}/{name}/summary.json").read()
    for bad in ("nan", "inf", "null"):
        assert bad not in text.lower(), f"{name}/summary.json contains {bad}"
    return json.loads(text)

def fan(name):
    return list(csv.DictReader(open(f"{work}/{name}/fan_chart.csv")))

def rel(a, b):
    return abs(a - b) / abs(b)

try:
    exec(code)
except AssertionError as err:
    print(f"[edge] {label}: {err} -> FAIL")
    sys.exit(1)
print(f"[edge] {label} -> PASS")
EOF
    then :; else failed=1; fi
}

# --- 1 trial, and fewer trials than workers -----------------------------------
run one_serial ./build/mc_serial --trials 1 &&
run one_np4 mpirun --bind-to none -np 4 ./build/mc_hybrid --trials 1 --threads 2 &&
check "trials=1: serial == 4 processes x 2 threads, std = 0" '
s, p = load("one_serial"), load("one_np4")
assert s["histogram"]["checksum"] == p["histogram"]["checksum"], "checksums differ"
f = s["final_value"]
assert s["params"]["trials"] == 1
assert f["std"] == 0 and f["min"] == f["max"] == f["mean"], f"bad stats {f}"
assert p["final_value"] == f, "final stats differ"
'

run ten_serial ./build/mc_serial --trials 10 &&
run ten_np3 mpirun --bind-to none -np 3 ./build/mc_hybrid --trials 10 --threads 2 &&
check "trials=10 < 3 processes x 2 threads x 4096: same as serial" '
s, p = load("ten_serial"), load("ten_np3")
assert s["histogram"] == p["histogram"], "histograms differ"
assert s["risk"] == p["risk"] and s["final_value"]["median"] == p["final_value"]["median"]
'

# --- sigma = 0: no randomness, so the answer is known exactly -----------------
run flat ./build/mc_serial --sigma 0 --mu 0.07 --trials 5000 &&
check "sigma=0, mu=7%: every future = V0 (1 + mu/12)^T exactly" '
s = load("flat"); f = s["final_value"]
exact = 10000 * (1 + 0.07 / 12) ** 120
mean, median = f["mean"], f["median"]
# summary.json stores 10 significant digits, so 1e-9 is "equal" at that precision.
assert rel(mean, exact) < 1e-9, f"mean {mean} vs {exact}"
assert f["min"] == f["max"] and f["std"] == 0, f"not identical: {f}"
assert rel(median, exact) < 1e-5, f"median {median} (histogram resolution)"
assert s["risk"]["var"] == 0 and s["risk"]["prob_loss"] == 0
'

run flat_loss ./build/mc_serial --sigma 0 --mu -0.05 --trials 5000 &&
check "sigma=0, mu=-5%: certain loss, VaR = V0 - V_T" '
s = load("flat_loss")
v_t = 10000 * (1 - 0.05 / 12) ** 120
assert s["risk"]["prob_loss"] == 1, s["risk"]
var = s["risk"]["var"]
assert rel(var, 10000 - v_t) < 1e-5, f"VaR {var} vs {10000 - v_t}"
'

# --- months = 1 ---------------------------------------------------------------
run one_month ./build/mc_serial --months 1 --trials 200000 &&
check "months=1: 2 fan-chart rows, mean = V0 (1 + mu/12) within 4 SE" '
s = load("one_month"); rows = fan("one_month")
assert [r["month"] for r in rows] == ["0", "1"], "wrong fan_chart rows"
expected = 10000 * (1 + 0.07 / 12)
se = 10000 * (0.15 / math.sqrt(12)) / math.sqrt(200000)
assert abs(s["final_value"]["mean"] - expected) < 4 * se, s["final_value"]["mean"]
'

# --- huge volatility: -100 % months wipe portfolios out ----------------------
run wipe_serial ./build/mc_serial --sigma 3 --trials 20000 &&
run wipe_np3 mpirun --bind-to none -np 3 ./build/mc_hybrid --sigma 3 --trials 20000 --threads 2 &&
check "sigma=300%: wiped-out futures give median 0, finite output, parallel == serial" '
s, p = load("wipe_serial"), load("wipe_np3")
assert s["risk"]["prob_total_loss"] > 0.99, s["risk"]
assert s["final_value"]["median"] == 0 and s["risk"]["var"] == 10000, s["final_value"]
assert s["histogram"]["checksum"] == p["histogram"]["checksum"], "checksums differ"
assert all(float(r["median"]) >= 0 for r in fan("wipe_serial"))
'

run partial ./build/mc_serial --sigma 1 --trials 20000 &&
check "sigma=100%: some wiped out, percentiles ordered, mean > median" '
s = load("partial"); f = s["final_value"]; r = s["risk"]
assert 0 < r["prob_total_loss"] < 0.5, r
assert f["p05"] <= f["median"] <= f["p95"], f
assert f["mean"] > f["median"], "a long right tail should pull the mean above the median"
'

# --- bad input under mpirun: clean exit, exactly one message -------------------
bad_failed=0
for bad in "--mu -12" "--months 100000000" "--trials 0" "--threads 999" "--sigma abc"; do
    # shellcheck disable=SC2086  # $bad must be split into flag and value
    if mpirun --bind-to none -np 3 ./build/mc_hybrid $bad --out "$WORK/bad" \
            > "$WORK/bad.log" 2>&1; then
        echo "[edge] mpirun mc_hybrid $bad: accepted -> FAIL"
        bad_failed=1
    elif [ "$(grep -c '^error:' "$WORK/bad.log")" -ne 1 ]; then
        echo "[edge] mpirun mc_hybrid $bad: expected exactly 1 error line -> FAIL"
        sed 's/^/    /' "$WORK/bad.log" | head -5
        bad_failed=1
    fi
done
[ $bad_failed -ne 0 ] && failed=1
[ $bad_failed -eq 0 ] && echo "[edge] bad input under mpirun: exit != 0 and one error message -> PASS"

exit $failed
