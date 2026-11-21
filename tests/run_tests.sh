#!/usr/bin/env bash
# Runs every test. Called by `make test` (which builds the binaries first).
# Exits with a non-zero code if anything fails.

set -u
cd "$(dirname "$0")/.."

failed=0

echo "== C++ unit tests =="
for test_bin in build/tests/test_*; do
    [ -x "$test_bin" ] || continue
    [[ "$test_bin" == *.d ]] && continue   # skip compiler dependency files
    if ! "$test_bin"; then
        failed=1
    fi
done

echo
echo "== End-to-end: mc_serial =="
out_dir="build/test_run"
rm -rf "$out_dir"
if ./build/mc_serial --trials 20000 --months 24 --out "$out_dir" > /dev/null; then
    # fan_chart.csv must have a header + months 0..24 = 26 lines.
    lines=$(wc -l < "$out_dir/fan_chart.csv")
    if [ "$lines" -eq 26 ] && [ -s "$out_dir/summary.json" ]; then
        echo "[e2e] mc_serial wrote fan_chart.csv ($lines lines) and summary.json -> PASS"
    else
        echo "[e2e] unexpected output (fan_chart.csv has $lines lines) -> FAIL"
        failed=1
    fi
else
    echo "[e2e] mc_serial exited with an error -> FAIL"
    failed=1
fi

# Bad input must be rejected with a non-zero exit code.
for bad in "--trials 0" "--sigma -0.1" "--confidence 1.5" "--months abc" "--unknown 1"; do
    # shellcheck disable=SC2086  # we want word splitting of $bad
    if ./build/mc_serial $bad --out "$out_dir" > /dev/null 2>&1; then
        echo "[e2e] '$bad' was accepted but should be rejected -> FAIL"
        failed=1
    fi
done
[ $failed -eq 0 ] && echo "[e2e] invalid inputs are rejected -> PASS"

echo
echo "== Consistency: serial == hybrid =="
if ! bash tests/consistency_test.sh; then
    failed=1
fi

# Phase 4 adds: pytest tests/python (C++ vs independent NumPy reference).

echo
if [ $failed -eq 0 ]; then
    echo "ALL TESTS PASSED"
else
    echo "SOME TESTS FAILED"
fi
exit $failed
