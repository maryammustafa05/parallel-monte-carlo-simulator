#!/usr/bin/env bash
# Scalability benchmark (objective O4, design decision D5).
#
# Strong scaling: a FIXED problem (--trials N) on more and more workers.
#   - serial baseline (mc_serial, no MPI/OpenMP) = the "1" in speedup = T1 / Tp
#   - every (processes P, threads T) with P*T <= cores, pinned to separate cores
#   - a few P*T = 2*cores configurations (more workers than cores, flagged
#     oversubscribed=1) to show what happens past the core count
# Weak scaling: the problem GROWS with the workers (N = per-worker trials x P),
#   so ideal time stays flat.
#
# Every configuration runs REPEATS times. Repeats are interleaved in rounds
# (round 1 of everything, then round 2, ...) so that if the host machine gets
# slower or faster for a while, all configurations are affected equally.
#
# Usage: bash scripts/run_benchmark.sh [--trials N] [--weak-trials N] [--repeats R] [--quick]
# Output: results/benchmark.csv, results/benchmark_weak.csv, results/benchmark_env.txt

set -euo pipefail
cd "$(dirname "$0")/.."

TRIALS=2000000            # strong scaling problem size (serial run ~ 20-60 s)
WEAK_TRIALS=400000        # weak scaling: trials per worker
REPEATS=3
OUT_DIR=results
while [ $# -gt 0 ]; do
    case "$1" in
        --trials)      TRIALS=$2; shift 2 ;;
        --weak-trials) WEAK_TRIALS=$2; shift 2 ;;
        --repeats)     REPEATS=$2; shift 2 ;;
        --quick)       TRIALS=100000; WEAK_TRIALS=20000; REPEATS=1; OUT_DIR=build/bench_quick; shift ;;
        *) echo "unknown option $1"; exit 1 ;;
    esac
done

CORES=$(nproc)
CSV="$OUT_DIR/benchmark.csv"
WEAK_CSV="$OUT_DIR/benchmark_weak.csv"
RUN_DIR=build/bench_runs
HEADER="mode,processes,threads,total_workers,trials,run,total_s,compute_s,reduce_s,oversubscribed"

# hwloc prints a harmless display warning on VMs; this hides only that (see README).
export HWLOC_COMPONENTS=-gl

make --no-print-directory serial hybrid > /dev/null
mkdir -p "$OUT_DIR" "$RUN_DIR"
echo "$HEADER" > "$CSV"
echo "$HEADER" > "$WEAK_CSV"

# Record the machine and conditions next to the numbers.
{
    echo "date: $(date -Iseconds)"
    echo "nproc: $CORES"
    lscpu | grep -E "Model name|Thread\(s\) per core|Core\(s\) per socket"
    echo "compiler: $(g++ --version | head -1)"
    echo "mpi: $(mpirun --version | head -1)"
    echo "trials (strong): $TRIALS, trials per worker (weak): $WEAK_TRIALS, repeats: $REPEATS"
} > "$OUT_DIR/benchmark_env.txt"

# Reads total/compute/reduce seconds from a run's summary.json -> "t,c,r"
read_timing() {
    python3 - "$1/summary.json" <<'EOF'
import json, sys
t = json.load(open(sys.argv[1]))["timing"]
print(f"{t['total_s']},{t['compute_s']},{t['reduce_s']}")
EOF
}

# run_one CSV MODE P T N RUN
run_one() {
    local csv=$1 mode=$2 p=$3 t=$4 n=$5 run=$6
    local out="$RUN_DIR/${mode}_p${p}_t${t}_n${n}_r${run}"
    local workers=$((p * t)) over=0
    if [ "$mode" = serial ]; then
        ./build/mc_serial --trials "$n" --out "$out" > /dev/null
    elif [ "$workers" -le "$CORES" ]; then
        # Each process gets its own block of T cores; its threads are pinned
        # inside that block (one thread per core), so no two workers share a core.
        OMP_PLACES=cores OMP_PROC_BIND=close \
            mpirun --map-by "slot:PE=$t" --bind-to core -np "$p" \
            ./build/mc_hybrid --trials "$n" --threads "$t" --out "$out" > /dev/null
    else
        # More workers than cores: no pinning possible; the OS time-shares the cores.
        over=1
        mpirun --bind-to none --oversubscribe -np "$p" \
            ./build/mc_hybrid --trials "$n" --threads "$t" --out "$out" > /dev/null
    fi
    local timing
    timing=$(read_timing "$out")
    echo "$mode,$p,$t,$workers,$n,$run,$timing,$over" >> "$csv"
    printf "  %-7s P=%d T=%d  N=%-9s run %d: %ss total\n" "$mode" "$p" "$t" "$n" "$run" "${timing%%,*}"
    rm -rf "$out"
}

# --- Strong-scaling configurations: every P x T that fits on the cores ---
CONFIGS=()
for ((p = 1; p <= CORES; p++)); do
    for ((t = 1; p * t <= CORES; t++)); do
        CONFIGS+=("$p $t")
    done
done
# ...plus 2x as many workers as cores, in four different shapes.
OVER=$((2 * CORES))
for shape in "$OVER 1" "1 $OVER"; do CONFIGS+=("$shape"); done
if [ $((OVER % 2)) -eq 0 ]; then CONFIGS+=("2 $((OVER / 2))" "$((OVER / 2)) 2"); fi

echo "== Strong scaling: N = $TRIALS trials, ${#CONFIGS[@]} parallel configurations + serial, $REPEATS rounds =="
for ((run = 1; run <= REPEATS; run++)); do
    echo "-- round $run of $REPEATS"
    run_one "$CSV" serial 1 1 "$TRIALS" "$run"   # serial baseline first in every round
    for cfg in "${CONFIGS[@]}"; do
        # shellcheck disable=SC2086  # split "P T" into two arguments
        run_one "$CSV" hybrid $cfg "$TRIALS" "$run"
    done
done

echo "== Weak scaling: $WEAK_TRIALS trials per MPI process, 1..$CORES processes =="
for ((run = 1; run <= REPEATS; run++)); do
    echo "-- round $run of $REPEATS"
    for ((p = 1; p <= CORES; p++)); do
        run_one "$WEAK_CSV" hybrid "$p" 1 $((WEAK_TRIALS * p)) "$run"
    done
done

rm -rf "$RUN_DIR"
echo
echo "wrote $CSV, $WEAK_CSV and $OUT_DIR/benchmark_env.txt"
echo "next: .venv/bin/python scripts/plot_benchmark.py --input $CSV --weak $WEAK_CSV"
