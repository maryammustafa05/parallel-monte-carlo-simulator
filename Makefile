# Makefile — Parallel Monte Carlo Investment & Risk Simulator
#
# Phase 0 skeleton: `serial` and `hybrid` are placeholders that get real rules
# in Phases 1–3. Today only `hello` (the MPI + OpenMP smoke test) is built.
#
# Two compilers on purpose:
#   CXX    = plain g++     → used for mc_serial (no MPI, no OpenMP: a fair baseline)
#   MPICXX = mpicxx        → used for mc_hybrid (g++ plus the MPI include/lib paths)

CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra
OMPFLAGS := -fopenmp
# We only use MPI's C API. This skips OpenMPI's deprecated C++ bindings header,
# whose own code triggers -Wcast-function-type warnings we can't fix.
MPIFLAGS := -DOMPI_SKIP_MPICXX

BUILD := build

.PHONY: all serial hybrid hello test bench clean

all: hello serial hybrid

# --- Phase 0: smoke test proving MPI + OpenMP work together ---------------
hello: $(BUILD)/hello

$(BUILD)/hello: docs/hello_hybrid.cpp | $(BUILD)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPIFLAGS) $< -o $@

# --- Placeholders (filled in later phases) --------------------------------
serial:
	@echo "[make] serial: not implemented yet (Phase 1)"

hybrid:
	@echo "[make] hybrid: not implemented yet (Phases 2-3)"

test:
	@bash tests/run_tests.sh

bench:
	@echo "[make] bench: not implemented yet (Phase 6)"

# --- Helpers ---------------------------------------------------------------
$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
