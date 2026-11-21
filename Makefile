# Makefile — Parallel Monte Carlo Investment & Risk Simulator
#
# Two compilers on purpose:
#   CXX    = plain g++     → used for src/core, mc_serial and the tests (no MPI, no OpenMP)
#   MPICXX = mpicxx        → used for mc_hybrid (g++ plus the MPI include/lib paths)
#
# src/core is compiled ONCE into build/core/*.o and those same object files are
# linked into every program. So mc_serial, the tests and (from Phase 2) mc_hybrid
# all run byte-for-byte the same simulation code — important for design decision D2.

CXX      ?= g++
MPICXX   ?= mpicxx
CXXFLAGS := -std=c++17 -O3 -march=native -Wall -Wextra
OMPFLAGS := -fopenmp
# We only use MPI's C API. This skips OpenMPI's deprecated C++ bindings header,
# whose own code triggers -Wcast-function-type warnings we can't fix.
MPIFLAGS := -DOMPI_SKIP_MPICXX
INCLUDES := -Isrc
# -MMD -MP: let the compiler record which headers each file uses, so editing a
# header rebuilds everything that includes it.
DEPFLAGS := -MMD -MP

BUILD := build

CORE_SRCS := $(wildcard src/core/*.cpp)
CORE_OBJS := $(patsubst src/%.cpp,$(BUILD)/%.o,$(CORE_SRCS))
TEST_SRCS := $(wildcard tests/cpp/test_*.cpp)
TEST_BINS := $(patsubst tests/cpp/%.cpp,$(BUILD)/tests/%,$(TEST_SRCS))

.PHONY: all serial hybrid hello tests test bench clean

all: hello serial hybrid tests

# --- Phase 0: smoke test proving MPI + OpenMP work together ---------------
hello: $(BUILD)/hello

$(BUILD)/hello: docs/hello_hybrid.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPIFLAGS) $< -o $@

# --- Phase 1: core engine + serial baseline --------------------------------
serial: $(BUILD)/mc_serial

$(BUILD)/mc_serial: $(BUILD)/serial/main_serial.o $(CORE_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Any src/<dir>/<file>.cpp → build/<dir>/<file>.o, compiled with plain g++.
$(BUILD)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPFLAGS) -c $< -o $@

# --- Tests -------------------------------------------------------------------
tests: $(TEST_BINS)

$(BUILD)/tests/%: tests/cpp/%.cpp $(CORE_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPFLAGS) $< $(CORE_OBJS) -o $@

test: tests serial
	@bash tests/run_tests.sh

# --- Placeholders (filled in later phases) --------------------------------
hybrid:
	@echo "[make] hybrid: not implemented yet (Phases 2-3)"

bench:
	@echo "[make] bench: not implemented yet (Phase 6)"

# --- Helpers ---------------------------------------------------------------
clean:
	rm -rf $(BUILD)

-include $(CORE_OBJS:.o=.d) $(BUILD)/serial/main_serial.d $(TEST_BINS:=.d)
