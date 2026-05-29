# Makefile for hnsw-vector-search
#
# Header-only library, so there is nothing to compile for the library itself.
# These targets build the test suite, the benchmark, and the demo.
#
#   make test    build and run the correctness suite
#   make bench   build and run the benchmark
#   make demo    build and run the usage example
#   make all     build all three binaries
#   make clean   remove build artifacts

CXX      ?= g++
CXXFLAGS := -std=c++20 -O3 -march=native -Wall -Wextra -Iinclude
BUILD    := build

.PHONY: all test bench demo clean

all: $(BUILD)/test_hnsw $(BUILD)/benchmark $(BUILD)/demo

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/test_hnsw: tests/test_hnsw.cpp include/hnsw/*.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) tests/test_hnsw.cpp -o $@

$(BUILD)/benchmark: bench/benchmark.cpp include/hnsw/*.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) bench/benchmark.cpp -o $@

$(BUILD)/demo: examples/demo.cpp include/hnsw/*.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) examples/demo.cpp -o $@

test: $(BUILD)/test_hnsw
	./$(BUILD)/test_hnsw

bench: $(BUILD)/benchmark
	./$(BUILD)/benchmark

demo: $(BUILD)/demo
	./$(BUILD)/demo

clean:
	rm -rf $(BUILD)
