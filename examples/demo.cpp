// demo.cpp
//
// Minimal end-to-end usage of the index. Build it, add vectors, query it.
// This is the smallest possible thing that shows the API.

#include <cstdio>
#include <random>
#include <vector>

#include "hnsw/hnsw.hpp"

using namespace hnsw;

int main() {
    const size_t dim = 64;
    const size_t n = 10000;

    // Make some random vectors to index.
    std::mt19937_64 rng(123);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<float> data(n * dim);
    for (auto& x : data) x = g(rng);

    // Build the index.
    HnswConfig cfg;
    cfg.M = 16;
    cfg.ef_construction = 200;
    cfg.max_elements = n;
    HnswIndex index(dim, Metric::L2, cfg);
    index.add_batch(data.data(), n);
    std::printf("indexed %zu vectors of dim %zu\n", index.size(), dim);

    // Query with vector number 42 and expect to find itself first.
    const float* query = &data[42 * dim];
    auto results = index.search(query, /*k=*/5, /*ef=*/64);

    std::printf("\ntop 5 neighbors of vector 42:\n");
    for (auto& [dist, id] : results) {
        std::printf("  id %-6zu  distance %.4f\n", id, dist);
    }

    // Persist and reload.
    index.save("/tmp/demo_index.bin");
    auto reloaded = HnswIndex::load("/tmp/demo_index.bin");
    std::printf("\nreloaded index holds %zu vectors\n", reloaded.size());

    return 0;
}
