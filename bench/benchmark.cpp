// benchmark.cpp
//
// Measures the two numbers that matter for an ANN index: recall against an
// exact brute-force ground truth, and query throughput. Sweeps the ef search
// parameter to trace the recall versus latency tradeoff, and reports build
// time and the speedup over brute force.
//
// Usage:
//   benchmark [n] [dim] [nq] [k]
// defaults: n=50000 dim=128 nq=1000 k=10
//
// Output: a human-readable table to stdout and a CSV to results/sweep.csv that
// the plotting script turns into the recall/latency curve.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <set>
#include <vector>

#include "hnsw/brute_force.hpp"
#include "hnsw/hnsw.hpp"

using namespace hnsw;
using Clock = std::chrono::high_resolution_clock;

static double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static std::vector<float> make_embedding_like_data(size_t n, size_t ambient,
                                                   size_t intrinsic,
                                                   size_t n_clusters,
                                                   uint64_t seed) {
    // Real embeddings have two properties that matter for ANN: they cluster,
    // and they live near a low-dimensional manifold inside the ambient space
    // (their intrinsic dimensionality is far below the vector length). This
    // generator reproduces both. We draw a low-dim latent per point around a
    // cluster center, then project it up to the ambient dimension through a
    // fixed random linear map and add a little noise.
    //
    // This matters because ANN indexes exploit low intrinsic dimensionality.
    // Benchmarking on pure i.i.d. gaussian noise in the full ambient dimension
    // measures the one regime where no graph index can win (every pair of
    // points is nearly equidistant), and is not representative of real use.
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.02f);
    std::normal_distribution<float> jitter(0.0f, 0.4f);

    // Random projection from intrinsic to ambient space.
    std::vector<float> proj(intrinsic * ambient);
    for (auto& x : proj) x = g(rng);

    // Cluster centers in latent space.
    std::vector<std::vector<float>> centers(n_clusters,
                                            std::vector<float>(intrinsic));
    for (auto& c : centers)
        for (auto& x : c) x = g(rng) * 2.0f;

    std::vector<float> out(n * ambient);
    std::uniform_int_distribution<size_t> pick(0, n_clusters - 1);
    std::vector<float> latent(intrinsic);
    for (size_t i = 0; i < n; ++i) {
        const auto& center = centers[pick(rng)];
        for (size_t j = 0; j < intrinsic; ++j)
            latent[j] = center[j] + jitter(rng);
        for (size_t d = 0; d < ambient; ++d) {
            float v = 0.0f;
            for (size_t j = 0; j < intrinsic; ++j)
                v += latent[j] * proj[j * ambient + d];
            out[i * ambient + d] = v + noise(rng);
        }
    }
    return out;
}

int main(int argc, char** argv) {
    const size_t n   = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 50000;
    const size_t dim = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 128;
    const size_t nq  = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 1000;
    const size_t k   = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 10;

    std::printf("dataset: %zu vectors, dim %zu, %zu queries, k=%zu\n\n",
                n, dim, nq, k);

    const size_t intrinsic = std::max<size_t>(dim / 8, 8);
    const size_t clusters = std::max<size_t>(n / 2000, 8);
    std::printf("data model: %zu intrinsic dims projected to %zu ambient, "
                "%zu clusters\n\n", intrinsic, dim, clusters);
    auto data = make_embedding_like_data(n, dim, intrinsic, clusters, 1);
    auto queries = make_embedding_like_data(nq, dim, intrinsic, clusters, 2);

    // ---- build the brute force ground truth ----
    BruteForce bf(dim, Metric::L2);
    bf.add_batch(data.data(), n);

    std::printf("computing brute force ground truth and baseline throughput\n");
    std::vector<std::set<size_t>> truth(nq);
    auto t0 = Clock::now();
    for (size_t q = 0; q < nq; ++q) {
        auto r = bf.search(&queries[q * dim], k);
        for (auto& [d, id] : r) truth[q].insert(id);
    }
    const double bf_time = seconds_since(t0);
    const double bf_qps = nq / bf_time;
    std::printf("  brute force: %.1f queries/sec (%.2f ms per query)\n\n",
                bf_qps, 1000.0 * bf_time / nq);

    // ---- build the HNSW index ----
    HnswConfig cfg;
    cfg.M = 16;
    cfg.ef_construction = 200;
    cfg.max_elements = n;
    HnswIndex index(dim, Metric::L2, cfg);

    std::printf("building HNSW index (M=%zu, ef_construction=%zu)\n",
                cfg.M, cfg.ef_construction);
    t0 = Clock::now();
    index.add_batch(data.data(), n);
    const double build_time = seconds_since(t0);
    std::printf("  built in %.2f s (%.0f inserts/sec)\n\n",
                build_time, n / build_time);

    // ---- sweep ef and measure recall + throughput ----
    std::printf("%-8s %-10s %-14s %-12s\n", "ef", "recall", "queries/sec", "speedup");
    std::printf("------------------------------------------------\n");

    std::ofstream csv("results/sweep.csv");
    csv << "ef,recall,qps,speedup,ms_per_query\n";

    const std::vector<size_t> ef_values = {10, 20, 40, 60, 100, 150, 200, 400};
    for (size_t ef : ef_values) {
        // warm up
        for (size_t q = 0; q < std::min<size_t>(nq, 50); ++q)
            index.search(&queries[q * dim], k, ef);

        size_t hits = 0;
        auto ts = Clock::now();
        for (size_t q = 0; q < nq; ++q) {
            auto r = index.search(&queries[q * dim], k, ef);
            for (auto& [d, id] : r)
                if (truth[q].count(id)) ++hits;
        }
        const double elapsed = seconds_since(ts);
        const double recall = static_cast<double>(hits) / (nq * k);
        const double qps = nq / elapsed;
        const double speedup = qps / bf_qps;
        const double ms = 1000.0 * elapsed / nq;

        std::printf("%-8zu %-10.4f %-14.0f %-12.1f\n", ef, recall, qps, speedup);
        csv << ef << "," << recall << "," << qps << "," << speedup << ","
            << ms << "\n";
    }
    csv.close();

    std::printf("\nbuild time %.2fs, index holds %zu vectors\n",
                build_time, index.size());
    std::printf("csv written to results/sweep.csv\n");
    return 0;
}
