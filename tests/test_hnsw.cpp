// test_hnsw.cpp
//
// Correctness tests. No framework, just asserts and a tiny harness so the
// suite builds with nothing but a compiler. Exit code 0 means all passed.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "hnsw/brute_force.hpp"
#include "hnsw/distance.hpp"
#include "hnsw/hnsw.hpp"

using namespace hnsw;

static int g_failures = 0;
#define CHECK(cond, msg)                                            \
    do {                                                            \
        if (!(cond)) {                                              \
            std::printf("  FAIL: %s\n", msg);                       \
            ++g_failures;                                           \
        }                                                           \
    } while (0)

static std::vector<float> random_vectors(size_t n, size_t dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(n * dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ---- distance kernels ----------------------------------------------------

static void test_distance_kernels() {
    std::printf("test_distance_kernels\n");
    const size_t dim = 130;  // not a multiple of 8, exercises the tail
    auto a = random_vectors(1, dim, 1);
    auto b = random_vectors(1, dim, 2);

    const float l2_s = l2_sqr_scalar(a.data(), b.data(), dim);
    const float l2_v = l2_sqr(a.data(), b.data(), dim);
    CHECK(std::fabs(l2_s - l2_v) < 1e-2f, "L2 scalar vs simd mismatch");

    const float ip_s = inner_product_scalar(a.data(), b.data(), dim);
    const float ip_v = inner_product(a.data(), b.data(), dim);
    CHECK(std::fabs(ip_s - ip_v) < 1e-2f, "inner product scalar vs simd mismatch");

    // distance to self is zero for L2
    CHECK(l2_sqr(a.data(), a.data(), dim) < 1e-4f, "L2 to self should be ~0");
}

// ---- brute force sanity --------------------------------------------------

static void test_brute_force() {
    std::printf("test_brute_force\n");
    const size_t dim = 16, n = 200;
    auto data = random_vectors(n, dim, 3);
    BruteForce bf(dim, Metric::L2);
    bf.add_batch(data.data(), n);
    CHECK(bf.size() == n, "brute force size wrong");

    // a vector should be its own nearest neighbor
    auto res = bf.search(&data[5 * dim], 1);
    CHECK(res.size() == 1, "expected 1 result");
    CHECK(res[0].second == 5, "nearest of a point should be itself");
    CHECK(res[0].first < 1e-4f, "distance to self should be ~0");

    // results sorted ascending by distance
    auto res5 = bf.search(&data[0], 5);
    for (size_t i = 1; i < res5.size(); ++i) {
        CHECK(res5[i - 1].first <= res5[i].first, "brute force not sorted");
    }
}

// ---- hnsw self-retrieval -------------------------------------------------

static void test_hnsw_self_retrieval() {
    std::printf("test_hnsw_self_retrieval\n");
    const size_t dim = 32, n = 500;
    auto data = random_vectors(n, dim, 4);

    HnswConfig cfg;
    cfg.M = 16;
    cfg.ef_construction = 100;
    cfg.max_elements = n;
    HnswIndex index(dim, Metric::L2, cfg);
    index.add_batch(data.data(), n);
    CHECK(index.size() == n, "hnsw size wrong");

    // every point should retrieve itself as the top-1 with high ef
    size_t correct = 0;
    for (size_t i = 0; i < n; ++i) {
        auto res = index.search(&data[i * dim], 1, 64);
        if (!res.empty() && res[0].second == i) ++correct;
    }
    const double rate = static_cast<double>(correct) / n;
    std::printf("  self-retrieval top1 rate: %.3f\n", rate);
    CHECK(rate > 0.98, "self-retrieval should be near perfect");
}

// ---- hnsw recall vs brute force -----------------------------------------

static void test_hnsw_recall() {
    std::printf("test_hnsw_recall\n");
    const size_t dim = 64, n = 2000, nq = 200, k = 10;
    auto data = random_vectors(n, dim, 5);
    auto queries = random_vectors(nq, dim, 6);

    BruteForce bf(dim, Metric::L2);
    bf.add_batch(data.data(), n);

    HnswConfig cfg;
    cfg.M = 16;
    cfg.ef_construction = 200;
    cfg.max_elements = n;
    HnswIndex index(dim, Metric::L2, cfg);
    index.add_batch(data.data(), n);

    size_t hits = 0, total = 0;
    for (size_t q = 0; q < nq; ++q) {
        auto truth = bf.search(&queries[q * dim], k);
        std::set<size_t> truth_ids;
        for (auto& [d, id] : truth) truth_ids.insert(id);

        auto approx = index.search(&queries[q * dim], k, 100);
        for (auto& [d, id] : approx) {
            if (truth_ids.count(id)) ++hits;
        }
        total += k;
    }
    const double recall = static_cast<double>(hits) / total;
    std::printf("  recall@%zu at ef=100: %.3f\n", k, recall);
    CHECK(recall > 0.90, "recall@10 should exceed 0.90 on random gaussian data");
}

// ---- higher ef gives higher recall --------------------------------------

static void test_ef_monotonicity() {
    std::printf("test_ef_monotonicity\n");
    const size_t dim = 48, n = 2000, nq = 100, k = 10;
    auto data = random_vectors(n, dim, 7);
    auto queries = random_vectors(nq, dim, 8);

    BruteForce bf(dim, Metric::L2);
    bf.add_batch(data.data(), n);

    HnswConfig cfg;
    cfg.M = 16;
    cfg.ef_construction = 200;
    cfg.max_elements = n;
    HnswIndex index(dim, Metric::L2, cfg);
    index.add_batch(data.data(), n);

    auto measure = [&](size_t ef) {
        size_t hits = 0, total = 0;
        for (size_t q = 0; q < nq; ++q) {
            auto truth = bf.search(&queries[q * dim], k);
            std::set<size_t> truth_ids;
            for (auto& [d, id] : truth) truth_ids.insert(id);
            auto approx = index.search(&queries[q * dim], k, ef);
            for (auto& [d, id] : approx)
                if (truth_ids.count(id)) ++hits;
            total += k;
        }
        return static_cast<double>(hits) / total;
    };

    const double r_low = measure(10);
    const double r_high = measure(150);
    std::printf("  recall ef=10: %.3f, ef=150: %.3f\n", r_low, r_high);
    CHECK(r_high >= r_low, "higher ef should not reduce recall");
}

// ---- persistence round trip ----------------------------------------------

static void test_save_load() {
    std::printf("test_save_load\n");
    const size_t dim = 24, n = 300, k = 5;
    auto data = random_vectors(n, dim, 9);

    HnswConfig cfg;
    cfg.M = 12;
    cfg.ef_construction = 100;
    cfg.max_elements = n;
    HnswIndex index(dim, Metric::L2, cfg);
    index.add_batch(data.data(), n);

    const char* path = "/tmp/hnsw_test_index.bin";
    index.save(path);
    HnswIndex loaded = HnswIndex::load(path);
    CHECK(loaded.size() == n, "loaded size mismatch");

    // queries should return identical results before and after reload
    bool identical = true;
    for (size_t q = 0; q < 20; ++q) {
        auto a = index.search(&data[q * dim], k, 50);
        auto b = loaded.search(&data[q * dim], k, 50);
        if (a.size() != b.size()) { identical = false; break; }
        for (size_t i = 0; i < a.size(); ++i) {
            if (a[i].second != b[i].second) { identical = false; break; }
        }
    }
    CHECK(identical, "results differ after save/load");
}

// ---- cosine metric on normalized vectors ---------------------------------

static void test_cosine_metric() {
    std::printf("test_cosine_metric\n");
    const size_t dim = 32, n = 500;
    auto data = random_vectors(n, dim, 10);
    for (size_t i = 0; i < n; ++i) normalize(&data[i * dim], dim);

    HnswConfig cfg;
    cfg.M = 16;
    cfg.ef_construction = 100;
    cfg.max_elements = n;
    HnswIndex index(dim, Metric::Cosine, cfg);
    index.add_batch(data.data(), n);

    size_t correct = 0;
    for (size_t i = 0; i < n; ++i) {
        auto res = index.search(&data[i * dim], 1, 64);
        if (!res.empty() && res[0].second == i) ++correct;
    }
    const double rate = static_cast<double>(correct) / n;
    std::printf("  cosine self-retrieval top1: %.3f\n", rate);
    CHECK(rate > 0.98, "cosine self-retrieval should be near perfect");
}

int main() {
    std::printf("running hnsw tests\n\n");
    test_distance_kernels();
    test_brute_force();
    test_hnsw_self_retrieval();
    test_hnsw_recall();
    test_ef_monotonicity();
    test_save_load();
    test_cosine_metric();

    std::printf("\n");
    if (g_failures == 0) {
        std::printf("all tests passed\n");
        return 0;
    }
    std::printf("%d checks failed\n", g_failures);
    return 1;
}
