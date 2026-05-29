// brute_force.hpp
//
// Exact k nearest neighbor search by scanning every vector. This is the ground
// truth the approximate index is measured against. Recall is defined as the
// fraction of the true k nearest (from this brute force search) that the HNSW
// index also returns.
//
// It is also a useful baseline on its own. For small datasets brute force with
// SIMD distance is faster than building and querying a graph, and it is always
// 100% recall. The whole point of HNSW is the regime where brute force gets too
// slow, so having both lets the benchmark show exactly where that crossover is.

#pragma once

#include <algorithm>
#include <cstddef>
#include <queue>
#include <utility>
#include <vector>

#include "distance.hpp"

namespace hnsw {

class BruteForce {
public:
    BruteForce(size_t dim, Metric metric)
        : dim_(dim), dist_{metric, dim} {}

    void add(const float* vec) {
        const size_t id = count_++;
        data_.insert(data_.end(), vec, vec + dim_);
        (void)id;
    }

    void add_batch(const float* vecs, size_t n) {
        data_.insert(data_.end(), vecs, vecs + n * dim_);
        count_ += n;
    }

    size_t size() const { return count_; }

    // Returns ids sorted by ascending distance, closest first.
    std::vector<std::pair<float, size_t>> search(const float* query, size_t k) const {
        std::priority_queue<std::pair<float, size_t>> heap;  // max-heap on distance
        for (size_t i = 0; i < count_; ++i) {
            const float d = dist_(query, &data_[i * dim_]);
            if (heap.size() < k) {
                heap.emplace(d, i);
            } else if (d < heap.top().first) {
                heap.pop();
                heap.emplace(d, i);
            }
        }
        std::vector<std::pair<float, size_t>> result(heap.size());
        size_t idx = heap.size();
        while (!heap.empty()) {
            result[--idx] = heap.top();
            heap.pop();
        }
        return result;
    }

private:
    size_t dim_;
    size_t count_ = 0;
    DistanceFunction dist_;
    std::vector<float> data_;
};

}  // namespace hnsw
