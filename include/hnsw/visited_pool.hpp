// visited_pool.hpp
//
// During a graph search we need to mark nodes as visited so we do not expand
// them twice. The naive approach allocates a fresh boolean array per query and
// zeroes it, which on a large index dominates query time.
//
// Instead we keep one array of version stamps. Each search bumps a global
// counter and a node counts as visited only if its stamp equals the current
// counter. No per-query clearing, no per-query allocation. When the counter
// would overflow we reset the array once, which is rare.
//
// A small pool hands out these arrays so concurrent searches each get their
// own without contending on allocation.

#pragma once

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace hnsw {

class VisitedList {
public:
    explicit VisitedList(size_t n) : size_(n), current_(1) {
        marks_.resize(n, 0);
    }

    void reset() {
        ++current_;
        if (current_ == 0) {                // wrapped around
            std::fill(marks_.begin(), marks_.end(), 0);
            current_ = 1;
        }
    }

    bool visited(size_t id) const { return marks_[id] == current_; }
    void mark(size_t id) { marks_[id] = current_; }

    void resize(size_t n) {
        if (n > size_) {
            marks_.resize(n, 0);
            size_ = n;
        }
    }

private:
    size_t size_;
    uint16_t current_;
    std::vector<uint16_t> marks_;
};

// Pool of reusable VisitedLists so multiple threads do not allocate per query.
class VisitedListPool {
public:
    VisitedListPool(size_t initial_pool, size_t n) : n_(n),
        mu_(std::make_unique<std::mutex>()) {
        for (size_t i = 0; i < initial_pool; ++i) {
            free_lists_.push_back(std::make_unique<VisitedList>(n));
        }
    }

    std::unique_ptr<VisitedList> acquire() {
        std::lock_guard<std::mutex> lock(*mu_);
        if (!free_lists_.empty()) {
            auto v = std::move(free_lists_.back());
            free_lists_.pop_back();
            v->reset();
            v->resize(n_);
            return v;
        }
        auto v = std::make_unique<VisitedList>(n_);
        return v;
    }

    void release(std::unique_ptr<VisitedList> v) {
        std::lock_guard<std::mutex> lock(*mu_);
        free_lists_.push_back(std::move(v));
    }

    void set_size(size_t n) { n_ = n; }

private:
    size_t n_;
    std::unique_ptr<std::mutex> mu_;
    std::vector<std::unique_ptr<VisitedList>> free_lists_;
};

}  // namespace hnsw
