// hnsw.hpp
//
// Hierarchical Navigable Small World graph index for approximate nearest
// neighbor search, after Malkov and Yashunin (2016).
//
// The structure is a stack of proximity graphs. The bottom layer (0) contains
// every point and has short-range links. Each layer above is exponentially
// sparser and carries long-range links. A search starts at the single entry
// point on the top layer, greedily walks toward the query, drops a layer, and
// repeats. The long-range links at the top cover distance fast, the dense
// bottom layer refines. It behaves like a skip list generalized to a graph.
//
// Three things make or break this implementation.
//
//   1. Layer assignment. Each new point gets a random max level drawn from a
//      geometric distribution with parameter mL = 1/ln(M). This is what makes
//      the upper layers sparse in the right proportion.
//
//   2. The neighbor selection heuristic. When connecting a new point, picking
//      the M closest candidates produces clustered links and a graph that is
//      hard to navigate. The heuristic instead prefers a candidate only if it
//      is closer to the new point than to any already-selected neighbor, which
//      spreads links across directions and keeps the graph connected and
//      navigable. This is the single most important detail.
//
//   3. The search beam. ef controls how many candidates the search keeps in
//      flight. Larger ef means higher recall and slower queries. It is the
//      main knob for the recall versus latency tradeoff.

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>

#include "distance.hpp"
#include "visited_pool.hpp"

namespace hnsw {

struct HnswConfig {
    size_t M = 16;                // links per node on layers > 0
    size_t ef_construction = 200; // candidate list size during build
    size_t max_elements = 0;      // capacity, 0 means grow as needed
    uint64_t seed = 42;
};

class HnswIndex {
public:
    HnswIndex(size_t dim, Metric metric, HnswConfig cfg = {})
        : dim_(dim),
          metric_(metric),
          dist_{metric, dim},
          M_(cfg.M),
          M_max_(cfg.M),
          M_max0_(cfg.M * 2),       // layer 0 gets double the links
          ef_construction_(std::max<size_t>(cfg.ef_construction, cfg.M)),
          mL_(1.0 / std::log(1.0 * cfg.M)),
          rng_(cfg.seed),
          visited_pool_(1, std::max<size_t>(cfg.max_elements, 1)) {
        if (cfg.max_elements > 0) reserve(cfg.max_elements);
    }

    size_t size() const { return cur_count_; }
    size_t dim() const { return dim_; }

    // Diagnostic snapshot of the graph structure. Useful for verifying the
    // index built sanely: average layer-0 degree should be close to M_max0,
    // and the level histogram should decay geometrically.
    struct Stats {
        size_t count;
        int max_level;
        double avg_degree_layer0;
        size_t min_degree_layer0;
        size_t isolated_nodes;   // nodes with zero layer-0 links
        std::vector<size_t> level_histogram;
    };

    Stats stats() const {
        Stats s{};
        s.count = cur_count_;
        s.max_level = max_level_;
        s.level_histogram.assign(max_level_ + 1, 0);
        size_t total_deg = 0;
        s.min_degree_layer0 = static_cast<size_t>(-1);
        for (size_t i = 0; i < cur_count_; ++i) {
            s.level_histogram[levels_[i]]++;
            const size_t deg = link_lists_[i][0].size();
            total_deg += deg;
            s.min_degree_layer0 = std::min(s.min_degree_layer0, deg);
            if (deg == 0) s.isolated_nodes++;
        }
        s.avg_degree_layer0 = cur_count_ ? static_cast<double>(total_deg) / cur_count_ : 0.0;
        return s;
    }

    void reserve(size_t n) {
        data_.reserve(n * dim_);
        levels_.reserve(n);
        link_lists_.reserve(n);
        visited_pool_.set_size(n);
    }

    // Insert a vector. Returns its assigned internal id.
    size_t add(const float* vec) {
        std::unique_lock<std::mutex> structure_lock(*global_mu_);

        const size_t id = cur_count_++;
        data_.insert(data_.end(), vec, vec + dim_);

        const int level = random_level();
        levels_.push_back(level);
        link_lists_.emplace_back(level + 1);   // layers 0..level

        visited_pool_.set_size(cur_count_);

        if (entry_point_ == kNone) {
            entry_point_ = id;
            max_level_ = level;
            return id;
        }

        size_t curr = entry_point_;
        float curr_dist = distance(id, curr);

        // Greedy descent through layers above the new point's top level.
        for (int lc = max_level_; lc > level; --lc) {
            bool changed = true;
            while (changed) {
                changed = false;
                const auto& neighbors = link_lists_[curr][lc];
                for (size_t n : neighbors) {
                    const float d = distance(id, n);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr = n;
                        changed = true;
                    }
                }
            }
        }

        // From the new point's top level down to 0, find neighbors and connect.
        for (int lc = std::min(level, max_level_); lc >= 0; --lc) {
            auto candidates = search_layer(id, curr, ef_construction_, lc);
            // candidates is a max-heap on distance; convert and select
            curr = candidates.empty() ? curr : best_of(candidates);

            // A new node connects to M neighbors at every layer. M_max / M_max0
            // are only the caps used when pruning an existing node whose list
            // overflowed from back-links.
            auto selected = select_neighbors_heuristic(id, candidates, M_);

            // Link new point to selected
            link_lists_[id][lc] = selected;

            // Link back, and prune the neighbor's list if it overflows
            for (size_t neighbor : selected) {
                auto& neighbor_links = link_lists_[neighbor][lc];
                neighbor_links.push_back(id);
                const size_t m_max = (lc == 0) ? M_max0_ : M_max_;
                if (neighbor_links.size() > m_max) {
                    prune_neighbors(neighbor, neighbor_links, m_max, lc);
                }
            }
        }

        if (level > max_level_) {
            max_level_ = level;
            entry_point_ = id;
        }
        return id;
    }

    void add_batch(const float* vecs, size_t n) {
        reserve(cur_count_ + n);
        for (size_t i = 0; i < n; ++i) add(vecs + i * dim_);
    }

    // Search for the k nearest neighbors of query. ef controls recall/speed.
    std::vector<std::pair<float, size_t>> search(const float* query, size_t k,
                                                 size_t ef) const {
        if (entry_point_ == kNone) return {};
        ef = std::max(ef, k);

        size_t curr = entry_point_;
        float curr_dist = dist_(query, &data_[curr * dim_]);

        for (int lc = max_level_; lc > 0; --lc) {
            bool changed = true;
            while (changed) {
                changed = false;
                for (size_t n : link_lists_[curr][lc]) {
                    const float d = dist_(query, &data_[n * dim_]);
                    if (d < curr_dist) {
                        curr_dist = d;
                        curr = n;
                        changed = true;
                    }
                }
            }
        }

        auto top = search_layer_query(query, curr, ef, 0);

        // top is a max-heap on distance; keep the k smallest
        while (top.size() > k) top.pop();
        std::vector<std::pair<float, size_t>> result(top.size());
        size_t idx = top.size();
        while (!top.empty()) {
            result[--idx] = top.top();
            top.pop();
        }
        return result;
    }

    // ---- persistence -----------------------------------------------------

    void save(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot open " + path + " for writing");
        write_pod(out, dim_);
        write_pod(out, static_cast<int>(metric_));
        write_pod(out, M_);
        write_pod(out, ef_construction_);
        write_pod(out, cur_count_);
        write_pod(out, entry_point_);
        write_pod(out, max_level_);
        out.write(reinterpret_cast<const char*>(data_.data()),
                  data_.size() * sizeof(float));
        for (size_t i = 0; i < cur_count_; ++i) {
            write_pod(out, levels_[i]);
            for (int lc = 0; lc <= levels_[i]; ++lc) {
                const auto& links = link_lists_[i][lc];
                write_pod(out, links.size());
                out.write(reinterpret_cast<const char*>(links.data()),
                          links.size() * sizeof(size_t));
            }
        }
    }

    static HnswIndex load(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open " + path + " for reading");
        size_t dim;
        int metric_int;
        read_pod(in, dim);
        read_pod(in, metric_int);
        HnswConfig cfg;
        read_pod(in, cfg.M);
        read_pod(in, cfg.ef_construction);
        HnswIndex idx(dim, static_cast<Metric>(metric_int), cfg);

        read_pod(in, idx.cur_count_);
        read_pod(in, idx.entry_point_);
        read_pod(in, idx.max_level_);
        idx.data_.resize(idx.cur_count_ * dim);
        in.read(reinterpret_cast<char*>(idx.data_.data()),
                idx.data_.size() * sizeof(float));
        idx.levels_.resize(idx.cur_count_);
        idx.link_lists_.resize(idx.cur_count_);
        for (size_t i = 0; i < idx.cur_count_; ++i) {
            read_pod(in, idx.levels_[i]);
            idx.link_lists_[i].resize(idx.levels_[i] + 1);
            for (int lc = 0; lc <= idx.levels_[i]; ++lc) {
                size_t n;
                read_pod(in, n);
                idx.link_lists_[i][lc].resize(n);
                in.read(reinterpret_cast<char*>(idx.link_lists_[i][lc].data()),
                        n * sizeof(size_t));
            }
        }
        idx.visited_pool_.set_size(idx.cur_count_);
        return idx;
    }

private:
    static constexpr size_t kNone = static_cast<size_t>(-1);
    using DistId = std::pair<float, size_t>;       // (distance, id)

    float distance(size_t a, size_t b) const {
        return dist_(&data_[a * dim_], &data_[b * dim_]);
    }

    int random_level() {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        double r = uniform(rng_);
        // avoid log(0)
        r = std::max(r, 1e-12);
        return static_cast<int>(-std::log(r) * mL_);
    }

    // Pick the closest id out of a max-heap of candidates.
    size_t best_of(const std::priority_queue<DistId>& heap) const {
        // heap top is the farthest; we need the closest, so scan a copy
        auto copy = heap;
        DistId best = {std::numeric_limits<float>::max(), kNone};
        while (!copy.empty()) {
            if (copy.top().first < best.first) best = copy.top();
            copy.pop();
        }
        return best.second;
    }

    // Search one layer during construction, starting from entry, returning a
    // max-heap of up to ef nearest candidates (by distance to point q_id).
    std::priority_queue<DistId> search_layer(size_t q_id, size_t entry,
                                             size_t ef, int layer) {
        auto visited = visited_pool_.acquire();
        visited->mark(entry);

        const float d_entry = distance(q_id, entry);
        // candidates: min-heap (closest first) using negated distance in a max-heap
        std::priority_queue<DistId, std::vector<DistId>, std::greater<DistId>> candidates;
        std::priority_queue<DistId> result;   // max-heap, farthest on top
        candidates.emplace(d_entry, entry);
        result.emplace(d_entry, entry);

        while (!candidates.empty()) {
            auto [cd, c] = candidates.top();
            candidates.pop();
            if (cd > result.top().first && result.size() >= ef) break;

            for (size_t n : link_lists_[c][layer]) {
                if (visited->visited(n)) continue;
                visited->mark(n);
                const float dn = distance(q_id, n);
                if (dn < result.top().first || result.size() < ef) {
                    candidates.emplace(dn, n);
                    result.emplace(dn, n);
                    if (result.size() > ef) result.pop();
                }
            }
        }
        visited_pool_.release(std::move(visited));
        return result;
    }

    // Same beam search but distances are to an external query vector.
    std::priority_queue<DistId> search_layer_query(const float* query,
                                                   size_t entry, size_t ef,
                                                   int layer) const {
        auto visited = visited_pool_.acquire();
        visited->mark(entry);

        const float d_entry = dist_(query, &data_[entry * dim_]);
        std::priority_queue<DistId, std::vector<DistId>, std::greater<DistId>> candidates;
        std::priority_queue<DistId> result;
        candidates.emplace(d_entry, entry);
        result.emplace(d_entry, entry);

        while (!candidates.empty()) {
            auto [cd, c] = candidates.top();
            candidates.pop();
            if (cd > result.top().first) break;

            for (size_t n : link_lists_[c][layer]) {
                if (visited->visited(n)) continue;
                visited->mark(n);
                const float dn = dist_(query, &data_[n * dim_]);
                if (dn < result.top().first || result.size() < ef) {
                    candidates.emplace(dn, n);
                    result.emplace(dn, n);
                    if (result.size() > ef) result.pop();
                }
            }
        }
        visited_pool_.release(std::move(visited));
        return result;
    }

    // The neighbor selection heuristic (Algorithm 4 from the paper). Given
    // candidates (max-heap by distance to q_id), choose a diverse subset: keep
    // a candidate only if it is closer to q than to every neighbor already
    // kept. This can return fewer than m, and that is intentional. The diverse
    // subset is what gives the graph long-range links and makes it navigable.
    // Padding the list back to m with the rejected near-duplicates (which an
    // earlier version did) collapses the graph toward a plain kNN graph and
    // tanks search efficiency, the true nearest neighbors stay reachable but
    // only after exploring 10x more nodes.
    std::vector<size_t> select_neighbors_heuristic(
            size_t q_id, const std::priority_queue<DistId>& candidates,
            size_t m) {
        (void)q_id;  // distances to q are already carried in candidates
        if (candidates.size() <= m) {
            // nothing to prune, return all candidate ids
            std::vector<size_t> all;
            auto copy = candidates;
            while (!copy.empty()) { all.push_back(copy.top().second); copy.pop(); }
            return all;
        }

        // Process candidates closest-first.
        std::vector<DistId> working;
        working.reserve(candidates.size());
        {
            auto copy = candidates;
            while (!copy.empty()) { working.push_back(copy.top()); copy.pop(); }
        }
        std::sort(working.begin(), working.end(),
                  [](const DistId& a, const DistId& b) { return a.first < b.first; });

        std::vector<size_t> result;
        result.reserve(m);
        for (const auto& [d_to_q, cand] : working) {
            if (result.size() >= m) break;
            bool keep = true;
            for (size_t chosen : result) {
                if (distance(cand, chosen) < d_to_q) {  // closer to a chosen than to q
                    keep = false;
                    break;
                }
            }
            if (keep) result.push_back(cand);
        }
        return result;
    }

    // When a node's neighbor list overflows after a back-link, re-select the
    // best m_max using the same heuristic so the graph stays navigable.
    void prune_neighbors(size_t node, std::vector<size_t>& links,
                         size_t m_max, int layer) {
        (void)layer;  // links already belong to this layer; param kept for clarity
        std::priority_queue<DistId> heap;
        for (size_t n : links) heap.emplace(distance(node, n), n);
        links = select_neighbors_heuristic(node, heap, m_max);
    }

    template <typename T>
    static void write_pod(std::ofstream& out, const T& v) {
        out.write(reinterpret_cast<const char*>(&v), sizeof(T));
    }
    template <typename T>
    static void read_pod(std::ifstream& in, T& v) {
        in.read(reinterpret_cast<char*>(&v), sizeof(T));
    }

    size_t dim_;
    Metric metric_;
    DistanceFunction dist_;

    size_t M_, M_max_, M_max0_, ef_construction_;
    double mL_;

    std::vector<float> data_;                       // flat row-major vectors
    std::vector<int> levels_;                       // max level per node
    // link_lists_[node][layer] = neighbor ids
    std::vector<std::vector<std::vector<size_t>>> link_lists_;

    size_t cur_count_ = 0;
    size_t entry_point_ = kNone;
    int max_level_ = -1;

    mutable std::mt19937_64 rng_;
    mutable VisitedListPool visited_pool_;
    std::unique_ptr<std::mutex> global_mu_ = std::make_unique<std::mutex>();
};

}  // namespace hnsw
