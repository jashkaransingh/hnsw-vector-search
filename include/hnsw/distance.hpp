// distance.hpp
//
// Distance kernels for vector search. Each metric has a scalar reference
// implementation and an AVX2 + FMA accelerated path selected at compile time.
//
// The AVX2 paths process 8 floats per instruction and use fused multiply-add
// to keep the accumulator in a single register. On a 128-dimensional vector
// that turns roughly 128 scalar mul-add pairs into 16 vector FMAs plus a final
// horizontal reduction, which is where most of the speedup in a search comes
// from since distance computation dominates the inner loop.
//
// When AVX2 is not available the scalar path runs and the results are
// bit-identical up to floating point reassociation, so correctness tests pass
// either way.

#pragma once

#include <cstddef>
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace hnsw {

enum class Metric { L2, InnerProduct, Cosine };

// ---- scalar reference implementations -----------------------------------

inline float l2_sqr_scalar(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline float inner_product_scalar(const float* a, const float* b, size_t dim) {
    float sum = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#if defined(__AVX2__)

// Horizontal sum of an 8-wide vector register.
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);                 // 4 partial sums
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline float l2_sqr_avx2(const float* a, const float* b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    const size_t limit = dim - (dim % 8);
    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);   // acc += diff * diff
    }
    float sum = hsum256(acc);
    for (; i < dim; ++i) {                          // tail
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline float inner_product_avx2(const float* a, const float* b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    const size_t limit = dim - (dim % 8);
    for (; i < limit; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    float sum = hsum256(acc);
    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

#endif  // __AVX2__

// ---- dispatched public functions ----------------------------------------

inline float l2_sqr(const float* a, const float* b, size_t dim) {
#if defined(__AVX2__)
    return l2_sqr_avx2(a, b, dim);
#else
    return l2_sqr_scalar(a, b, dim);
#endif
}

inline float inner_product(const float* a, const float* b, size_t dim) {
#if defined(__AVX2__)
    return inner_product_avx2(a, b, dim);
#else
    return inner_product_scalar(a, b, dim);
#endif
}

// A distance is something we minimize. For inner product and cosine on
// normalized vectors, similarity is maximized, so we return 1 - similarity to
// turn it into a distance that the search can minimize uniformly.
struct DistanceFunction {
    Metric metric;
    size_t dim;

    float operator()(const float* a, const float* b) const {
        switch (metric) {
            case Metric::L2:
                return l2_sqr(a, b, dim);
            case Metric::InnerProduct:
            case Metric::Cosine:
                // assumes vectors are L2-normalized for cosine; for raw inner
                // product the ordering is still correct
                return 1.0f - inner_product(a, b, dim);
        }
        return 0.0f;
    }
};

// Normalize a vector in place to unit L2 length, for cosine similarity use.
inline void normalize(float* v, size_t dim) {
    float norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) norm += v[i] * v[i];
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        const float inv = 1.0f / norm;
        for (size_t i = 0; i < dim; ++i) v[i] *= inv;
    }
}

}  // namespace hnsw
