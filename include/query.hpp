#pragma once

#include <numeric>
#include <cstddef>

struct query_object {
    float* rotated_query;
    float cb1_sumq;
    float cbex_sumq;

    query_object() = default;

    explicit query_object(float* rotated_query, size_t padded_dim, size_t ex_bits): rotated_query(rotated_query) {
        float sumq = std::accumulate(rotated_query, rotated_query + padded_dim, 0.0f);
        cb1_sumq = sumq * ((1 << 1) - 1) / 2.F;
        cbex_sumq = sumq * ((1 << (ex_bits + 1)) - 1) / 2.F;
    }
};