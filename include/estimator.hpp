#pragma once

#include "rabitqlib/utils/space.hpp"
#include "query.hpp"

using namespace rabitqlib;

float distance_one_bit(query_object* query, const char* binary_code, float one_bit_factor, size_t padded_dim) {
    float ip_x0_qr = mask_ip_x0_q(query->rotated_query, (uint64_t*)binary_code, padded_dim);
    return (ip_x0_qr - query->cb1_sumq) * one_bit_factor;
}

float distance_ex_bits(
    query_object* query, 
    const char* ex_code,
    size_t ex_bits,
    float (*ip_func_)(const float*, const uint8_t*, size_t), 
    float one_bit_dist, 
    float one_bit_factor, 
    float ex_factor, 
    size_t padded_dim
) {
    float ip_ex_qr = ip_func_(query->rotated_query, (uint8_t*)ex_code, padded_dim);
    float ip_yubar_qr = (1 << ex_bits) * (one_bit_dist / one_bit_factor + query->cb1_sumq) + ip_ex_qr;
    return (ip_yubar_qr - query->cbex_sumq) * ex_factor;
}