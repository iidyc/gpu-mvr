#pragma once

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/quantization/pack_excode.hpp"

using namespace rabitqlib;

constexpr std::array<float, 9> kTightStart = {
    0,
    0.15,
    0.20,
    0.52,
    0.59,
    0.71,
    0.75,
    0.77,
    0.81,
};

// pack 0/1 data to usigned integer
inline void pack_binary(
    const int* __restrict__ binary_code, uint64_t* __restrict__ compact_code, size_t length
) {
    constexpr size_t kTypeBits = sizeof(uint64_t) * 8;

    for (size_t i = 0; i < length; i += kTypeBits) {
        uint64_t cur = 0;
        for (size_t j = 0; j < kTypeBits; ++j) {
            cur |= (static_cast<uint64_t>(binary_code[i + j]) << (kTypeBits - 1 - j));
        }
        *compact_code = cur;
        ++compact_code;
    }
}

inline void encode_one_bit(const float* rotated_data, size_t dim, uint64_t* one_bit_code, float* factor) {
    std::vector<int> binary_code(dim);
    ConstRowMajorArrayMap<float> data_arr(rotated_data, 1, dim);
    // x_u is the binary code
    RowMajorArrayMap<int> x_u(binary_code.data(), 1, static_cast<long>(dim));
    x_u = (data_arr > 0).template cast<int>();
    pack_binary(binary_code.data(), one_bit_code, dim);

    // compute factor: 1 / <o_bar, o>
    float cb = -((1 << 1) - 1) / 2.F;
    RowMajorArray<float> xu_cb = x_u.template cast<float>() + cb;
    float ip_obar_o = dot_product<float>(xu_cb.data(), rotated_data, dim);
    *factor = 1.0F / ip_obar_o;
}

inline double best_rescale_factor(const float* o_abs, size_t dim, size_t ex_bits) {
    constexpr double kEps = 1e-5;
    constexpr int kNEnum = 10;
    double max_o = *std::max_element(o_abs, o_abs + dim);

    double t_end = static_cast<double>(((1 << ex_bits) - 1) + kNEnum) / max_o;
    double t_start = t_end * kTightStart[ex_bits];

    std::vector<int> cur_o_bar(dim);
    double sqr_denominator = static_cast<double>(dim) * 0.25;
    double numerator = 0;

    for (size_t i = 0; i < dim; ++i) {
        int cur = static_cast<int>((t_start * o_abs[i]) + kEps);
        cur_o_bar[i] = cur;
        sqr_denominator += cur * cur + cur;
        numerator += (cur + 0.5) * o_abs[i];
    }

    std::priority_queue<
        std::pair<double, size_t>,
        std::vector<std::pair<double, size_t>>,
        std::greater<>>
        next_t;

    for (size_t i = 0; i < dim; ++i) {
        next_t.emplace(static_cast<double>(cur_o_bar[i] + 1) / o_abs[i], i);
    }

    double max_ip = 0;
    double t = 0;

    while (!next_t.empty()) {
        double cur_t = next_t.top().first;
        size_t update_id = next_t.top().second;
        next_t.pop();

        cur_o_bar[update_id]++;
        int update_o_bar = cur_o_bar[update_id];
        sqr_denominator += 2 * update_o_bar;
        numerator += o_abs[update_id];

        double cur_ip = numerator / std::sqrt(sqr_denominator);
        if (cur_ip > max_ip) {
            max_ip = cur_ip;
            t = cur_t;
        }

        if (update_o_bar < (1 << ex_bits) - 1) {
            double t_next = static_cast<double>(update_o_bar + 1) / o_abs[update_id];
            if (t_next < t_end) {
                next_t.emplace(t_next, update_id);
            }
        }
    }

    return t;
}

inline void quantize_ex(const float* rotated_data, uint8_t* ex_code, size_t dim, size_t ex_bits) {
    ConstRowMajorArrayMap<float> data_arr(rotated_data, 1, dim);
    RowMajorArray<float> abs_data = data_arr.rowwise().normalized().abs();

    constexpr double kEps = 1e-5;
    double t = best_rescale_factor(abs_data.data(), dim, ex_bits);
    double ipnorm = 0;

    std::vector<int> tmp_code(dim);
    for (size_t i = 0; i < dim; i++) {
        // compute and store code
        tmp_code[i] = static_cast<int>((t * abs_data.data()[i]) + kEps);
        if (tmp_code[i] >= (1 << ex_bits)) {
            tmp_code[i] = (1 << ex_bits) - 1;
        }
        ex_code[i] = static_cast<uint8_t>(tmp_code[i]);
    }

    // revert codes for negative dims
    int32_t mask = (1 << ex_bits) - 1;
    for (size_t j = 0; j < dim; ++j) {
        if (rotated_data[j] < 0) {
            uint8_t tmp = ex_code[j];
            ex_code[j] = (~tmp) & mask;
        }
    }
}

inline void encode_ex_bits(const float* rotated_data, size_t dim, size_t ex_bits, uint8_t* compact_ex_code, float* factor) {
    std::vector<uint8_t> ex_code(dim);
    quantize_ex(rotated_data, ex_code.data(), dim, ex_bits);

    // compute factor: 1 / <o_bar, o>
    RowMajorArray<int> total_code =
        RowMajorArrayMap<uint8_t>(ex_code.data(), 1, dim).template cast<int>();
    for (size_t i = 0; i < dim; ++i) {
        total_code(0, i) += static_cast<int>(rotated_data[i] >= 0) << ex_bits;
    }
    float cb = -(static_cast<float>(1 << ex_bits) - 0.5F);
    RowMajorArray<float> xu_cb = total_code.template cast<float>() + cb;
    float ip_obar_o = dot_product<float>(xu_cb.data(), rotated_data, dim);
    *factor = 1.0F / ip_obar_o;

    quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
        ex_code.data(), compact_ex_code, dim, ex_bits
    );
}
