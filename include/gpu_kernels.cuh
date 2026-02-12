#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include "gpu_config.cuh"

__global__ void stage1_binary_ip_kernel_v2(
    const float* __restrict__ d_queries,
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_emb_ids,
    const int*   __restrict__ d_pair_offsets,
    float* __restrict__ d_out_dists,
    size_t max_embs_per_query
) {
    __shared__ float smem_query[PADDED_DIM];  // 512 bytes

    const int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

    // Coalesced load of query vector into shared memory
    const float* q_ptr = d_queries + query_idx * PADDED_DIM;
    #pragma unroll
    for (int i = threadIdx.x; i < PADDED_DIM; i += blockDim.x) {
        smem_query[i] = q_ptr[i];
    }
    __syncthreads();

    const float cb1_sumq = d_cb1_sumq[query_idx];
    const size_t pair_start = d_pair_offsets[query_idx];
    const size_t pair_end   = d_pair_offsets[query_idx + 1];
    const size_t num_embs   = pair_end - pair_start;

    // Grid-stride loop: each thread handles one embedding per iteration
    for (size_t idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         idx < num_embs;
         idx += (size_t)blockDim.x * gridDim.x)
    {
        const size_t emb_id = d_emb_ids[pair_start + idx];
        // Vectorized 128-bit load of binary code (2 x uint64_t)
        const uint64_t* code_ptr =
            (const uint64_t*)(d_one_bit_code + emb_id * CODE_BYTES);
        uint64_t code_regs[NUM_U64];
        code_regs[0] = code_ptr[0];
        code_regs[1] = code_ptr[1];

        float ip = 0.0f;
        #pragma unroll
        for (int blk = 0; blk < NUM_U64; blk++) {
            uint64_t bits = code_regs[blk];
            int base = blk * 64;
            // Skip zero bits using __ffsll intrinsic
            while (bits) {
                int pos = __ffsll(bits) - 1;
                ip += smem_query[base + pos];
                bits &= bits - 1;  // clear lowest set bit
            }
        }
        float dist = (ip - cb1_sumq) * d_one_bit_factor[emb_id];

        d_out_dists[query_idx * max_embs_per_query + idx] = dist;
    }
}

__global__ void stage2_binary_ip_kernel_v2(
    const float* __restrict__ d_queries,       // [Q_DOCLEN * PADDED_DIM]
    const char*  __restrict__ d_one_bit_code,  // [n * CODE_BYTES]
    const float* __restrict__ d_one_bit_factor,// [n]
    const float* __restrict__ d_cb1_sumq,      // [Q_DOCLEN]
    const size_t* __restrict__ d_token_ids,    // [total_tokens]
    float* __restrict__ d_out_dists,           // [Q_DOCLEN * total_tokens]
    size_t total_tokens
) {
    // Load ALL query vectors into shared memory (16 KB)
    __shared__ float smem_queries[Q_DOCLEN * PADDED_DIM];
    // Cache cb1_sumq in shared memory to avoid repeated global reads (128 B)
    __shared__ float smem_cb1_sumq[Q_DOCLEN];

    // Cooperative load: 256 threads load 32*128 = 4096 floats → 16 iterations
    const int total_query_floats = Q_DOCLEN * PADDED_DIM;
    for (int i = threadIdx.x; i < total_query_floats; i += blockDim.x) {
        smem_queries[i] = d_queries[i];
    }
    if (threadIdx.x < Q_DOCLEN) {
        smem_cb1_sumq[threadIdx.x] = d_cb1_sumq[threadIdx.x];
    }
    __syncthreads();

    // Grid-stride loop: each thread processes one token against all queries
    for (size_t tok_idx = threadIdx.x + (size_t)blockIdx.x * blockDim.x;
         tok_idx < total_tokens;
         tok_idx += (size_t)blockDim.x * gridDim.x)
    {
        // Load binary code and factor ONCE per token
        const size_t token_id = d_token_ids[tok_idx];
        const float factor = d_one_bit_factor[token_id];

        // Load binary code into registers (2 x uint64_t)
        const uint64_t* code_ptr =
            (const uint64_t*)(d_one_bit_code + token_id * CODE_BYTES);
        uint64_t code_regs[NUM_U64];
        code_regs[0] = code_ptr[0];
        code_regs[1] = code_ptr[1];

        // Score against all Q_DOCLEN queries
        #pragma unroll
        for (int q = 0; q < Q_DOCLEN; q++) {
            const float* q_smem = smem_queries + q * PADDED_DIM;
            const float cb1_sumq = smem_cb1_sumq[q];

            float ip = 0.0f;
            #pragma unroll
            for (int blk = 0; blk < NUM_U64; blk++) {
                uint64_t bits = code_regs[blk];
                int base = blk * 64;
                // Fully unrolled loop — compiler emits predicated FADD with zero branching.
                // Do NOT replace with __ffsll: data-dependent branching causes warp divergence
                // multiplied 32x across queries, resulting in 3.6x slowdown.
                #pragma unroll
                for (int i = 0; i < 64; i++) {
                    if ((bits >> i) & 1ULL)
                        ip += q_smem[base + i];
                }
            }

            d_out_dists[q * total_tokens + tok_idx] =
                (ip - cb1_sumq) * factor;
        }
    }
}

__global__ void doc_score_kernel(
    const float*  __restrict__ d_token_dists,  // [Q_DOCLEN][total_tokens]
    const size_t* __restrict__ d_candidate_offsets,
    float*        d_doc_scores,
    size_t total_tokens,
    size_t num_candidates
) {
    constexpr int TILE_T = 8;
    __shared__ float tile[Q_DOCLEN * TILE_T];
    __shared__ float max_vals[Q_DOCLEN];
    __shared__ float reduce_buf[256];  // Assuming max 256 threads per block

    size_t cand_idx = blockIdx.x;
    if (cand_idx >= num_candidates) return;

    size_t tok_start = d_candidate_offsets[cand_idx];
    size_t tok_end = d_candidate_offsets[cand_idx + 1];
    size_t num_tokens = tok_end - tok_start;

    // Initialize per-query max values cooperatively (all threads participate)
    for (int j = threadIdx.x; j < Q_DOCLEN; j += blockDim.x) {
        max_vals[j] = -FLT_MAX;
    }
    __syncthreads();

    // Tile loop is the outer loop — ALL threads reach every __syncthreads()
    for (size_t t_base = 0; t_base < num_tokens; t_base += TILE_T) {
        int tile_size = ((size_t)TILE_T < num_tokens - t_base) ? TILE_T : (int)(num_tokens - t_base);

        // Cooperative tile load: ALL threads participate
        __syncthreads();
        for (int idx = threadIdx.x; idx < Q_DOCLEN * tile_size; idx += blockDim.x) {
            int q = idx / tile_size;
            int t_local = idx % tile_size;
            tile[q * TILE_T + t_local] = d_token_dists[q * total_tokens + tok_start + t_base + t_local];
        }
        __syncthreads();

        // Each thread updates max for its assigned queries
        for (size_t j = threadIdx.x; j < Q_DOCLEN; j += blockDim.x) {
            float local_max = max_vals[j];
            for (int t_local = 0; t_local < tile_size; t_local++) {
                local_max = fmaxf(local_max, tile[j * TILE_T + t_local]);
            }
            max_vals[j] = local_max;
        }
    }
    __syncthreads();

    // Sum max values across queries
    float my_sum = 0.0f;
    for (size_t j = threadIdx.x; j < Q_DOCLEN; j += blockDim.x) {
        my_sum += max_vals[j];
    }

    // Block-level reduction
    reduce_buf[threadIdx.x] = my_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce_buf[threadIdx.x] += reduce_buf[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_doc_scores[cand_idx] = reduce_buf[0];
    }
}

__global__ void extract_one_bit_dists_kernel(
    const float*  __restrict__ d_token_dists,     // [Q_DOCLEN][total_tokens] GPU layout
    const size_t* __restrict__ d_candidate_offsets,
    const int*    __restrict__ d_selected_indices,
    float*        d_out_one_bit_dists,             // [token][query] output layout
    const size_t* __restrict__ d_out_offsets,
    size_t total_tokens,
    size_t k
) {
    size_t sel_idx = blockIdx.x;
    if (sel_idx >= k) return;

    size_t cand_idx = d_selected_indices[sel_idx];
    size_t tok_start = d_candidate_offsets[cand_idx];
    size_t tok_end = d_candidate_offsets[cand_idx + 1];
    size_t num_tokens = tok_end - tok_start;
    size_t out_base = d_out_offsets[sel_idx];
    size_t total_elems = num_tokens * Q_DOCLEN;

    // Iterate over output elements linearly for coalesced writes
    // Output layout: [token][query] → out[(out_base + t) * Q_DOCLEN + q]
    // Adjacent threads write adjacent addresses when iterating linearly
    for (size_t i = threadIdx.x; i < total_elems; i += blockDim.x) {
        size_t t_local = i / Q_DOCLEN;
        size_t q_idx = i - t_local * Q_DOCLEN;  // avoid expensive modulo

        // Read from [query][token] layout
        float val = d_token_dists[q_idx * total_tokens + tok_start + t_local];
        // Write to [token][query] layout
        d_out_one_bit_dists[(out_base + t_local) * Q_DOCLEN + q_idx] = val;
    }
}

__global__ void gather_token_ids_kernel(
    const int*    __restrict__ d_candidate_doc_ids,
    const int*    __restrict__ d_doc_ptrs,
    const size_t* __restrict__ d_candidate_offsets,
    size_t*       d_out_token_ids,
    size_t num_candidates
) {
    size_t cand_idx = blockIdx.x;
    if (cand_idx >= num_candidates) return;

    int doc_id = d_candidate_doc_ids[cand_idx];
    int doc_start = d_doc_ptrs[doc_id];
    int doc_end = d_doc_ptrs[doc_id + 1];
    size_t out_offset = d_candidate_offsets[cand_idx];

    for (int t = threadIdx.x; t < (doc_end - doc_start); t += blockDim.x) {
        d_out_token_ids[out_offset + t] = doc_start + t;
    }
}

/**
 * Gather document pointers for top-k docs to compute token counts.
 * Each thread handles one document.
 */
__global__ void gather_doc_lengths_kernel(
    const int* __restrict__ d_topk_doc_ids,
    const int* __restrict__ d_doc_ptrs,
    int*       d_doc_lengths,
    size_t k
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    
    int doc_id = d_topk_doc_ids[idx];
    int doc_len = d_doc_ptrs[doc_id + 1] - d_doc_ptrs[doc_id];
    d_doc_lengths[idx] = doc_len;
}

// Atomic max for floats using compare-and-swap
__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                       __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

/**
 * Atomic max-pooling and aggregation for Stage 1.
 * Each (emb_id, query_idx) → (doc_id, query_idx) → atomic max into doc_query_matrix
 * Then sum across queries for each doc.
 */
__global__ void aggregate_stage1_atomic_kernel_opt(
    const size_t* __restrict__ d_emb_ids,
    const float*  __restrict__ d_emb_dists,       // [Q_DOCLEN * max_embs_per_query] layout
    const int*    __restrict__ d_pair_offsets,
    const int*    __restrict__ d_doc_ids,
    float*        d_doc_query_max,                 // [num_docs * Q_DOCLEN] matrix
    size_t        num_docs,
    size_t        total_pairs,
    size_t        max_embs_per_query               // for indexing into new layout
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    // Find which query this embedding belongs to
    int q_idx = 0;
    #pragma unroll
    for (int q = 0; q < Q_DOCLEN; ++q) {
        if (idx < d_pair_offsets[q + 1]) {
            q_idx = q;
            break;
        }
    }

    size_t emb_id = d_emb_ids[idx];
    int doc_id = d_doc_ids[emb_id];

    // Calculate local pair index within query and use new layout
    int local_pair_idx = idx - d_pair_offsets[q_idx];
    float dist = d_emb_dists[q_idx * max_embs_per_query + local_pair_idx];

    // Bounds check
    if (doc_id >= num_docs) return;

    // Atomic max into doc_query_max[q_idx * num_docs + doc_id] (transposed layout)
    size_t matrix_idx = (size_t)q_idx * num_docs + doc_id;
    atomicMaxFloat(&d_doc_query_max[matrix_idx], dist);
}

/**
 * Sum max distances across queries for each document.
 */
__global__ void sum_doc_scores_kernel(
    const float* __restrict__ d_doc_query_max,  // [num_docs * Q_DOCLEN]
    float*       d_doc_scores,                   // [num_docs]
    size_t       num_docs
) {
    size_t doc_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (doc_id >= num_docs) return;

    float score = 0.0f;
    #pragma unroll
    for (size_t q = 0; q < Q_DOCLEN; ++q) {
        score += d_doc_query_max[q * num_docs + doc_id];  // transposed: coalesced reads
    }
    d_doc_scores[doc_id] = score;
}


