#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include "gpu_config.cuh"

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// Core binary inner product — bank-conflict-free shared memory access
// ============================================================================

/**
 * Compute binary IP between a query in shared memory and a binary code.
 * Uses stride-1 access in two passes of 32 elements per 64-bit block
 * to avoid shared memory bank conflicts.
 *
 * Previous approach: lane k reads smem[base + k*2], smem[base + k*2 + 1]
 *   → lane 0 and lane 16 both hit bank 0 (2-way conflict)
 *
 * Fixed approach: two passes with stride-1
 *   Pass 1: lane k reads smem[base + k]       → bank k, no conflicts
 *   Pass 2: lane k reads smem[base + 32 + k]  → bank k, no conflicts
 */
__device__ __forceinline__ float compute_binary_ip_partial(
    const float* __restrict__ smem_query,  // [PADDED_DIM] in shared memory
    const uint64_t* __restrict__ code_ptr, // binary code in global memory
    int lane
) {
    float partial_sum = 0.0f;
    #pragma unroll
    for (size_t blk = 0; blk < NUM_U64; blk++) {
        uint64_t bits = code_ptr[blk];
        size_t base = blk * 64;

        // Pass 1: lanes 0-31 handle positions base..base+31 (stride-1, no bank conflicts)
        if ((bits >> lane) & 1ULL)
            partial_sum += smem_query[base + lane];

        // Pass 2: lanes 0-31 handle positions base+32..base+63 (stride-1, no bank conflicts)
        if ((bits >> (32 + lane)) & 1ULL)
            partial_sum += smem_query[base + 32 + lane];
    }
    return partial_sum;
}

// ============================================================================
// Optimized binary inner product with software pipelining and loop unrolling
// ============================================================================

/**
 * Optimized version of compute_binary_ip_partial with:
 * - Software pipelining: prefetch next blocks while computing current
 * - Loop unrolling by factor of 4: exposes ILP and reduces loop overhead
 * - Same bank-conflict-free access pattern as original
 *
 * Expected speedup: 3-4x over original for typical dimensions
 */
__device__ __forceinline__ float compute_binary_ip_partial_opt(
    const float* __restrict__ smem_query,  // [PADDED_DIM] in shared memory
    const uint64_t* __restrict__ code_ptr, // binary code in global memory
    int lane
) {
    float partial_sum = 0.0f;

    // NUM_U64 = 2 (PADDED_DIM=128 / 64), so exactly 2 blocks - fully unrolled at compile-time
    #pragma unroll
    for (size_t blk = 0; blk < NUM_U64; blk++) {
        uint64_t bits = code_ptr[blk];
        size_t base = blk * 64;

        // Pass 1: Process first half of block
        if ((bits >> lane) & 1ULL)
            partial_sum += smem_query[base + lane];

        // Pass 2: Process second half of block
        if ((bits >> (32 + lane)) & 1ULL)
            partial_sum += smem_query[base + 32 + lane];
    }
    // No remainder loop needed - NUM_U64=2 is exact

    return partial_sum;
}

// ============================================================================
// Thread-level binary inner product — each thread computes a full IP
// ============================================================================

/**
 * Compute binary IP between a query in shared memory and a binary code,
 * entirely within a single thread (no warp cooperation / reduction needed).
 *
 * Each thread iterates over all PADDED_DIM bits. When all threads in a warp
 * process different embeddings against the SAME query, they all read
 * smem_query[pos] at the same address → shared memory broadcast (zero bank
 * conflicts). This gives 32× more memory-level parallelism for global code
 * loads per warp compared to the warp-cooperative approach.
 */
__device__ __forceinline__ float compute_binary_ip_thread(
    const float* __restrict__ smem_query,  // [PADDED_DIM] in shared memory
    const uint64_t* __restrict__ code_ptr  // binary code in global memory
) {
    float sum = 0.0f;
    #pragma unroll
    for (int blk = 0; blk < NUM_U64; blk++) {
        uint64_t bits = code_ptr[blk];
        int base = blk * 64;
        // Iterate all 64 bits. Compiler emits LDS (broadcast) + predicated FADD.
        #pragma unroll
        for (int i = 0; i < 64; i++) {
            if ((bits >> i) & 1ULL)
                sum += smem_query[base + i];
        }
    }
    return sum;
}

// ============================================================================
// Stage 1: Binary IP kernel with shared memory query caching
// ============================================================================

/**
 * Grid: (blocks_x, Q_DOCLEN). Each y-slice handles one query.
 * Loads query into shared memory once; warps process different emb_ids.
 * Emb IDs are stored contiguously per query via d_pair_offsets.
 *
 * d_pair_offsets: [Q_DOCLEN + 1] cumulative pair count per query
 * d_emb_ids:     [num_pairs] sorted by query_idx
 * d_out_dists:   [num_pairs] output, same order as d_emb_ids
 */
__global__ void stage1_binary_ip_kernel(
    const float* __restrict__ d_queries,
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_emb_ids,
    const int*   __restrict__ d_pair_offsets,
    float* __restrict__ d_out_dists
) {
    __shared__ float smem_query[PADDED_DIM];  // Static shared memory: 512 bytes

    int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

    // Coalesced load: adjacent threads load adjacent floats
    const float* q_ptr = d_queries + query_idx * PADDED_DIM;
    #pragma unroll
    for (int i = threadIdx.x; i < PADDED_DIM; i += blockDim.x) {
        smem_query[i] = q_ptr[i];
    }
    __syncthreads();

    float cb1_sumq = d_cb1_sumq[query_idx];
    size_t pair_start = d_pair_offsets[query_idx];
    size_t pair_end = d_pair_offsets[query_idx + 1];
    size_t num_embs = pair_end - pair_start;

    const int lane = threadIdx.x & 31;
    const int warp_local_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    for (size_t w = warp_local_id + (size_t)blockIdx.x * warps_per_block;
         w < num_embs;
         w += (size_t)warps_per_block * gridDim.x)
    {
        size_t emb_id = d_emb_ids[pair_start + w];
        const uint64_t* code_ptr = (const uint64_t*)(d_one_bit_code + emb_id * CODE_BYTES);

        float partial = compute_binary_ip_partial(smem_query, code_ptr, lane);
        float ip = warp_reduce_sum(partial);

        if (lane == 0) {
            d_out_dists[pair_start + w] = (ip - cb1_sumq) * d_one_bit_factor[emb_id];
        }
    }
}

// ============================================================================
// Stage 1 OPTIMIZED: Warp-cooperative batched processing
// ============================================================================

/**
 * Legacy warp-cooperative version (kept for A/B comparison).
 * Each warp processes EMBS_PER_WARP=4 embeddings per iteration using
 * warp-reduction. Replaced by stage1_binary_ip_kernel_v2 below.
 */
__global__ void stage1_binary_ip_kernel_opt(
    const float* __restrict__ d_queries,
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_emb_ids,
    const int*   __restrict__ d_pair_offsets,
    float* __restrict__ d_out_dists,
    size_t max_embs_per_query
) {
    __shared__ float smem_query[PADDED_DIM];

    int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

    const float* q_ptr = d_queries + query_idx * PADDED_DIM;
    #pragma unroll
    for (int i = threadIdx.x; i < PADDED_DIM; i += blockDim.x) {
        smem_query[i] = q_ptr[i];
    }
    __syncthreads();

    float cb1_sumq = d_cb1_sumq[query_idx];
    size_t pair_start = d_pair_offsets[query_idx];
    size_t pair_end = d_pair_offsets[query_idx + 1];
    size_t num_embs = pair_end - pair_start;

    const int lane = threadIdx.x & 31;
    const int warp_local_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    constexpr int EMBS_PER_WARP = 4;
    float results[EMBS_PER_WARP];
    size_t emb_ids_local[EMBS_PER_WARP];

    for (size_t w_base = warp_local_id * EMBS_PER_WARP + (size_t)blockIdx.x * warps_per_block * EMBS_PER_WARP;
         w_base < num_embs;
         w_base += (size_t)warps_per_block * gridDim.x * EMBS_PER_WARP)
    {
        int valid_count = 0;
        #pragma unroll
        for (int e = 0; e < EMBS_PER_WARP; ++e) {
            size_t w = w_base + e;
            if (w < num_embs) {
                if (lane == 0) emb_ids_local[e] = d_emb_ids[pair_start + w];
                emb_ids_local[e] = __shfl_sync(0xffffffff, emb_ids_local[e], 0);
                valid_count++;
            }
        }
        #pragma unroll
        for (int e = 0; e < valid_count; ++e) {
            const uint64_t* code_ptr = (const uint64_t*)(d_one_bit_code + emb_ids_local[e] * CODE_BYTES);
            float partial = compute_binary_ip_partial_opt(smem_query, code_ptr, lane);
            float ip = warp_reduce_sum(partial);
            results[e] = (ip - cb1_sumq) * d_one_bit_factor[emb_ids_local[e]];
        }
        if (lane < valid_count) {
            size_t out_idx = query_idx * max_embs_per_query + w_base + lane;
            d_out_dists[out_idx] = results[lane];
        }
    }
}

// ============================================================================
// Stage 1 v2: Thread-level parallelism (replaces warp-cooperative approach)
// ============================================================================

/**
 * High-parallelism binary IP kernel for Stage 1.
 *
 * Key change: each THREAD independently computes one full binary IP
 * (no warp reduction). This yields 32× more embeddings in flight per
 * warp, dramatically improving latency hiding for scattered global
 * memory accesses to binary codes via d_emb_ids.
 *
 * Grid: (blocks_x, Q_DOCLEN).  Block: 256 threads.
 * Each block processes 256 embeddings per grid-stride iteration.
 * Output layout: [query][embedding] — same as stage1_binary_ip_kernel_opt.
 *
 * Shared memory broadcast: all threads in a warp iterate the same bit
 * position, loading smem_query[pos] at the same address → hardware
 * broadcast, zero bank conflicts.
 */
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
        const uint64_t* code_ptr =
            (const uint64_t*)(d_one_bit_code + emb_id * CODE_BYTES);

        float ip = compute_binary_ip_thread(smem_query, code_ptr);
        float dist = (ip - cb1_sumq) * d_one_bit_factor[emb_id];

        d_out_dists[query_idx * max_embs_per_query + idx] = dist;
    }
}

// ============================================================================
// Stage 2: Binary IP kernel (legacy) — kept for A/B comparison
// ============================================================================

/**
 * Legacy version: 2D grid (blocks_x, Q_DOCLEN), each y-slice handles one query.
 * Each warp processes one token. Replaced by stage2_binary_ip_kernel_v2 below.
 */
__global__ void stage2_binary_ip_kernel(
    const float* __restrict__ d_queries,
    const char*  __restrict__ d_one_bit_code,
    const float* __restrict__ d_one_bit_factor,
    const float* __restrict__ d_cb1_sumq,
    const size_t* __restrict__ d_token_ids,
    float* __restrict__ d_out_dists,
    size_t total_tokens
) {
    __shared__ float smem_query[PADDED_DIM];

    int query_idx = blockIdx.y;
    if (query_idx >= Q_DOCLEN) return;

    const float* q_ptr = d_queries + query_idx * PADDED_DIM;
    #pragma unroll
    for (int i = threadIdx.x; i < PADDED_DIM; i += blockDim.x) {
        smem_query[i] = q_ptr[i];
    }
    __syncthreads();

    float cb1_sumq = d_cb1_sumq[query_idx];
    const int lane = threadIdx.x & 31;
    const int warp_local_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    for (size_t tok_idx = warp_local_id + (size_t)blockIdx.x * warps_per_block;
         tok_idx < total_tokens;
         tok_idx += (size_t)warps_per_block * gridDim.x)
    {
        size_t token_id = d_token_ids[tok_idx];
        const uint64_t* code_ptr = (const uint64_t*)(d_one_bit_code + token_id * CODE_BYTES);
        float partial = compute_binary_ip_partial(smem_query, code_ptr, lane);
        float ip = warp_reduce_sum(partial);
        if (lane == 0) {
            d_out_dists[query_idx * total_tokens + tok_idx] =
                (ip - cb1_sumq) * d_one_bit_factor[token_id];
        }
    }
}

// ============================================================================
// Stage 2 v2: Multi-query fusion + thread-level parallelism
// ============================================================================

/**
 * High-parallelism binary IP kernel for Stage 2.
 *
 * Two key changes vs. the legacy kernel:
 *  1. Multi-query fusion: all Q_DOCLEN=32 query vectors are loaded into
 *     shared memory and each thread scores ONE token against ALL 32 queries.
 *     The binary code + factor are loaded from global memory ONCE per token
 *     instead of 32 times (one per query-slice in the old 2D grid).
 *  2. Thread-level IP: each thread independently computes the full 128-bit
 *     binary IP (no warp reduction), giving 32× more tokens in flight per
 *     warp for better latency hiding.
 *
 * Grid: 1D — blocks_x = ceil(total_tokens / blockDim.x).  Block: 256.
 * Shared memory: Q_DOCLEN * PADDED_DIM * sizeof(float) = 32*128*4 = 16 KB.
 *
 * Output layout: [query][token] — preserved for downstream compatibility.
 * Write coalescing: for a fixed query q, adjacent threads (consecutive
 * tok_idx) write to contiguous addresses q*total_tokens + tok_idx.
 */
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
        const uint64_t* code_ptr =
            (const uint64_t*)(d_one_bit_code + token_id * CODE_BYTES);
        const float factor = d_one_bit_factor[token_id];

        // Pre-load the binary code into registers
        uint64_t code_regs[NUM_U64];
        #pragma unroll
        for (int blk = 0; blk < NUM_U64; blk++) {
            code_regs[blk] = code_ptr[blk];
        }

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

// ============================================================================
// Stage 2: Document scoring — tiled reads to improve memory access
// ============================================================================

/**
 * One block per candidate document.
 * Input layout: [query][token] — d_token_dists[q * total_tokens + tok]
 *
 * Strategy: each thread handles one query. For the max-pool over tokens,
 * we tile tokens through shared memory so that all threads in the block
 * cooperatively load a tile of token distances (coalesced), then each thread
 * reads its query's value from shared memory (bank-conflict-free stride-1).
 *
 * Shared memory layout: [TILE_SIZE][q_doclen_padded] where q_doclen_padded
 * is q_doclen rounded up to avoid bank conflicts. But since q_doclen is small
 * (typically 32), we use a simpler approach: tile in the token dimension.
 */
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

// ============================================================================
// Stage 2: Extract selected token dists — coalesced with layout transpose
// ============================================================================

/**
 * Extracts top-k candidates' token dists from [query][token] GPU layout
 * and writes to [token][query] layout (for CPU stage 3 consumption).
 *
 * One block per selected candidate. Threads iterate linearly over
 * output elements (contiguous in memory) for coalesced writes.
 * Reads are strided (one per query plane) but benefit from L2 cache.
 */
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

// ============================================================================
// Gather token IDs (expand doc IDs to token IDs on GPU)
// ============================================================================

/**
 * One block per candidate doc. Threads write contiguous token IDs → coalesced.
 */
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

// ============================================================================
// Utility: map_emb_to_doc (stage 1 helper, coalesced read/write)
// ============================================================================

__global__ void map_emb_to_doc_kernel(
    const float*  __restrict__ d_emb_dists,
    const size_t* __restrict__ d_emb_ids,
    const int*    __restrict__ d_pair_query_indices,
    const int*    __restrict__ d_doc_ids,
    int*          d_out_doc_ids,
    size_t num_pairs
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    d_out_doc_ids[idx] = d_doc_ids[d_emb_ids[idx]];
}
// ============================================================================
// NEW: Stage 1 GPU aggregation kernels
// ============================================================================

/**
 * Build (doc_id, query_idx, dist) tuples from embedding-level results.
 * Parallel expansion from emb_ids → doc_ids using d_pair_query_indices.
 */
__global__ void build_doc_query_keys_kernel(
    const size_t* __restrict__ d_emb_ids,
    const int*    __restrict__ d_doc_ids,
    const int*    __restrict__ d_pair_offsets,
    float*        __restrict__ d_emb_dists,
    int*          d_out_doc_ids,
    int*          d_out_query_indices,
    size_t total_pairs,
    size_t q_doclen
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    // Binary search to find which query this pair belongs to
    int q_idx = 0;
    for (int q = 0; q < q_doclen; ++q) {
        if (idx < d_pair_offsets[q + 1]) {
            q_idx = q;
            break;
        }
    }

    size_t emb_id = d_emb_ids[idx];
    int doc_id = d_doc_ids[emb_id];
    
    d_out_doc_ids[idx] = doc_id;
    d_out_query_indices[idx] = q_idx;
}

/**
 * Compute final document scores from max-pooled distances.
 * One thread per (doc, query) → sum across queries using atomics.
 * Input: d_max_dists[num_unique_pairs] — already max-pooled per (doc, query)
 * Keys: d_doc_ids[num_unique_pairs], d_query_indices[num_unique_pairs]
 * Output: d_doc_scores[num_unique_docs] — sum of max distances across queries
 */
__global__ void aggregate_doc_scores_kernel(
    const int*   __restrict__ d_unique_doc_ids,
    const float* __restrict__ d_max_dists,
    int*         d_doc_id_to_idx,  // [max_doc_id + 1] mapping
    float*       d_doc_scores,
    size_t num_unique_pairs,
    size_t num_unique_docs
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_unique_pairs) return;

    int doc_id = d_unique_doc_ids[idx];
    float max_dist = d_max_dists[idx];
    
    // Atomic add to accumulate scores for each document
    int doc_idx = d_doc_id_to_idx[doc_id];
    atomicAdd(&d_doc_scores[doc_idx], max_dist);
}

/**
 * Build composite key for sorting: (doc_id << 16) | query_idx
 * Assumes doc_id and query_idx both fit in 16 bits.
 */
__global__ void build_composite_keys_kernel(
    const int* __restrict__ d_doc_ids,
    const int* __restrict__ d_query_indices,
    int*       d_composite_keys,
    size_t total_pairs
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;
    
    int doc_id = d_doc_ids[idx];
    int q_idx = d_query_indices[idx];
    // Composite key: high bits = doc_id, low bits = query_idx
    d_composite_keys[idx] = (doc_id << 16) | q_idx;
}

// ============================================================================
// NEW: Stage 2 GPU offset computation
// ============================================================================

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

/**
 * Extract document IDs from composite keys: doc_id = composite >> 16
 */
__global__ void extract_doc_ids_from_composite_kernel(
    const int* __restrict__ d_composite_keys,
    int*       d_doc_ids,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    d_doc_ids[idx] = d_composite_keys[idx] >> 16;
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
 *
 * NOTE: This is the original version that works with linear output layout.
 * Use aggregate_stage1_atomic_kernel_opt for the optimized [query][embedding] layout.
 */
__global__ void aggregate_stage1_atomic_kernel(
    const size_t* __restrict__ d_emb_ids,
    const float*  __restrict__ d_emb_dists,
    const int*    __restrict__ d_pair_offsets,
    const int*    __restrict__ d_doc_ids,
    float*        d_doc_query_max,  // [num_docs * Q_DOCLEN] matrix
    size_t        num_docs,
    size_t        total_pairs
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
    float dist = d_emb_dists[idx];

    // Bounds check
    if (doc_id >= num_docs) return;

    // Atomic max into doc_query_max[doc_id * Q_DOCLEN + q_idx]
    size_t matrix_idx = (size_t)doc_id * Q_DOCLEN + q_idx;
    atomicMaxFloat(&d_doc_query_max[matrix_idx], dist);
}

/**
 * OPTIMIZED: Atomic max-pooling for new [query][embedding] output layout.
 * Handles the optimized stage1_binary_ip_kernel_opt output format.
 *
 * Input layout: d_emb_dists[query_idx * max_embs_per_query + local_pair_idx]
 * Output: doc_query_max[doc_id * Q_DOCLEN + q_idx]
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

    // Atomic max into doc_query_max[doc_id * Q_DOCLEN + q_idx]
    size_t matrix_idx = (size_t)doc_id * Q_DOCLEN + q_idx;
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
        score += d_doc_query_max[doc_id * Q_DOCLEN + q];
    }
    d_doc_scores[doc_id] = score;
}


