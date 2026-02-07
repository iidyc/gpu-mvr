#pragma once

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>

// Prevent Eigen from adding __host__ __device__ annotations to its functions.
// Eigen is only used on the CPU side; without this, nvcc flags warning #20014-D
// for Eigen internals that call host-only STL functions.
#define EIGEN_NO_CUDA

#include <vector>
#include <queue>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cfloat>

#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "quantization.hpp"
#include "estimator.hpp"
#include "query.hpp"
#include "ivf_pg.hpp"
#include "gpu_kernels.cuh"

using namespace rabitqlib;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

struct gpu_mvr_index {
    // Scalar metadata
    size_t n;
    size_t d;
    size_t n_clusters;
    size_t ex_bits;
    size_t padded_dim_;
    size_t num_docs;

    int max_doc_len;       // for workspace sizing
    int max_cluster_size;  // for workspace sizing

    // CPU-side data
    Rotator<float>* rotator_;
    IVF_PG* ivf;
    std::vector<char> ex_code_;          // CPU only - too large for GPU
    std::vector<float> one_bit_factor_;  // CPU copy for stage 3
    std::vector<float> ex_factor_;       // CPU only - stage 3 only
    std::vector<int> doc_ids_;           // CPU copy
    std::vector<int> doc_ptrs_;          // CPU copy for stage 3
    float (*ip_func_)(const float*, const uint8_t*, size_t);

    // GPU persistent data (copied once at construction)
    char*  d_one_bit_code_;     // [n * padded_dim_ / 8]
    float* d_one_bit_factor_;   // [n]
    int*   d_doc_ids_;          // [n]
    int*   d_doc_ptrs_;         // [num_docs + 1]

    // Pre-allocated GPU workspace (sized for worst-case per search)
    struct Workspace {
        // Query data (shared across stages)
        float* d_queries;       // [max_q_doclen * padded_dim]
        float* d_cb1_sumq;      // [max_q_doclen]

        // Stage 1 workspace
        size_t* d_emb_ids;      // [max_stage1_pairs]
        int*    d_pair_offsets;  // [max_q_doclen + 1]
        float*  d_emb_dists;    // [max_stage1_pairs]
        int*    d_pair_doc_ids; // [max_stage1_pairs]
        int*    d_pair_query_indices; // [max_stage1_pairs]

        // Stage 2 workspace
        size_t* d_token_ids;         // [max_stage2_tokens]
        size_t* d_candidate_offsets; // [max_stage2_candidates + 1]
        float*  d_token_dists;       // [max_stage2_tokens * max_q_doclen]
        float*  d_doc_scores;        // [max_stage2_candidates]

        // Extract workspace
        size_t* d_selected_indices;  // [max_stage2_k]
        size_t* d_out_offsets;       // [max_stage2_k + 1]
        float*  d_out_one_bit_dists; // [max_stage2_k_tokens * max_q_doclen]

        // Stage 1 CAGRA batched search workspace
        float*        d_cagra_dists;     // [max_q_doclen * max_nprobe]
        faiss::idx_t* d_cagra_labels;    // [max_q_doclen * max_nprobe]

        // Pinned host memory for fast H2D/D2H
        float*  h_pinned_queries;
        float*  h_pinned_cb1_sumq;
        size_t* h_pinned_emb_ids;
        int*    h_pinned_pair_offsets;
        float*  h_pinned_dists;
        faiss::idx_t* h_pinned_cagra_labels; // [max_q_doclen * max_nprobe]

        size_t max_q_doclen;
        size_t max_nprobe;
        size_t max_stage1_pairs;
        size_t max_stage2_candidates;
        size_t max_stage2_tokens;
        size_t max_stage2_k;
        size_t max_stage2_k_tokens;
    } ws_;

    // Construct from file
    gpu_mvr_index(const std::string& filename, const std::vector<int>& doc_lens,
                  size_t max_q_doclen = 64, size_t max_nprobe = 256) {
        std::ifstream inf(filename, std::ios::binary);
        inf.read((char*)&n, sizeof(size_t));
        inf.read((char*)&d, sizeof(size_t));
        inf.read((char*)&n_clusters, sizeof(size_t));
        inf.read((char*)&ex_bits, sizeof(size_t));
        inf.read((char*)&padded_dim_, sizeof(size_t));

        std::vector<char> one_bit_code(n * padded_dim_ / 8);
        ex_code_.resize(n * padded_dim_ * ex_bits / 8);
        one_bit_factor_.resize(n);
        ex_factor_.resize(n);

        inf.read(one_bit_code.data(), one_bit_code.size());
        inf.read(ex_code_.data(), ex_code_.size());
        inf.read((char*)one_bit_factor_.data(), n * sizeof(float));
        inf.read((char*)ex_factor_.data(), n * sizeof(float));
        inf.close();

        rotator_ = choose_rotator<float>(d, RotatorType::FhtKacRotator, padded_dim_);
        std::ifstream rot_in("rotator.bin", std::ios::binary);
        rotator_->load(rot_in);
        rot_in.close();

        ip_func_ = select_excode_ipfunc(ex_bits);

        ivf = new IVF_PG(n_clusters, d, PGType::CAGRA);
        ivf->load(filename);
        max_cluster_size = ivf->max_cluster_size();

        set_doc_mapping(doc_lens);

        // Allocate persistent GPU data
        size_t code_bytes = n * padded_dim_ / 8;
        CUDA_CHECK(cudaMalloc(&d_one_bit_code_, code_bytes));
        CUDA_CHECK(cudaMalloc(&d_one_bit_factor_, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_doc_ids_, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_doc_ptrs_, (num_docs + 1) * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_one_bit_code_, one_bit_code.data(), code_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_one_bit_factor_, one_bit_factor_.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_doc_ids_, doc_ids_.data(), n * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_doc_ptrs_, doc_ptrs_.data(), (num_docs + 1) * sizeof(int), cudaMemcpyHostToDevice));

        // Allocate workspace
        allocate_workspace(max_q_doclen, max_nprobe);
    }

    void set_doc_mapping(const std::vector<int>& doc_lens) {
        num_docs = doc_lens.size();
        doc_ptrs_.resize(num_docs + 1, 0);
        for (size_t i = 0; i < num_docs; ++i) {
            max_doc_len = std::max(max_doc_len, doc_lens[i]);
            doc_ptrs_[i + 1] = doc_ptrs_[i] + doc_lens[i];
        }
        doc_ids_.resize(n);
        for (size_t i = 0; i < num_docs; ++i) {
            for (size_t j = 0; j < doc_lens[i]; ++j) {
                doc_ids_[doc_ptrs_[i] + j] = i;
            }
        }
    }

    void allocate_workspace(size_t max_q_doclen, size_t max_nprobe) {
        // Estimate worst-case sizes
        // Stage 1: nprobe clusters * max_cluster_size * q_doclen pairs
        ws_.max_q_doclen = max_q_doclen;
        ws_.max_nprobe = max_nprobe;
        ws_.max_stage1_pairs = max_nprobe * max_cluster_size * max_q_doclen;
        ws_.max_stage2_candidates = 20000;  // k_rank_cluster
        // Estimate avg ~100 tokens per doc
        ws_.max_stage2_tokens = ws_.max_stage2_candidates * max_doc_len;
        ws_.max_stage2_k = 1000;  // k_rank_all_tokens
        ws_.max_stage2_k_tokens = ws_.max_stage2_k * max_doc_len;

        // Query workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_queries, max_q_doclen * padded_dim_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_cb1_sumq, max_q_doclen * sizeof(float)));

        // Stage 1 workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_emb_ids, ws_.max_stage1_pairs * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pair_offsets, (max_q_doclen + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_emb_dists, ws_.max_stage1_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pair_doc_ids, ws_.max_stage1_pairs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pair_query_indices, ws_.max_stage1_pairs * sizeof(int)));

        // Stage 2 workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_token_ids, ws_.max_stage2_tokens * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_candidate_offsets, (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_token_dists, ws_.max_stage2_tokens * max_q_doclen * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_doc_scores, ws_.max_stage2_candidates * sizeof(float)));

        // Extract workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_selected_indices, ws_.max_stage2_k * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_out_offsets, (ws_.max_stage2_k + 1) * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_out_one_bit_dists, ws_.max_stage2_k_tokens * max_q_doclen * sizeof(float)));

        // CAGRA batched search workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_cagra_dists, max_q_doclen * max_nprobe * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_cagra_labels, max_q_doclen * max_nprobe * sizeof(faiss::idx_t)));

        // Pinned host memory
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_queries, max_q_doclen * padded_dim_ * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_cb1_sumq, max_q_doclen * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_emb_ids, ws_.max_stage1_pairs * sizeof(size_t)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_pair_offsets, (max_q_doclen + 1) * sizeof(int)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_dists, ws_.max_stage2_k_tokens * max_q_doclen * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_cagra_labels, max_q_doclen * max_nprobe * sizeof(faiss::idx_t)));
    }

    size_t doc_len(size_t doc_id) const {
        return doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id];
    }

    // ======================== SEARCH PIPELINE ========================

    std::vector<size_t> search(const float* queries, size_t q_doclen, size_t k, size_t nprobe) {
        int k_rank_cluster = 20000;
        int k_rank_all_tokens = 1000;

        // Step 1: Rotate queries on CPU into pinned memory
        for (size_t i = 0; i < q_doclen; ++i) {
            rotator_->rotate(&queries[i * d], &ws_.h_pinned_queries[i * padded_dim_]);
        }

        // Build query objects on CPU
        std::vector<query_object> query_objs(q_doclen);
        for (size_t i = 0; i < q_doclen; ++i) {
            query_objs[i] = query_object(&ws_.h_pinned_queries[i * padded_dim_], padded_dim_, ex_bits);
            ws_.h_pinned_cb1_sumq[i] = query_objs[i].cb1_sumq;
        }

        // Upload from pinned memory (faster than pageable)
        CUDA_CHECK(cudaMemcpy(ws_.d_queries, ws_.h_pinned_queries,
                              q_doclen * padded_dim_ * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ws_.d_cb1_sumq, ws_.h_pinned_cb1_sumq,
                              q_doclen * sizeof(float), cudaMemcpyHostToDevice));

        // Stage 1: GPU
        std::vector<size_t> stage1_doc_ids;
        rank_cluster_dists_gpu(query_objs.data(), q_doclen, nprobe, k_rank_cluster, stage1_doc_ids);

        // Stage 2: GPU → outputs to CPU for stage 3
        std::vector<size_t> stage2_doc_ids;
        std::vector<float> one_bit_dists;
        rank_all_tokens_1bit_gpu(q_doclen, stage1_doc_ids, k_rank_all_tokens,
                                 stage2_doc_ids, one_bit_dists);

        // Stage 3: CPU
        std::vector<size_t> result;
        rank_all_tokens_exbits_cpu(query_objs.data(), q_doclen, stage2_doc_ids,
                                   one_bit_dists, k, result);

        return result;
    }

    // ======================== STAGE 1: GPU ========================
    // Sparse approach: only process and sort found docs, not all num_docs.

    void rank_cluster_dists_gpu(
        query_object* h_query_objs,
        size_t q_doclen, size_t nprobe, size_t k,
        std::vector<size_t>& output_ids
    ) {
        // 1. Batched CAGRA search: one call for all q_doclen queries on GPU
        //    ws_.d_queries is already on GPU from the caller.
        ivf->search_batch_gpu(ws_.d_queries, q_doclen, nprobe,
                              ws_.d_cagra_dists, ws_.d_cagra_labels);

        // Copy cluster labels back to CPU (one transfer instead of q_doclen)
        CUDA_CHECK(cudaMemcpy(ws_.h_pinned_cagra_labels, ws_.d_cagra_labels,
                              q_doclen * nprobe * sizeof(faiss::idx_t), cudaMemcpyDeviceToHost));

        // Expand cluster IDs to embedding IDs via inv_list
        ws_.h_pinned_pair_offsets[0] = 0;
        size_t total_pairs = 0;
        std::vector<std::vector<size_t>> per_query_ids(q_doclen);

        for (size_t j = 0; j < q_doclen; ++j) {
            for (size_t p = 0; p < nprobe; ++p) {
                faiss::idx_t cluster_id = ws_.h_pinned_cagra_labels[j * nprobe + p];
                if (cluster_id < 0) continue;
                size_t start = ivf->cluster_pos[cluster_id];
                size_t end = ivf->cluster_pos[cluster_id + 1];
                for (size_t i = start; i < end; ++i) {
                    per_query_ids[j].push_back(ivf->inv_list[i]);
                }
            }
            total_pairs += per_query_ids[j].size();
            ws_.h_pinned_pair_offsets[j + 1] = total_pairs;
        }

        if (total_pairs == 0) return;

        // Flatten into pinned memory contiguously by query
        for (size_t j = 0; j < q_doclen; ++j) {
            memcpy(&ws_.h_pinned_emb_ids[ws_.h_pinned_pair_offsets[j]],
                   per_query_ids[j].data(),
                   per_query_ids[j].size() * sizeof(size_t));
        }

        // Upload emb_ids and pair_offsets
        CUDA_CHECK(cudaMemcpy(ws_.d_emb_ids, ws_.h_pinned_emb_ids,
                              total_pairs * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ws_.d_pair_offsets, ws_.h_pinned_pair_offsets,
                              (q_doclen + 1) * sizeof(int), cudaMemcpyHostToDevice));

        // 2. Compute 1-bit distances using shared-memory kernel
        //    Grid: (blocks_x, q_doclen) — each row of blocks handles one query
        size_t max_embs_per_query = 0;
        for (size_t j = 0; j < q_doclen; ++j)
            max_embs_per_query = std::max(max_embs_per_query, per_query_ids[j].size());

        int threads_per_block = 256;
        int warps_per_block = threads_per_block / 32;
        int blocks_x = (max_embs_per_query + warps_per_block - 1) / warps_per_block;
        size_t smem_size = padded_dim_ * sizeof(float);

        dim3 grid(blocks_x, q_doclen);
        stage1_binary_ip_kernel<<<grid, threads_per_block, smem_size>>>(
            ws_.d_queries, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
            ws_.d_emb_ids, ws_.d_pair_offsets, ws_.d_emb_dists,
            padded_dim_, q_doclen
        );
        CUDA_CHECK(cudaGetLastError());

        // 3. Sparse aggregation on CPU — faster than dense GPU approach for typical sizes
        //    Copy distances back and aggregate
        std::vector<float> h_emb_dists(total_pairs);
        CUDA_CHECK(cudaMemcpy(h_emb_dists.data(), ws_.d_emb_dists,
                              total_pairs * sizeof(float), cudaMemcpyDeviceToHost));

        // Aggregate: for each (query, doc), track max distance; then sum across queries
        std::unordered_map<int, std::vector<float>> doc_max_dists;  // doc_id -> [q_doclen] max dists

        for (size_t j = 0; j < q_doclen; ++j) {
            size_t start = ws_.h_pinned_pair_offsets[j];
            size_t end = ws_.h_pinned_pair_offsets[j + 1];
            for (size_t p = start; p < end; ++p) {
                size_t emb_id = ws_.h_pinned_emb_ids[p];
                int doc_id = doc_ids_[emb_id];
                auto& maxd = doc_max_dists[doc_id];
                if (maxd.empty()) maxd.assign(q_doclen, 0.0F);
                maxd[j] = std::max(maxd[j], h_emb_dists[p]);
            }
        }

        // Score and top-k
        std::priority_queue<std::pair<float, size_t>> max_heap;
        for (auto& [doc_id, maxd] : doc_max_dists) {
            float score = 0.0f;
            for (size_t j = 0; j < q_doclen; ++j) score += maxd[j];
            max_heap.emplace(score, doc_id);
        }

        size_t actual_k = std::min(k, doc_max_dists.size());
        output_ids.reserve(actual_k);
        for (size_t i = 0; i < actual_k && !max_heap.empty(); ++i) {
            output_ids.push_back(max_heap.top().second);
            max_heap.pop();
        }
    }

    // ======================== STAGE 2: GPU ========================

    void rank_all_tokens_1bit_gpu(
        size_t q_doclen,
        std::vector<size_t>& input_ids,
        size_t k,
        std::vector<size_t>& output_ids,
        std::vector<float>& one_bit_dists
    ) {
        size_t num_candidates = input_ids.size();
        if (num_candidates == 0) return;

        // 1. Build candidate token list on CPU
        std::vector<size_t> candidate_offsets(num_candidates + 1, 0);
        for (size_t i = 0; i < num_candidates; ++i) {
            candidate_offsets[i + 1] = candidate_offsets[i] + doc_len(input_ids[i]);
        }
        size_t total_tokens = candidate_offsets[num_candidates];

        std::vector<size_t> token_ids(total_tokens);
        for (size_t i = 0; i < num_candidates; ++i) {
            size_t doc_id = input_ids[i];
            size_t offset = candidate_offsets[i];
            for (size_t t = 0; t < doc_len(doc_id); ++t) {
                token_ids[offset + t] = doc_ptrs_[doc_id] + t;
            }
        }

        // 2. Upload to pre-allocated GPU workspace
        CUDA_CHECK(cudaMemcpy(ws_.d_token_ids, token_ids.data(),
                              total_tokens * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ws_.d_candidate_offsets, candidate_offsets.data(),
                              (num_candidates + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

        // 3. Compute all (query, token) 1-bit distances with shared-memory kernel
        //    Grid: (blocks_x, q_doclen)
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / 32;
        int blocks_x = (total_tokens + warps_per_block - 1) / warps_per_block;
        size_t smem_size = padded_dim_ * sizeof(float);

        dim3 grid(blocks_x, q_doclen);
        stage2_binary_ip_kernel<<<grid, threads_per_block, smem_size>>>(
            ws_.d_queries, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
            ws_.d_token_ids, ws_.d_token_dists,
            padded_dim_, q_doclen, total_tokens
        );
        CUDA_CHECK(cudaGetLastError());

        // 4. Compute doc scores with shared-memory reduction
        int score_threads = 128;
        // Round up to power of 2 for clean reduction
        while (score_threads < (int)q_doclen && score_threads < 256) score_threads *= 2;
        score_threads = std::min(score_threads, 256);
        constexpr int TILE_T = 8;  // must match doc_score_kernel
        size_t score_smem = (q_doclen * (TILE_T + 1) + score_threads) * sizeof(float);

        doc_score_kernel<<<num_candidates, score_threads, score_smem>>>(
            ws_.d_token_dists, ws_.d_candidate_offsets, ws_.d_doc_scores,
            q_doclen, total_tokens, num_candidates
        );
        CUDA_CHECK(cudaGetLastError());

        // 5. Top-k via thrust sort on pre-allocated workspace
        thrust::device_vector<size_t> d_cand_indices(num_candidates);
        thrust::sequence(d_cand_indices.begin(), d_cand_indices.end());

        thrust::device_ptr<float> scores_ptr(ws_.d_doc_scores);
        thrust::sort_by_key(scores_ptr, scores_ptr + num_candidates,
                            d_cand_indices.begin(), thrust::greater<float>());

        size_t actual_k = std::min(k, num_candidates);

        // 6. Extract only top-k token dists on GPU, then copy to CPU
        std::vector<size_t> h_top_k_indices(actual_k);
        thrust::copy(d_cand_indices.begin(), d_cand_indices.begin() + actual_k,
                     h_top_k_indices.begin());

        // Build output offsets for selected docs
        std::vector<size_t> out_offsets(actual_k + 1, 0);
        for (size_t i = 0; i < actual_k; ++i) {
            size_t cand_idx = h_top_k_indices[i];
            out_offsets[i + 1] = out_offsets[i] +
                (candidate_offsets[cand_idx + 1] - candidate_offsets[cand_idx]);
        }
        size_t total_selected_tokens = out_offsets[actual_k];

        // Upload selection metadata to GPU
        CUDA_CHECK(cudaMemcpy(ws_.d_selected_indices, h_top_k_indices.data(),
                              actual_k * sizeof(size_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(ws_.d_out_offsets, out_offsets.data(),
                              (actual_k + 1) * sizeof(size_t), cudaMemcpyHostToDevice));

        // Extract on GPU
        extract_one_bit_dists_kernel<<<actual_k, 256>>>(
            ws_.d_token_dists, ws_.d_candidate_offsets,
            ws_.d_selected_indices, ws_.d_out_one_bit_dists,
            ws_.d_out_offsets, q_doclen, total_tokens, actual_k
        );
        CUDA_CHECK(cudaGetLastError());

        // Copy only selected dists to pinned host memory
        size_t copy_size = total_selected_tokens * q_doclen;
        CUDA_CHECK(cudaMemcpy(ws_.h_pinned_dists, ws_.d_out_one_bit_dists,
                              copy_size * sizeof(float), cudaMemcpyDeviceToHost));

        // Build output
        output_ids.resize(actual_k);
        one_bit_dists.resize(copy_size);

        for (size_t i = 0; i < actual_k; ++i) {
            output_ids[i] = input_ids[h_top_k_indices[i]];
        }
        memcpy(one_bit_dists.data(), ws_.h_pinned_dists, copy_size * sizeof(float));
    }

    // ======================== STAGE 3: CPU ========================

    void rank_all_tokens_exbits_cpu(
        query_object* queries,
        size_t q_doclen,
        std::vector<size_t>& input_ids,
        std::vector<float>& one_bit_dists,
        size_t k,
        std::vector<size_t>& output_ids
    ) {
        std::vector<size_t> candidate_doc_ptrs(input_ids.size() + 1);
        size_t total_tokens = 0;
        for (size_t i = 0; i < input_ids.size(); ++i) {
            total_tokens += doc_len(input_ids[i]);
            candidate_doc_ptrs[i + 1] = total_tokens;
        }
        std::priority_queue<std::pair<float, size_t>> max_heap;
        for (size_t idx = 0; idx < input_ids.size(); ++idx) {
            size_t doc_id = input_ids[idx];
            float doc_score = 0.0F;
            for (size_t j = 0; j < q_doclen; ++j) {
                float max_token_score = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < doc_len(doc_id); ++i) {
                    size_t tid = doc_ptrs_[doc_id] + i;
                    float dist = distance_ex_bits(
                        queries + j,
                        &ex_code_[tid * padded_dim_ * ex_bits / 8],
                        ex_bits,
                        ip_func_,
                        one_bit_dists[(candidate_doc_ptrs[idx] + i) * q_doclen + j],
                        one_bit_factor_[tid],
                        ex_factor_[tid],
                        padded_dim_
                    );
                    max_token_score = std::max(max_token_score, dist);
                }
                doc_score += max_token_score;
            }
            max_heap.emplace(doc_score, doc_id);
        }
        for (size_t i = 0; i < k && !max_heap.empty(); ++i) {
            output_ids.push_back(max_heap.top().second);
            max_heap.pop();
        }
    }

    ~gpu_mvr_index() {
        // Free persistent GPU data
        CUDA_CHECK(cudaFree(d_one_bit_code_));
        CUDA_CHECK(cudaFree(d_one_bit_factor_));
        CUDA_CHECK(cudaFree(d_doc_ids_));
        CUDA_CHECK(cudaFree(d_doc_ptrs_));

        // Free workspace
        CUDA_CHECK(cudaFree(ws_.d_queries));
        CUDA_CHECK(cudaFree(ws_.d_cb1_sumq));
        CUDA_CHECK(cudaFree(ws_.d_emb_ids));
        CUDA_CHECK(cudaFree(ws_.d_pair_offsets));
        CUDA_CHECK(cudaFree(ws_.d_emb_dists));
        CUDA_CHECK(cudaFree(ws_.d_pair_doc_ids));
        CUDA_CHECK(cudaFree(ws_.d_pair_query_indices));
        CUDA_CHECK(cudaFree(ws_.d_token_ids));
        CUDA_CHECK(cudaFree(ws_.d_candidate_offsets));
        CUDA_CHECK(cudaFree(ws_.d_token_dists));
        CUDA_CHECK(cudaFree(ws_.d_doc_scores));
        CUDA_CHECK(cudaFree(ws_.d_selected_indices));
        CUDA_CHECK(cudaFree(ws_.d_out_offsets));
        CUDA_CHECK(cudaFree(ws_.d_out_one_bit_dists));
        CUDA_CHECK(cudaFree(ws_.d_cagra_dists));
        CUDA_CHECK(cudaFree(ws_.d_cagra_labels));

        // Free pinned memory
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_queries));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_cb1_sumq));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_emb_ids));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_dists));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_cagra_labels));

        delete rotator_;
        delete ivf;
    }
};
