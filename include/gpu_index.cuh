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
#include <cub/cub.cuh>

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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>

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

// === Producer-Consumer Pipeline for Stage 2 → 3 overlap ===

struct ConsumerBatch {
    std::vector<size_t> doc_ids;
    std::vector<float> one_bit_dists;   // [token][query] layout
    std::vector<size_t> doc_ptrs;       // prefix sum, size doc_ids.size()+1
    bool done = false;
};

struct PipelineQueue {
    std::deque<ConsumerBatch> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;

    void push(ConsumerBatch&& batch) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(batch));
        cv_.notify_one();
    }

    ConsumerBatch pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]{ return !queue_.empty(); });
        ConsumerBatch batch = std::move(queue_.front());
        queue_.pop_front();
        return batch;
    }
};

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
        int*    d_selected_indices;  // [max_stage2_k]
        size_t* d_out_offsets;       // [max_stage2_k + 1]
        float*  d_out_one_bit_dists; // [max_stage2_k_tokens * max_q_doclen]

        // Stage 1 CAGRA batched search workspace
        float*        d_cagra_dists;     // [max_q_doclen * max_nprobe]
        faiss::idx_t* d_cagra_labels;    // [max_q_doclen * max_nprobe]

        // === NEW: GPU-only Stage 1 aggregation workspace ===
        int*    d_sorted_doc_ids;        // [max_stage1_pairs] for segmented reduction
        int*    d_sorted_query_indices;  // [max_stage1_pairs]
        float*  d_sorted_dists;          // [max_stage1_pairs]
        int*    d_unique_doc_ids;        // [estimated_num_docs]
        int*    d_doc_offsets;           // [estimated_num_docs + 1]
        float*  d_stage1_doc_scores;     // [estimated_num_docs]
        float*  d_doc_query_max;         // [estimated_num_docs * max_q_doclen] matrix
        int*    d_num_unique_docs;       // [1] output count
        void*   d_cub_temp_storage;      // CUB temporary buffer
        size_t  cub_temp_storage_bytes;  // Size of CUB buffer

        // === NEW: GPU top-k workspace ===
        float*  d_topk_scores;           // [max_stage2_candidates]
        int*    d_topk_doc_ids;          // [max_stage2_candidates]
        int*    d_topk_indices;          // [max_stage2_candidates]

        // Pinned host memory for fast H2D/D2H
        float*  h_pinned_queries;
        float*  h_pinned_cb1_sumq;
        size_t* h_pinned_emb_ids;
        int*    h_pinned_pair_offsets;
        float*  h_pinned_dists;
        faiss::idx_t* h_pinned_cagra_labels; // [max_q_doclen * max_nprobe]
        float*  h_pinned_batch_scores;       // [max_stage2_candidates] batch D2H

        size_t max_q_doclen;
        size_t max_nprobe;
        size_t max_stage1_pairs;
        size_t max_stage2_candidates;
        size_t max_stage2_tokens;
        size_t max_stage2_k;
        size_t max_stage2_k_tokens;
        size_t estimated_num_docs;

        // === NEW: CUDA streams for async pipeline ===
        cudaStream_t stream_compute;
        cudaStream_t stream_h2d;
        cudaStream_t stream_d2h;

        // === NEW: Profiling events ===
#ifdef GPU_MVR_PROFILE
        cudaEvent_t event_start, event_end;
        cudaEvent_t event_stage1_start, event_stage1_end;
        cudaEvent_t event_stage2_start, event_stage2_end;
        cudaEvent_t event_cagra_start, event_cagra_end;
        cudaEvent_t event_stage1_rest_start, event_stage1_rest_end;
#endif
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
        ws_.estimated_num_docs = (size_t)num_docs;

        // Query workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_queries, max_q_doclen * padded_dim_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_cb1_sumq, max_q_doclen * sizeof(float)));

        // Stage 1 workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_emb_ids, ws_.max_stage1_pairs * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pair_offsets, (max_q_doclen + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_emb_dists, ws_.max_stage1_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pair_doc_ids, ws_.max_stage1_pairs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_pair_query_indices, ws_.max_stage1_pairs * sizeof(int)));

        // === NEW: Stage 1 GPU aggregation workspace ===
        CUDA_CHECK(cudaMalloc(&ws_.d_sorted_doc_ids, ws_.max_stage1_pairs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_sorted_query_indices, ws_.max_stage1_pairs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_sorted_dists, ws_.max_stage1_pairs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_unique_doc_ids, ws_.estimated_num_docs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_doc_offsets, (ws_.estimated_num_docs + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_stage1_doc_scores, ws_.estimated_num_docs * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_doc_query_max, ws_.estimated_num_docs * max_q_doclen * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_num_unique_docs, sizeof(int)));
        
        // Query CUB temporary storage size (dry run)
        ws_.d_cub_temp_storage = nullptr;
        ws_.cub_temp_storage_bytes = 0;
        // Estimate for sort pairs and segmented reduce operations
        size_t temp_bytes_sort = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, temp_bytes_sort,
            ws_.d_sorted_doc_ids, ws_.d_sorted_doc_ids,
            ws_.d_sorted_dists, ws_.d_sorted_dists,
            ws_.max_stage1_pairs
        );
        ws_.cub_temp_storage_bytes = std::max(ws_.cub_temp_storage_bytes, temp_bytes_sort);
        CUDA_CHECK(cudaMalloc(&ws_.d_cub_temp_storage, ws_.cub_temp_storage_bytes));

        // === NEW: GPU top-k workspace (must fit num_docs for Stage 1 sort) ===
        size_t topk_buf_size = std::max((size_t)num_docs, ws_.max_stage2_candidates);
        CUDA_CHECK(cudaMalloc(&ws_.d_topk_scores, topk_buf_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_topk_doc_ids, topk_buf_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&ws_.d_topk_indices, topk_buf_size * sizeof(int)));

        // Stage 2 workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_token_ids, ws_.max_stage2_tokens * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_candidate_offsets, (ws_.max_stage2_candidates + 1) * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&ws_.d_token_dists, ws_.max_stage2_tokens * max_q_doclen * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&ws_.d_doc_scores, ws_.max_stage2_candidates * sizeof(float)));

        // Extract workspace
        CUDA_CHECK(cudaMalloc(&ws_.d_selected_indices, ws_.max_stage2_k * sizeof(int)));
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
        CUDA_CHECK(cudaMallocHost(&ws_.h_pinned_batch_scores, ws_.max_stage2_candidates * sizeof(float)));

        // === NEW: Create CUDA streams ===
        CUDA_CHECK(cudaStreamCreate(&ws_.stream_compute));
        CUDA_CHECK(cudaStreamCreate(&ws_.stream_h2d));
        CUDA_CHECK(cudaStreamCreate(&ws_.stream_d2h));

        // === NEW: Create profiling events ===
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventCreate(&ws_.event_start));
        CUDA_CHECK(cudaEventCreate(&ws_.event_end));
        CUDA_CHECK(cudaEventCreate(&ws_.event_stage1_start));
        CUDA_CHECK(cudaEventCreate(&ws_.event_stage1_end));
        CUDA_CHECK(cudaEventCreate(&ws_.event_stage2_start));
        CUDA_CHECK(cudaEventCreate(&ws_.event_stage2_end));
        CUDA_CHECK(cudaEventCreate(&ws_.event_cagra_start));
        CUDA_CHECK(cudaEventCreate(&ws_.event_cagra_end));
        CUDA_CHECK(cudaEventCreate(&ws_.event_stage1_rest_start));
        CUDA_CHECK(cudaEventCreate(&ws_.event_stage1_rest_end));
#endif
    }

    size_t doc_len(size_t doc_id) const {
        return doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id];
    }

    // ======================== SEARCH PIPELINE ========================

    std::vector<size_t> search(const float* queries, size_t q_doclen, size_t k, size_t nprobe) {
        int k_rank_cluster = 20000;
        int k_rank_all_tokens = 1000;

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_start, ws_.stream_compute));
#endif

        // Step 1: Rotate queries on CPU into pinned memory (async-ready)
        for (size_t i = 0; i < q_doclen; ++i) {
            rotator_->rotate(&queries[i * d], &ws_.h_pinned_queries[i * padded_dim_]);
        }

        // Build query objects on CPU
        std::vector<query_object> query_objs(q_doclen);
        for (size_t i = 0; i < q_doclen; ++i) {
            query_objs[i] = query_object(&ws_.h_pinned_queries[i * padded_dim_], padded_dim_, ex_bits);
            ws_.h_pinned_cb1_sumq[i] = query_objs[i].cb1_sumq;
        }

        // Upload from pinned memory using async stream
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_queries, ws_.h_pinned_queries,
                                   q_doclen * padded_dim_ * sizeof(float),
                                   cudaMemcpyHostToDevice, ws_.stream_h2d));
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_cb1_sumq, ws_.h_pinned_cb1_sumq,
                                   q_doclen * sizeof(float),
                                   cudaMemcpyHostToDevice, ws_.stream_h2d));

        // Wait for H2D transfer before compute
        cudaEvent_t h2d_done;
        CUDA_CHECK(cudaEventCreate(&h2d_done));
        CUDA_CHECK(cudaEventRecord(h2d_done, ws_.stream_h2d));
        CUDA_CHECK(cudaStreamWaitEvent(ws_.stream_compute, h2d_done));

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_start, ws_.stream_compute));
#endif

        // Stage 1: GPU-only aggregation and top-k
        int actual_k_stage1 = 0;
        rank_cluster_dists_gpu(query_objs.data(), q_doclen, nprobe, k_rank_cluster,
                              actual_k_stage1, ws_.stream_compute);

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventRecord(ws_.event_stage2_start, ws_.stream_compute));
#endif

        // === Pipeline: Stage 2 (GPU producer) + Stage 3 (CPU consumer) overlap ===
        PipelineQueue pipeline_queue;
        std::vector<size_t> result;

        // Launch consumer thread for Stage 3 (ex-bits refinement on CPU)
        std::thread consumer_thread(&gpu_mvr_index::stage3_consumer, this,
                                   std::ref(pipeline_queue), query_objs.data(),
                                   q_doclen, k, std::ref(result));

        // Run producer (Stage 2: batched 1-bit scoring) on main thread
        rank_all_tokens_1bit_batched_gpu(q_doclen, actual_k_stage1, k_rank_all_tokens,
                                         pipeline_queue, ws_.stream_compute);

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_stage2_end, ws_.stream_compute));
#endif

        // Wait for consumer to finish
        consumer_thread.join();

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_end, ws_.stream_compute));
        CUDA_CHECK(cudaEventSynchronize(ws_.event_end));

        float total_time, stage1_time, stage2_time;
        float cagra_time, stage1_rest_time;
        CUDA_CHECK(cudaEventElapsedTime(&total_time, ws_.event_start, ws_.event_end));
        CUDA_CHECK(cudaEventElapsedTime(&stage1_time, ws_.event_stage1_start, ws_.event_stage1_end));
        CUDA_CHECK(cudaEventElapsedTime(&stage2_time, ws_.event_stage2_start, ws_.event_stage2_end));
        CUDA_CHECK(cudaEventElapsedTime(&cagra_time, ws_.event_cagra_start, ws_.event_cagra_end));
        CUDA_CHECK(cudaEventElapsedTime(&stage1_rest_time, ws_.event_stage1_rest_start, ws_.event_stage1_rest_end));

        std::cout << "[PROFILE] Total GPU time: " << total_time << " ms\n";
        std::cout << "[PROFILE] Stage 1 time: " << stage1_time << " ms\n";
        std::cout << "[PROFILE]   - CAGRA search: " << cagra_time << " ms\n";
        std::cout << "[PROFILE]   - Stage 1 rest: " << stage1_rest_time << " ms\n";
        std::cout << "[PROFILE] Stage 2 time: " << stage2_time << " ms\n";
#endif

        CUDA_CHECK(cudaEventDestroy(h2d_done));

        return result;
    }

    // ======================== STAGE 1: GPU ========================
    // GPU-only approach: eliminates CPU aggregation round-trip

    void rank_cluster_dists_gpu(
        query_object* h_query_objs,
        size_t q_doclen, size_t nprobe, size_t k,
        int& actual_k_out,
        cudaStream_t stream = 0
    ) {
#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_cagra_start, stream));
#endif

        // 1. Batched CAGRA search: one call for all q_doclen queries on GPU
        //    ws_.d_queries is already on GPU from the caller.
        ivf->search_batch_gpu(ws_.d_queries, q_doclen, nprobe,
                              ws_.d_cagra_dists, ws_.d_cagra_labels);

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_cagra_end, stream));
#endif

        // Copy cluster labels back to CPU (one transfer instead of q_doclen)
        CUDA_CHECK(cudaMemcpy(ws_.h_pinned_cagra_labels, ws_.d_cagra_labels,
                              q_doclen * nprobe * sizeof(faiss::idx_t), cudaMemcpyDeviceToHost));

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_rest_start, stream));
#endif

        // Expand cluster IDs to embedding IDs via inv_list (CPU - needed for sparse access)
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

        if (total_pairs == 0) {
            actual_k_out = 0;
            return;
        }

        // Flatten into pinned memory contiguously by query
        for (size_t j = 0; j < q_doclen; ++j) {
            memcpy(&ws_.h_pinned_emb_ids[ws_.h_pinned_pair_offsets[j]],
                   per_query_ids[j].data(),
                   per_query_ids[j].size() * sizeof(size_t));
        }

        // Upload emb_ids and pair_offsets
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_emb_ids, ws_.h_pinned_emb_ids,
                                   total_pairs * sizeof(size_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_pair_offsets, ws_.h_pinned_pair_offsets,
                                   (q_doclen + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));

        // 2. Compute 1-bit distances using shared-memory kernel
        size_t max_embs_per_query = 0;
        for (size_t j = 0; j < q_doclen; ++j)
            max_embs_per_query = std::max(max_embs_per_query, per_query_ids[j].size());

        int threads_per_block = 256;
        int warps_per_block = threads_per_block / 32;
        int blocks_x = (max_embs_per_query + warps_per_block - 1) / warps_per_block;
        size_t smem_size = padded_dim_ * sizeof(float);

        dim3 grid(blocks_x, q_doclen);
        stage1_binary_ip_kernel<<<grid, threads_per_block, smem_size, stream>>>(
            ws_.d_queries, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
            ws_.d_emb_ids, ws_.d_pair_offsets, ws_.d_emb_dists,
            padded_dim_, q_doclen
        );
        CUDA_CHECK(cudaGetLastError());

        // === NEW: GPU-only aggregation using atomic operations ===
        // 3. Initialize doc_query_max matrix to 0 (matches original CPU logic)
        size_t matrix_size = num_docs * q_doclen;
        CUDA_CHECK(cudaMemsetAsync(ws_.d_doc_query_max, 0, matrix_size * sizeof(float), stream));

        // 4. Atomic aggregation: max per (doc, query)
        int thread_count = 256;
        int block_count = (total_pairs + thread_count - 1) / thread_count;
        
        aggregate_stage1_atomic_kernel<<<block_count, thread_count, 0, stream>>>(
            ws_.d_emb_ids, ws_.d_emb_dists, ws_.d_pair_offsets, d_doc_ids_,
            ws_.d_doc_query_max,  // doc_query_max matrix
            q_doclen, num_docs, total_pairs
        );
        CUDA_CHECK(cudaGetLastError());

        // 5. Sum across queries for each doc
        int doc_blocks = (num_docs + thread_count - 1) / thread_count;
        sum_doc_scores_kernel<<<doc_blocks, thread_count, 0, stream>>>(
            ws_.d_doc_query_max,  // doc_query_max matrix
            ws_.d_stage1_doc_scores,
            num_docs, q_doclen
        );
        CUDA_CHECK(cudaGetLastError());

        // 6. Top-k selection: sort all docs by score descending
        // Docs with no hits will have -inf score and sort to the bottom
        actual_k_out = std::min((int)k, (int)num_docs);
        
        // Initialize indices 0..num_docs-1
        thrust::device_ptr<int> indices_ptr(ws_.d_topk_indices);
        thrust::sequence(thrust::cuda::par.on(stream),
                        indices_ptr, indices_ptr + num_docs);

        // Sort (score, index) pairs in descending order
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes,
            ws_.d_stage1_doc_scores, ws_.d_topk_scores,
            ws_.d_topk_indices, ws_.d_sorted_doc_ids,  // reuse as temp indices out
            num_docs, 0, 32, stream
        );
        temp_bytes = std::max(temp_bytes, ws_.cub_temp_storage_bytes);
        if (temp_bytes > ws_.cub_temp_storage_bytes) {
            CUDA_CHECK(cudaFree(ws_.d_cub_temp_storage));
            CUDA_CHECK(cudaMalloc(&ws_.d_cub_temp_storage, temp_bytes));
            ws_.cub_temp_storage_bytes = temp_bytes;
        }

        cub::DeviceRadixSort::SortPairsDescending(
            ws_.d_cub_temp_storage, temp_bytes,
            ws_.d_stage1_doc_scores, ws_.d_topk_scores,
            ws_.d_topk_indices, ws_.d_sorted_doc_ids,
            num_docs, 0, 32, stream
        );
        CUDA_CHECK(cudaGetLastError());

        // Top-k doc IDs are just the top-k indices (since doc_id == index in [0..num_docs-1])
        // Copy to d_topk_doc_ids
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_topk_doc_ids, ws_.d_sorted_doc_ids,
                                   actual_k_out * sizeof(int), cudaMemcpyDeviceToDevice, stream));

        // Top-k doc IDs now in ws_.d_topk_doc_ids[0..actual_k_out-1]
        // These remain on GPU for Stage 2 consumption

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventRecord(ws_.event_stage1_rest_end, stream));
#endif
    }

    // ======================== STAGE 2: GPU ========================

    void rank_all_tokens_1bit_gpu(
        size_t q_doclen,
        int num_candidates,  // from Stage 1
        size_t k,
        std::vector<size_t>& output_ids,
        std::vector<float>& one_bit_dists,
        cudaStream_t stream = 0
    ) {
        if (num_candidates == 0) return;

        // === NEW: Compute offsets on GPU ===
        // 1. Gather document lengths
        int threads = 256;
        int blocks = (num_candidates + threads - 1) / threads;
        
        gather_doc_lengths_kernel<<<blocks, threads, 0, stream>>>(
            ws_.d_topk_doc_ids, d_doc_ptrs_,
            ws_.d_pair_doc_ids,  // reuse as doc_lengths buffer
            num_candidates
        );
        CUDA_CHECK(cudaGetLastError());

        // 2. Compute prefix sum to get candidate_offsets (int lengths → size_t offsets)
        size_t zero_val = 0;
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_candidate_offsets, &zero_val, sizeof(size_t),
                                   cudaMemcpyHostToDevice, stream));
        thrust::device_ptr<int> len_ptr(ws_.d_pair_doc_ids);
        thrust::device_ptr<size_t> off_ptr(ws_.d_candidate_offsets + 1);
        thrust::inclusive_scan(thrust::cuda::par.on(stream),
                              len_ptr, len_ptr + num_candidates,
                              off_ptr);

        // Get total tokens
        size_t total_tokens;
        CUDA_CHECK(cudaMemcpyAsync(&total_tokens, ws_.d_candidate_offsets + num_candidates,
                                   sizeof(size_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 3. Build token IDs on GPU
        gather_token_ids_kernel<<<num_candidates, 256, 0, stream>>>(
            ws_.d_topk_doc_ids,
            d_doc_ptrs_,
            ws_.d_candidate_offsets,
            ws_.d_token_ids,
            num_candidates
        );
        CUDA_CHECK(cudaGetLastError());

        // 4. Compute all (query, token) 1-bit distances with shared-memory kernel
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / 32;
        int blocks_x = (total_tokens + warps_per_block - 1) / warps_per_block;
        size_t smem_size = padded_dim_ * sizeof(float);

        dim3 grid(blocks_x, q_doclen);
        stage2_binary_ip_kernel<<<grid, threads_per_block, smem_size, stream>>>(
            ws_.d_queries, d_one_bit_code_, d_one_bit_factor_, ws_.d_cb1_sumq,
            ws_.d_token_ids, ws_.d_token_dists,
            padded_dim_, q_doclen, total_tokens
        );
        CUDA_CHECK(cudaGetLastError());

        // 5. Compute doc scores with shared-memory reduction
        int score_threads = 128;
        while (score_threads < (int)q_doclen && score_threads < 256) score_threads *= 2;
        score_threads = std::min(score_threads, 256);
        constexpr int TILE_T = 8;
        size_t score_smem = (q_doclen * (TILE_T + 1) + score_threads) * sizeof(float);

        doc_score_kernel<<<num_candidates, score_threads, score_smem, stream>>>(
            ws_.d_token_dists, ws_.d_candidate_offsets, ws_.d_doc_scores,
            q_doclen, total_tokens, num_candidates
        );
        CUDA_CHECK(cudaGetLastError());

        // 6. Top-k via CUB sort (descending)
        thrust::device_ptr<int> indices_ptr(ws_.d_selected_indices);
        thrust::sequence(thrust::cuda::par.on(stream),
                        indices_ptr, indices_ptr + num_candidates);

        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes,
            ws_.d_doc_scores, ws_.d_topk_scores,  // reuse topk buffers
            ws_.d_selected_indices, ws_.d_topk_indices,
            num_candidates, 0, 32, stream
        );
        temp_bytes = std::max(temp_bytes, ws_.cub_temp_storage_bytes);
        if (temp_bytes > ws_.cub_temp_storage_bytes) {
            CUDA_CHECK(cudaFree(ws_.d_cub_temp_storage));
            CUDA_CHECK(cudaMalloc(&ws_.d_cub_temp_storage, temp_bytes));
            ws_.cub_temp_storage_bytes = temp_bytes;
        }

        cub::DeviceRadixSort::SortPairsDescending(
            ws_.d_cub_temp_storage, temp_bytes,
            ws_.d_doc_scores, ws_.d_topk_scores,
            ws_.d_selected_indices, ws_.d_topk_indices,
            num_candidates, 0, 32, stream
        );
        CUDA_CHECK(cudaGetLastError());

        size_t actual_k = std::min(k, (size_t)num_candidates);

        // 7. Build output offsets for selected docs (using CPU for small arrays)
        // Copy top-k sort permutation indices to CPU
        std::vector<int> h_top_k_indices(actual_k);
        CUDA_CHECK(cudaMemcpyAsync(h_top_k_indices.data(), ws_.d_topk_indices,
                                   actual_k * sizeof(int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Build output offsets on CPU (small array)
        std::vector<size_t> candidate_offsets_cpu(num_candidates + 1);
        CUDA_CHECK(cudaMemcpy(candidate_offsets_cpu.data(), ws_.d_candidate_offsets,
                             (num_candidates + 1) * sizeof(size_t), cudaMemcpyDeviceToHost));

        std::vector<size_t> out_offsets(actual_k + 1, 0);
        for (size_t i = 0; i < actual_k; ++i) {
            size_t cand_idx = h_top_k_indices[i];
            out_offsets[i + 1] = out_offsets[i] +
                (candidate_offsets_cpu[cand_idx + 1] - candidate_offsets_cpu[cand_idx]);
        }
        size_t total_selected_tokens = out_offsets[actual_k];

        // Upload selection metadata to GPU
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_selected_indices, h_top_k_indices.data(),
                                   actual_k * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(ws_.d_out_offsets, out_offsets.data(),
                                   (actual_k + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream));

        // 8. Extract on GPU
        extract_one_bit_dists_kernel<<<actual_k, 256, 0, stream>>>(
            ws_.d_token_dists, ws_.d_candidate_offsets,
            ws_.d_selected_indices, ws_.d_out_one_bit_dists,
            ws_.d_out_offsets, q_doclen, total_tokens, actual_k
        );
        CUDA_CHECK(cudaGetLastError());

        // 9. Copy only selected dists to pinned host memory
        size_t copy_size = total_selected_tokens * q_doclen;
        CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_dists, ws_.d_out_one_bit_dists,
                                   copy_size * sizeof(float), cudaMemcpyDeviceToHost, stream));

        // Also get the top-k doc IDs (need to gather from original candidates)
        std::vector<int> h_candidate_doc_ids(num_candidates);
        CUDA_CHECK(cudaMemcpy(h_candidate_doc_ids.data(), ws_.d_topk_doc_ids,
                             num_candidates * sizeof(int), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Build output
        output_ids.resize(actual_k);
        one_bit_dists.resize(copy_size);

        for (size_t i = 0; i < actual_k; ++i) {
            output_ids[i] = h_candidate_doc_ids[h_top_k_indices[i]];
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
#pragma omp parallel for
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
#pragma omp critical
            max_heap.emplace(doc_score, doc_id);
        }
        for (size_t i = 0; i < k && !max_heap.empty(); ++i) {
            output_ids.push_back(max_heap.top().second);
            max_heap.pop();
        }
    }

    // ======================== PIPELINE: Batched Stage 2 Producer ========================

    void rank_all_tokens_1bit_batched_gpu(
        size_t q_doclen, int num_candidates, size_t k,
        PipelineQueue& queue, cudaStream_t stream
    ) {
        if (num_candidates == 0) {
            ConsumerBatch done_batch;
            done_batch.done = true;
            queue.push(std::move(done_batch));
            return;
        }

        // Get all candidate doc IDs to CPU (single D2H transfer)
        std::vector<int> h_all_doc_ids(num_candidates);
        CUDA_CHECK(cudaMemcpy(h_all_doc_ids.data(), ws_.d_topk_doc_ids,
                              num_candidates * sizeof(int), cudaMemcpyDeviceToHost));

        // Running top-k min-heap: (score, global_candidate_index)
        using HeapEntry = std::pair<float, int>;
        std::priority_queue<HeapEntry, std::vector<HeapEntry>,
                           std::greater<HeapEntry>> min_heap;

        const int batch_size = 5000;

        for (int batch_start = 0; batch_start < num_candidates;
             batch_start += batch_size) {
            int batch_end = std::min(batch_start + batch_size, num_candidates);
            int batch_count = batch_end - batch_start;

            // A. Gather doc lengths
            int threads = 256;
            int blocks = (batch_count + threads - 1) / threads;
            gather_doc_lengths_kernel<<<blocks, threads, 0, stream>>>(
                ws_.d_topk_doc_ids + batch_start, d_doc_ptrs_,
                ws_.d_pair_doc_ids, batch_count
            );
            CUDA_CHECK(cudaGetLastError());

            // B. Prefix sum → candidate_offsets
            size_t zero_val = 0;
            CUDA_CHECK(cudaMemcpyAsync(ws_.d_candidate_offsets, &zero_val,
                        sizeof(size_t), cudaMemcpyHostToDevice, stream));
            thrust::device_ptr<int> len_ptr(ws_.d_pair_doc_ids);
            thrust::device_ptr<size_t> off_ptr(ws_.d_candidate_offsets + 1);
            thrust::inclusive_scan(thrust::cuda::par.on(stream),
                                  len_ptr, len_ptr + batch_count, off_ptr);

            size_t batch_total_tokens;
            CUDA_CHECK(cudaMemcpyAsync(&batch_total_tokens,
                        ws_.d_candidate_offsets + batch_count,
                        sizeof(size_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            if (batch_total_tokens == 0) continue;

            // C. Gather token IDs
            gather_token_ids_kernel<<<batch_count, 256, 0, stream>>>(
                ws_.d_topk_doc_ids + batch_start, d_doc_ptrs_,
                ws_.d_candidate_offsets, ws_.d_token_ids, batch_count
            );
            CUDA_CHECK(cudaGetLastError());

            // D. 1-bit binary IP
            int tpb = 256, wpb = tpb / 32;
            int bx = (batch_total_tokens + wpb - 1) / wpb;
            size_t smem = padded_dim_ * sizeof(float);
            dim3 grid(bx, q_doclen);
            stage2_binary_ip_kernel<<<grid, tpb, smem, stream>>>(
                ws_.d_queries, d_one_bit_code_, d_one_bit_factor_,
                ws_.d_cb1_sumq, ws_.d_token_ids, ws_.d_token_dists,
                padded_dim_, q_doclen, batch_total_tokens
            );
            CUDA_CHECK(cudaGetLastError());

            // E. Doc scoring
            int st = 128;
            while (st < (int)q_doclen && st < 256) st *= 2;
            st = std::min(st, 256);
            constexpr int TILE_T = 8;
            size_t ssm = (q_doclen * (TILE_T + 1) + st) * sizeof(float);
            doc_score_kernel<<<batch_count, st, ssm, stream>>>(
                ws_.d_token_dists, ws_.d_candidate_offsets, ws_.d_doc_scores,
                q_doclen, batch_total_tokens, batch_count
            );
            CUDA_CHECK(cudaGetLastError());

            // F. D2H batch scores
            CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_batch_scores,
                        ws_.d_doc_scores, batch_count * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // G. Update running top-k
            for (int i = 0; i < batch_count; ++i) {
                float score = ws_.h_pinned_batch_scores[i];
                int gidx = batch_start + i;
                if ((int)min_heap.size() < (int)k) {
                    min_heap.push({score, gidx});
                } else if (score > min_heap.top().first) {
                    min_heap.pop();
                    min_heap.push({score, gidx});
                }
            }

            // H. Find heap entries from this batch to extract
            std::vector<int> to_extract;
            {
                auto heap_copy = min_heap;
                while (!heap_copy.empty()) {
                    auto entry = heap_copy.top();
                    heap_copy.pop();
                    if (entry.second >= batch_start &&
                        entry.second < batch_end) {
                        to_extract.push_back(entry.second - batch_start);
                    }
                }
                std::sort(to_extract.begin(), to_extract.end());
            }
            if (to_extract.empty()) continue;

            // I. Read batch offsets, build extraction metadata
            std::vector<size_t> batch_offsets_cpu(batch_count + 1);
            CUDA_CHECK(cudaMemcpy(batch_offsets_cpu.data(),
                        ws_.d_candidate_offsets,
                        (batch_count + 1) * sizeof(size_t),
                        cudaMemcpyDeviceToHost));

            int num_extract = to_extract.size();
            std::vector<size_t> out_offsets(num_extract + 1, 0);
            for (int i = 0; i < num_extract; ++i) {
                int bi = to_extract[i];
                out_offsets[i + 1] = out_offsets[i] +
                    (batch_offsets_cpu[bi + 1] - batch_offsets_cpu[bi]);
            }
            size_t total_extract_tokens = out_offsets[num_extract];

            // J. Upload indices + offsets, extract, D2H
            CUDA_CHECK(cudaMemcpyAsync(ws_.d_selected_indices,
                        to_extract.data(), num_extract * sizeof(int),
                        cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(ws_.d_out_offsets, out_offsets.data(),
                        (num_extract + 1) * sizeof(size_t),
                        cudaMemcpyHostToDevice, stream));

            extract_one_bit_dists_kernel<<<num_extract, 256, 0, stream>>>(
                ws_.d_token_dists, ws_.d_candidate_offsets,
                ws_.d_selected_indices, ws_.d_out_one_bit_dists,
                ws_.d_out_offsets, q_doclen, batch_total_tokens, num_extract
            );
            CUDA_CHECK(cudaGetLastError());

            size_t copy_elems = total_extract_tokens * q_doclen;
            CUDA_CHECK(cudaMemcpyAsync(ws_.h_pinned_dists,
                        ws_.d_out_one_bit_dists,
                        copy_elems * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // K. Build consumer batch and push
            ConsumerBatch cbatch;
            cbatch.doc_ids.resize(num_extract);
            cbatch.doc_ptrs.resize(num_extract + 1, 0);
            cbatch.one_bit_dists.resize(copy_elems);
            for (int i = 0; i < num_extract; ++i) {
                int bi = to_extract[i];
                cbatch.doc_ids[i] = h_all_doc_ids[batch_start + bi];
                cbatch.doc_ptrs[i + 1] = cbatch.doc_ptrs[i] +
                    (batch_offsets_cpu[bi + 1] - batch_offsets_cpu[bi]);
            }
            memcpy(cbatch.one_bit_dists.data(), ws_.h_pinned_dists,
                   copy_elems * sizeof(float));
            queue.push(std::move(cbatch));
        }

        // Signal completion
        ConsumerBatch done_batch;
        done_batch.done = true;
        queue.push(std::move(done_batch));
    }

    // ======================== PIPELINE: Stage 3 Consumer ========================

    void stage3_consumer(
        PipelineQueue& queue, query_object* queries, size_t q_doclen,
        size_t k, std::vector<size_t>& output_ids
    ) {
        std::priority_queue<std::pair<float, size_t>> max_heap;

        while (true) {
            ConsumerBatch batch = queue.pop();
            if (batch.done) break;

            size_t nbatch = batch.doc_ids.size();
#pragma omp parallel for schedule(dynamic)
            for (size_t idx = 0; idx < nbatch; ++idx) {
                size_t doc_id = batch.doc_ids[idx];
                float doc_score = 0.0f;
                for (size_t j = 0; j < q_doclen; ++j) {
                    float max_ts = -std::numeric_limits<float>::infinity();
                    for (size_t i = 0; i < doc_len(doc_id); ++i) {
                        size_t tid = doc_ptrs_[doc_id] + i;
                        float dist = distance_ex_bits(
                            queries + j,
                            &ex_code_[tid * padded_dim_ * ex_bits / 8],
                            ex_bits, ip_func_,
                            batch.one_bit_dists[(batch.doc_ptrs[idx] + i) * q_doclen + j],
                            one_bit_factor_[tid],
                            ex_factor_[tid],
                            padded_dim_
                        );
                        max_ts = std::max(max_ts, dist);
                    }
                    doc_score += max_ts;
                }
#pragma omp critical
                max_heap.emplace(doc_score, doc_id);
            }
        }

        output_ids.reserve(k);
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

        // === NEW: Free GPU-only aggregation workspace ===
        CUDA_CHECK(cudaFree(ws_.d_sorted_doc_ids));
        CUDA_CHECK(cudaFree(ws_.d_sorted_query_indices));
        CUDA_CHECK(cudaFree(ws_.d_sorted_dists));
        CUDA_CHECK(cudaFree(ws_.d_unique_doc_ids));
        CUDA_CHECK(cudaFree(ws_.d_doc_offsets));
        CUDA_CHECK(cudaFree(ws_.d_stage1_doc_scores));
        CUDA_CHECK(cudaFree(ws_.d_doc_query_max));
        CUDA_CHECK(cudaFree(ws_.d_num_unique_docs));
        CUDA_CHECK(cudaFree(ws_.d_cub_temp_storage));
        CUDA_CHECK(cudaFree(ws_.d_topk_scores));
        CUDA_CHECK(cudaFree(ws_.d_topk_doc_ids));
        CUDA_CHECK(cudaFree(ws_.d_topk_indices));

        // Free pinned memory
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_queries));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_cb1_sumq));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_emb_ids));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_dists));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_cagra_labels));
        CUDA_CHECK(cudaFreeHost(ws_.h_pinned_batch_scores));

        // === NEW: Destroy streams and events ===
        CUDA_CHECK(cudaStreamDestroy(ws_.stream_compute));
        CUDA_CHECK(cudaStreamDestroy(ws_.stream_h2d));
        CUDA_CHECK(cudaStreamDestroy(ws_.stream_d2h));

#ifdef GPU_MVR_PROFILE
        CUDA_CHECK(cudaEventDestroy(ws_.event_start));
        CUDA_CHECK(cudaEventDestroy(ws_.event_end));
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage1_start));
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage1_end));
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage2_start));
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage2_end));
        CUDA_CHECK(cudaEventDestroy(ws_.event_cagra_start));
        CUDA_CHECK(cudaEventDestroy(ws_.event_cagra_end));
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage1_rest_start));
        CUDA_CHECK(cudaEventDestroy(ws_.event_stage1_rest_end));
#endif

        delete rotator_;
        delete ivf;
    }
};
