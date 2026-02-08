#pragma once

#include <vector>
#include <queue>
#include <unordered_map>
#include <omp.h>

#include "rabitqlib/quantization/rabitq_impl.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "quantization.hpp"
#include "estimator.hpp"
#include "query.hpp"
#include "ivf_pg.hpp"

using namespace rabitqlib;

struct cpu_mvr_index {
    size_t n;           // number of vectors
    size_t d;           // dimension
    size_t n_clusters;  // number of clusters
    size_t ex_bits;     // n_bits = 1 + ex_bits
    size_t padded_dim_; // multiple of 64 dimension after padding
    Rotator<float>* rotator_;

    IVF_PG* ivf;

    std::vector<char> one_bit_code_;
    std::vector<char> ex_code_;
    std::vector<float> one_bit_factor_;
    std::vector<float> ex_factor_;

    size_t num_docs;
    std::vector<int> doc_ids_;  // document ids for each vector, size n
    std::vector<int> doc_ptrs_; // pointers to the start of each document, size num_docs + 1

    float (*ip_func_)(const float*, const uint8_t*, size_t);

    cpu_mvr_index(const std::string& filename) {
        load(filename);
        ip_func_ = select_excode_ipfunc(ex_bits);
    }

    cpu_mvr_index(
        size_t n,
        size_t d,
        size_t n_clusters,
        size_t ex_bits
    ): n(n), d(d), n_clusters(n_clusters), ex_bits(ex_bits) {
        rotator_ = choose_rotator<float>(d, RotatorType::FhtKacRotator, round_up_to_multiple(d, 64));
        std::ifstream rot_in("rotator.bin", std::ios::binary);
        rotator_->load(rot_in);
        rot_in.close();
        padded_dim_ = rotator_->size();

        one_bit_code_.resize(n * padded_dim_ / 8);
        ex_code_.resize(n * padded_dim_ * ex_bits / 8);
        one_bit_factor_.resize(n);
        ex_factor_.resize(n);

        ip_func_ = select_excode_ipfunc(ex_bits);

        ivf = new IVF_PG(n_clusters, d);
    }

    void build_index(const float* data) {
        quantize(data);
        // ivf->build_from_existing();
    }

    /***
     * @brief Quantize data points into RabitQ codes
     * @param data Data points to be quantized (n*d)
     */
    void quantize(const float* data) {
        size_t batch_size = 10240;
        for (size_t start = 0; start < n; start += batch_size) {
            // std::cout << "Quantizing data points " << start << " to "
            //           << std::min(start + batch_size, n) - 1 << " / " << n << "\n"
            //           << std::flush;
            size_t end = std::min(start + batch_size, n);
            size_t current_batch_size = end - start;

            std::vector<float> rotated_data(current_batch_size * padded_dim_);
#pragma omp parallel for
            for (size_t i = 0; i < current_batch_size; ++i) {
                // Rotate data
                rotator_->rotate(&data[(start + i) * d], &rotated_data[i * padded_dim_]);

                // Encode one-bit quantization and compute factor
                encode_one_bit(
                    &rotated_data[i * padded_dim_],
                    padded_dim_,
                    reinterpret_cast<uint64_t*>(&one_bit_code_[(start + i) * padded_dim_ / 8]),
                    &one_bit_factor_[start + i]
                );

                encode_ex_bits(
                    &rotated_data[i * padded_dim_],
                    padded_dim_,
                    ex_bits,
                    reinterpret_cast<uint8_t*>(&ex_code_[(start + i) * padded_dim_ * ex_bits / 8]),
                    &ex_factor_[start + i]
                );
            }
        }
    }

    void set_doc_mapping(const std::vector<int>& doc_lens) {
        num_docs = doc_lens.size();
        doc_ptrs_.resize(num_docs + 1);
        for (size_t i = 0; i < num_docs; ++i) {
            doc_ptrs_[i + 1] = doc_ptrs_[i] + doc_lens[i];
        }
        doc_ids_.resize(n);
        for (size_t i = 0; i < num_docs; ++i) {
            for (size_t j = 0; j < doc_lens[i]; ++j) {
                doc_ids_[doc_ptrs_[i] + j] = i;
            }
        }
        if (doc_ptrs_[num_docs] != n) {
            throw std::runtime_error("Error in set_doc_mapping: total number of vectors does not match!");
        }
    }

    size_t doc_len(size_t doc_id) const {
        return doc_ptrs_[doc_id + 1] - doc_ptrs_[doc_id];
    }

    std::vector<size_t> search(const float* queries, size_t q_doclen, size_t k, size_t nprobe) {
        int k_rank_cluster = 20000;
        int k_rank_all_tokens = 1000;

        std::vector<float> rotated_queries(q_doclen * padded_dim_);
        for (size_t i = 0; i < q_doclen; ++i) {
            rotator_->rotate(&queries[i * d], &rotated_queries[i * padded_dim_]);
        }
        std::vector<query_object> query_objs(q_doclen);
        for (size_t i = 0; i < q_doclen; ++i) {
            query_objs[i] = query_object(&rotated_queries[i * padded_dim_], padded_dim_, ex_bits);
        }
        std::vector<size_t> rank_cluster_doc_ids;
        rank_cluster_dists(query_objs.data(), q_doclen, nprobe, k_rank_cluster, rank_cluster_doc_ids);
        std::vector<size_t> rank_all_tokens_ids;
        std::vector<float> one_bit_dists;
        rank_all_tokens_1bit(query_objs.data(), q_doclen, rank_cluster_doc_ids, k_rank_all_tokens, rank_all_tokens_ids, one_bit_dists);
        std::vector<size_t> result;
        rank_all_tokens_exbits(
            query_objs.data(),
            q_doclen,
            rank_all_tokens_ids,
            one_bit_dists,
            k,
            result
        );
        return result;
    }

    void rank_cluster_dists(query_object* queries, size_t q_doclen, size_t nprobe, size_t k, std::vector<size_t>& output_ids) {
        std::vector<bool> doc_found(num_docs, false);
        std::vector<float> doc_dists(q_doclen * num_docs);
        double gather_matrix_time = 0.0;
        for (int j = 0; j < q_doclen; ++j) {
            std::vector<size_t> ids;
            ivf->search(queries[j].rotated_query, nprobe, ids);
            for (size_t idx = 0; idx < ids.size(); ++idx) {
                size_t emb_id = ids[idx];
                float dist = distance_one_bit(queries + j, &one_bit_code_[emb_id * padded_dim_ / 8], one_bit_factor_[emb_id], padded_dim_);
                int doc_id = doc_ids_[emb_id];
                doc_found[doc_id] = true;
                doc_dists[j * num_docs + doc_id] = std::max(doc_dists[j * num_docs + doc_id], dist);
            }
        }
        std::priority_queue<std::pair<float, int>> max_heap;
        for (int doc_id = 0; doc_id < num_docs; ++doc_id) {
            if (doc_found[doc_id]) {
                float score = 0.0F;
                for (int j = 0; j < q_doclen; ++j) {
                    score += doc_dists[j * num_docs + doc_id];
                }
                max_heap.emplace(score, doc_id);
            }
        }
        for (int i = 0; i < k && !max_heap.empty(); ++i) {
            output_ids.push_back(max_heap.top().second);
            max_heap.pop();
        }
    }

    void rank_all_tokens_1bit(
        query_object* queries, 
        size_t q_doclen, 
        std::vector<size_t>& input_ids, 
        size_t k, 
        std::vector<size_t>& output_ids,
        std::vector<float>& one_bit_dists
    ) {
        std::unordered_map<size_t, size_t> doc_id_to_index;
        std::priority_queue<std::pair<float, size_t>> max_heap;
        size_t total_tokens = 0;
        std::vector<size_t> candidate_doc_ptrs(input_ids.size() + 1);
        for (size_t i = 0; i < input_ids.size(); ++i) {
            doc_id_to_index[input_ids[i]] = i;
            total_tokens += doc_len(input_ids[i]);
            candidate_doc_ptrs[i + 1] = total_tokens;
        }
        std::vector<float> token_dists(total_tokens * q_doclen);
        for (size_t idx = 0; idx < input_ids.size(); ++idx) {
            size_t doc_id = input_ids[idx];
            float doc_score = 0.0F;
            for (int j = 0; j < q_doclen; ++j) {
                float max_token_score = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < doc_len(doc_id); ++i) {
                    size_t tid = doc_ptrs_[doc_id] + i;
                    float dist = distance_one_bit(queries + j, &one_bit_code_[tid * padded_dim_ / 8], one_bit_factor_[tid], padded_dim_);
                    token_dists[(candidate_doc_ptrs[idx] + i) * q_doclen + j] = dist;
                    max_token_score = std::max(max_token_score, dist);
                }
                doc_score += max_token_score;
            }
            max_heap.emplace(doc_score, doc_id);
        }
        for (int i = 0; i < k && !max_heap.empty(); ++i) {
            size_t doc_id = max_heap.top().second;
            output_ids.push_back(doc_id);
            for (size_t t = 0; t < doc_len(doc_id); ++t) {
                size_t doc_idx = doc_id_to_index[doc_id];
                for (int j = 0; j < q_doclen; ++j) {
                    one_bit_dists.push_back(token_dists[(candidate_doc_ptrs[doc_idx] + t) * q_doclen + j]);
                }
            }
            max_heap.pop();
        }
    }

    void rank_all_tokens_exbits(
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
            for (int j = 0; j < q_doclen; ++j) {
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
        for (int i = 0; i < k && !max_heap.empty(); ++i) {
            output_ids.push_back(max_heap.top().second);
            max_heap.pop();
        }
    }

    void save(const std::string& filename) const {
        std::ofstream of(filename, std::ios::binary);
        of.write((char*)&n, sizeof(size_t));
        of.write((char*)&d, sizeof(size_t));
        of.write((char*)&n_clusters, sizeof(size_t));
        of.write((char*)&ex_bits, sizeof(size_t));
        of.write((char*)&padded_dim_, sizeof(size_t));
        of.write(one_bit_code_.data(), one_bit_code_.size());
        of.write(ex_code_.data(), ex_code_.size());
        of.write((char*)one_bit_factor_.data(), one_bit_factor_.size() * sizeof(float));
        of.write((char*)ex_factor_.data(), ex_factor_.size() * sizeof(float));
        of.close();
        // ivf->save(filename);
    }

    void load(const std::string& filename) {
        std::ifstream inf(filename, std::ios::binary);
        inf.read((char*)&n, sizeof(size_t));
        inf.read((char*)&d, sizeof(size_t));
        inf.read((char*)&n_clusters, sizeof(size_t));
        inf.read((char*)&ex_bits, sizeof(size_t));
        inf.read((char*)&padded_dim_, sizeof(size_t));
        one_bit_code_.resize(n * padded_dim_ / 8);
        ex_code_.resize(n * padded_dim_ * ex_bits / 8);
        one_bit_factor_.resize(n);
        ex_factor_.resize(n);
        inf.read(one_bit_code_.data(), one_bit_code_.size());
        inf.read(ex_code_.data(), ex_code_.size());
        inf.read((char*)one_bit_factor_.data(), one_bit_factor_.size() * sizeof(float));
        inf.read((char*)ex_factor_.data(), ex_factor_.size() * sizeof(float));
        rotator_ = choose_rotator<float>(d, RotatorType::FhtKacRotator, padded_dim_);
        std::ifstream rot_in("rotator.bin", std::ios::binary);
        rotator_->load(rot_in);
        rot_in.close();
        inf.close();
        ivf = new IVF_PG(n_clusters, d);
        ivf->load(filename);
    }

    ~cpu_mvr_index() {
        delete rotator_;
        delete ivf;
    }
};