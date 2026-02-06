#include "index.hpp"
#include "io.hpp"

#include "utils.hpp"

int main() {
    int k = 100;
    int nprobe = 128;
    size_t num_d, num_q, d, q_doclen, num_docs;
    std::vector<float> Q = load_query(q_doclen, num_q, d);
    std::vector<int> doclens = load_doclens();
    gpu_mvr_index index("2097152_4.index");
    index.set_doc_mapping(doclens);

    // std::vector<float> rotated_q(d);
    // index.rotator_->rotate(&Q[0], rotated_q.data());
    // query_object query_obj(rotated_q.data(), index.padded_dim_, index.ex_bits);
    // float one_bit_dist = distance_one_bit(&query_obj, index.one_bit_code_.data(), index.one_bit_factor_[0], index.padded_dim_);
    // float ex_dist = distance_ex_bits(&query_obj, index.ex_code_.data(), index.ex_bits, index.ip_func_, one_bit_dist, index.one_bit_factor_[0], index.ex_factor_[0], index.padded_dim_);
    // std::cout << "Distance check: " << ex_dist << "\n" << std::flush;

    // return 0;

    Timer timer;
    timer.tick();
    int nq = 100;
    std::vector<std::vector<size_t>> results(nq);
#pragma omp parallel for
    for (size_t i = 0; i < nq; ++i) {
        results[i] = index.search(&Q[i * q_doclen * d], q_doclen, k, nprobe);
    }
    timer.tuck("Search time for " + std::to_string(nq) + " queries.");
    auto ground_truth = read_gt_tsv(num_q, 1000);
    compute_recall(ground_truth, results, k);

    return 0;
}