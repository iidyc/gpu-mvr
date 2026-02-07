#include "gpu_index.cuh"
#include "io.hpp"
#include "utils.hpp"

int main() {
    int k = 100;
    int nprobe = 128;
    size_t num_q, d, q_doclen;
    std::vector<float> Q = load_query(q_doclen, num_q, d);
    std::vector<int> doclens = load_doclens();

    gpu_mvr_index index("2097152_4.index", doclens);

    Timer timer;
    timer.tick();
    int nq = 100;
    std::vector<std::vector<size_t>> results(nq);
    for (size_t i = 0; i < nq; ++i) {
        results[i] = index.search(&Q[i * q_doclen * d], q_doclen, k, nprobe);
    }
    timer.tuck("GPU search time for " + std::to_string(nq) + " queries.");

    auto ground_truth = read_gt_tsv(num_q, 1000);
    compute_recall(ground_truth, results, k);

    return 0;
}
