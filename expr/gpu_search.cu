#include "gpu_index.cuh"
#include "io.hpp"
#include "utils.hpp"

int main() {
    int k = 100;
    size_t num_q, d, q_doclen_file;
    std::vector<float> Q = load_query(q_doclen_file, num_q, d);
    std::vector<int> doclens = load_doclens();
    auto ground_truth = read_gt_tsv(num_q, 1000);

    // Validate that query file matches compiled Q_DOCLEN
    if (q_doclen_file != Q_DOCLEN) {
        std::cerr << "ERROR: Query file q_doclen=" << q_doclen_file
                  << " does not match compiled Q_DOCLEN=" << Q_DOCLEN << std::endl;
        std::cerr << "Please recompile with matching Q_DOCLEN in gpu_config.cuh" << std::endl;
        return 1;
    }

    gpu_mvr_index index("2097152_4_new.index", doclens);

    Timer timer;
    timer.tick();
    int nq = 5;
    std::vector<std::vector<size_t>> results(nq);
    for (size_t i = 0; i < nq; ++i) {
        results[i] = index.search(&Q[i * Q_DOCLEN * d], k);
    }
    timer.tuck("GPU search time for " + std::to_string(nq) + " queries.");

    compute_recall(ground_truth, results, k);

    return 0;
}
