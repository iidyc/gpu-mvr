

class index {
public:
    size_t n;          // number of vectors
    size_t d;          // dimension
    size_t n_clusters; // number of clusters
    size_t n_bits;     // n_bits = 1 + n_exbits



    index(
        size_t n,
        size_t d,
        size_t n_clusters,
        size_t n_bits
    ): n(n), d(d), n_clusters(n_clusters), n_bits(n_bits) {

    }

    /***
     * @brief Quantize data points into RabitQ codes
     * @param data Data points to be quantized (n*d)
     */
    void quantize(const float* data) {

    }
};