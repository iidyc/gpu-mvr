#include "ivf_pg.hpp"

int main() {
    int n, d;
    size_t num_embeddings;
    std::ifstream emb_file("centroids.bin", std::ios::binary);
    emb_file.read(reinterpret_cast<char*>(&n), sizeof(int));
    emb_file.read(reinterpret_cast<char*>(&d), sizeof(int));
    num_embeddings = size_t(n);
    std::vector<float> embeddings(num_embeddings * d);
    emb_file.read(reinterpret_cast<char*>(embeddings.data()), embeddings.size() * sizeof(float));
    emb_file.close();
    std::cout << ">>> Loaded " << num_embeddings << " embeddings of dimension " << d << std::endl;

    PG_CAGRA pg_cagra(n, d);
    pg_cagra.build_index(embeddings.data());
    std::vector<size_t> results;
    pg_cagra.search(embeddings.data(), 10, results);
    // pg_cagra.save("index.cagra");
}