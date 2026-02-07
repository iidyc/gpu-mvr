#pragma once

#include <iostream>
#include <fstream>
#include <vector>

std::vector<float> load_data(size_t& num_embeddings, size_t& d) {
    int n_, d_;
    std::ifstream emb_file("embeddings.bin", std::ios::binary);
    emb_file.read(reinterpret_cast<char*>(&n_), sizeof(int));
    emb_file.read(reinterpret_cast<char*>(&d_), sizeof(int));
    num_embeddings = size_t(n_);
    d = size_t(d_);
    std::vector<float> embeddings(num_embeddings * d);
    emb_file.read(reinterpret_cast<char*>(embeddings.data()), embeddings.size() * sizeof(float));
    emb_file.close();
    std::cout << ">>> Loaded " << num_embeddings << " embeddings of dimension " << d << std::endl;
    return std::move(embeddings);
}

std::vector<float> load_query(size_t& q_doclen, size_t& num_q, size_t& d) {
    std::ifstream qemb_file("query_embeddings.bin", std::ios::binary);
    int num_q_, q_doclen_, d_;
    qemb_file.read(reinterpret_cast<char*>(&num_q_), sizeof(int));
    qemb_file.read(reinterpret_cast<char*>(&q_doclen_), sizeof(int));
    qemb_file.read(reinterpret_cast<char*>(&d_), sizeof(int));
    num_q = size_t(num_q_);
    q_doclen = size_t(q_doclen_);
    d = size_t(d_);
    std::vector<float> Q(num_q * q_doclen * d);
    qemb_file.read(reinterpret_cast<char*>(Q.data()), Q.size() * sizeof(float));
    qemb_file.close();
    std::cout << ">>> Loaded " << num_q << " queries, each with " << q_doclen << " embeddings of dimension " << d << std::endl;
    return std::move(Q);
}

std::vector<int> load_doclens() {
    std::ifstream doc_lens_file("doclens.bin", std::ios::binary);
    int doclens_size;
    doc_lens_file.read(reinterpret_cast<char*>(&doclens_size), sizeof(int));
    std::vector<int> doclens(doclens_size);
    doc_lens_file.read(reinterpret_cast<char*>(doclens.data()), doclens.size() * sizeof(int));
    doc_lens_file.close();
    std::cout << ">>> Loaded " << doclens_size << " document lengths" << std::endl;
    return std::move(doclens);
}