#include "rabitqlib/third/hnswlib/hnswlib.h"

// abstract class for PG index
struct PG {
    size_t n;
    size_t d;

    PG(size_t n, size_t d) : n(n), d(d) {}

    virtual void build_index(const float* data) const = 0;
    virtual void search(const float* query, size_t k, std::vector<size_t>& results) const = 0;
    virtual void save(const std::string& filename) const = 0;
    virtual void load(const std::string& filename) = 0;
    virtual ~PG() = default;
};

struct PG_HNSW: PG {
    size_t M = 16;
    size_t ef_construction = 500;
    hnswlib::L2Space space_;
    hnswlib::HierarchicalNSW<float>* hnsw_index;

    PG_HNSW(size_t n, size_t d): PG(n, d), space_(d) {
        hnsw_index = new hnswlib::HierarchicalNSW<float>(&space_, n, M, ef_construction);
    }

    void build_index(const float* data) const override {
#pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            hnsw_index->addPoint(data + i * d, i);
        }
    }

    void search(const float* query, size_t k, std::vector<size_t>& results) const override {
        hnsw_index->setEf(std::max(768UL, 2 * k));
        auto result = hnsw_index->searchKnn(query, k);
        while (!result.empty()) {
            results.push_back(result.top().second);
            result.pop();
        }
    }

    void save(const std::string& filename) const override {
        hnsw_index->saveIndex(filename + ".hnsw");
    }

    void load(const std::string& filename) override {
        hnsw_index->loadIndex(filename + ".hnsw", &space_, n);
    }

    ~PG_HNSW() {
        delete hnsw_index;
    }
};

struct IVF_PG {
    size_t n_clusters;
    size_t d;
    PG* pg_index;

    std::vector<int> inv_list;
    std::vector<size_t> cluster_pos;

    IVF_PG(size_t n_clusters, size_t d): n_clusters(n_clusters), d(d) {
        pg_index = new PG_HNSW(n_clusters, d);
    }

    void search(const float* query, size_t n_probe, std::vector<size_t>& results) const {
        std::vector<size_t> cluster_ids;
        pg_index->search(query, n_probe, cluster_ids);
        for (size_t id : cluster_ids) {
            size_t start = cluster_pos[id];
            size_t end = cluster_pos[id + 1];
            for (size_t i = start; i < end; ++i) {
                results.push_back(inv_list[i]);
            }
        }
    }

    void build_index(const float* data) {

    }

    void build_from_existing() {
        std::ifstream list_no_in("list_nos.bin", std::ios::binary);
        int n;
        list_no_in.read((char*)&n, sizeof(int));
        std::vector<int> list_nos(n);
        list_no_in.read((char*)list_nos.data(), n * sizeof(int));

        std::vector<std::vector<int>> clusters(n_clusters);
        for (int i = 0; i < n; ++i) {
            clusters[list_nos[i]].push_back(i);
        }
        size_t cumu_size = 0;
        for (size_t i = 0; i < n_clusters; ++i) {
            cluster_pos.push_back(cumu_size);
            cumu_size += clusters[i].size();
            inv_list.insert(inv_list.end(), clusters[i].begin(), clusters[i].end());
        }
        cluster_pos.push_back(cumu_size);

        pg_index->load("index");
    }

    void save(const std::string& filename) const {
        pg_index->save(filename + ".ivf");
        std::ofstream of(filename + ".ivf", std::ios::binary);
        size_t n = inv_list.size();
        of.write((char*)&n, sizeof(size_t));
        of.write((char*)inv_list.data(), n * sizeof(int));
        of.write((char*)cluster_pos.data(), cluster_pos.size() * sizeof(size_t));
        of.close();
    }

    void load(const std::string& filename) {
        std::ifstream inf(filename + ".ivf", std::ios::binary);
        size_t n;
        inf.read((char*)&n, sizeof(size_t));
        inv_list.resize(n);
        inf.read((char*)inv_list.data(), n * sizeof(int));
        cluster_pos.resize(n_clusters + 1);
        inf.read((char*)cluster_pos.data(), (n_clusters + 1) * sizeof(size_t));
        inf.close();

        pg_index->load(filename + ".ivf");
    }
    
    ~IVF_PG() {
        delete pg_index;
    }
};
