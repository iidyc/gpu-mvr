#include "index.hpp"
#include "io.hpp"

int main() {
    size_t n, d;
    std::vector<float> data = load_data(n, d);
    cpu_mvr_index idx(n, d, 2097152, 4);
    idx.build_index(data.data());
    idx.save("2097152_4_new.index");
    return 0;
}