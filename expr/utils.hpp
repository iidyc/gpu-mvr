#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_set>

struct Timer {
    std::chrono::_V2::system_clock::time_point s;
    std::chrono::_V2::system_clock::time_point e;
    std::chrono::duration<double> diff;

    void tick() {
        s = std::chrono::high_resolution_clock::now();
    }

    void tuck(std::string message, bool print = true) {
        e = std::chrono::high_resolution_clock::now();
        diff = e - s;
        if (print) {
            std::cout << "[" << diff.count() << " s] " << message << std::endl;
        }
    }
};

std::vector<std::vector<size_t>> read_gt_tsv(int num_queries, int top_k) {
    const std::string baseline_tsv_name = "lotte-groundtruth-top1000--.tsv";
    std::vector<std::vector<size_t>> ground_truth(num_queries, std::vector<size_t>(top_k, -1));
    std::ifstream file(baseline_tsv_name);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << baseline_tsv_name << std::endl;
        return ground_truth; 
    }
    std::string line;
    // We don't need 'count' for the map approach, but we read the file line by line
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string segment;
        std::vector<std::string> arr;
        while (std::getline(ss, segment, '\t')) {
            arr.push_back(segment);
        }
        if (arr.size() < 3) {
            std::cerr << "Warning: Skipping malformed line." << std::endl;
            continue;
        }
        int qID = std::stoi(arr[0]);
        size_t itemID = std::stoull(arr[1]);
        int rank = std::stoi(arr[2]);
        ground_truth[qID][rank] = itemID;
    }
    return ground_truth;
}

void compute_recall(
    const std::vector<std::vector<size_t>>& ground_truth,
    const std::vector<std::vector<size_t>>& retrieved,
    int top_k
) {
    int num_queries = retrieved.size();
    int total_recall = 0;
    for (int i = 0; i < num_queries; ++i) {
        const auto& gt = ground_truth[i];
        const auto& ret = retrieved[i];
        std::unordered_set<size_t> gt_set(gt.begin(), gt.begin() + top_k);
        int correct = 0;
        for (int j = 0; j < top_k; ++j) {
            if (gt_set.find(ret[j]) != gt_set.end()) {
                correct++;
            }
        }
        total_recall += correct;
    }
    double recall = static_cast<double>(total_recall) / (num_queries * top_k);
    std::cout << "Recall@" + std::to_string(top_k) + ": " << recall << std::endl;
}