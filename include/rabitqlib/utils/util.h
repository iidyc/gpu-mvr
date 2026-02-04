#pragma once

#include <chrono>
#include <iostream>
#include <unordered_map>

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

struct Stats {
    std::unordered_map<std::string, double> stats;

    void add_stat(const std::string& key, double value) {
        if (stats.find(key) != stats.end()) {
            stats[key] += value;
        } else {
            stats[key] = value;
        }
    }
};

void aggregate_stats(const std::vector<Stats>& stats) {
    Stats total_stat;
    for (const auto& stat : stats) {
        for (const auto& pair : stat.stats) {
            total_stat.add_stat(pair.first, pair.second);
        }
    }
    for (auto& pair : total_stat.stats) {
        pair.second /= stats.size();
        std::cout << "[Avg " << pair.first << "] = " << pair.second << std::endl;
    }
}