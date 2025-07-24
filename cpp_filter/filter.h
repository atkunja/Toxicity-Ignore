#pragma once
#include <string>
#include <vector>
#include <unordered_map>

struct Model {
    std::vector<double> weights;
    double bias;
    std::unordered_map<std::string, int> vocab;
};

Model load_model(const std::string& filename);
