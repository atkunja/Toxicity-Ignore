#include "filter.h"
#include "json.hpp"
#include <fstream>
#include <iostream>

using json = nlohmann::json;

Model load_model(const std::string& filename) {
    Model model;
    std::ifstream infile(filename);
    json j;
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open model file: " << filename << std::endl;
        exit(1);
    }
    infile >> j;

    model.bias = j["bias"];

    // Load weights
    for (auto& w : j["weights"]) {
        model.weights.push_back(w);
    }

    // Load vocab
    for (auto& [word, index] : j["vocab"].items()) {
        model.vocab[word] = index;
    }

    return model;
}
