#include "vectorizer.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <iostream>

// Lowercase and remove all punctuation.
std::string clean_token(const std::string& token) {
    std::string cleaned;
    for (char c : token) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            cleaned += std::tolower(static_cast<unsigned char>(c));
        }
        // else skip punctuation
    }
    return cleaned;
}

std::vector<double> vectorize(const std::string& text, const Model& model, std::set<std::string>& unknowns) {
    std::vector<double> vec(model.weights.size(), 0.0);

    std::istringstream iss(text);
    std::string token;
    while (iss >> token) {
        std::string cleaned = clean_token(token);
        if (cleaned.empty()) continue;
        auto it = model.vocab.find(cleaned);
        if (it != model.vocab.end()) {
            vec[it->second] += 1.0;
        } else {
            unknowns.insert(cleaned);
        }
    }
    return vec;
}
