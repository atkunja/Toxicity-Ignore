#include <iostream>
#include <fstream>
#include <set>
#include "filter.h"
#include "vectorizer.h"
#include <cmath>

// Sigmoid helper
double sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

// Log feedback to a CSV file for retraining
void log_unknown(const std::string& text, int label) {
    std::ofstream out("feedback_log.csv", std::ios::app); // Append mode
    out << "\"" << text << "\"," << label << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./toxfilter \"your text here\"\n";
        return 1;
    }

    // Reconstruct the input text
    std::string text;
    for (int i = 1; i < argc; ++i) {
        if (i > 1) text += " ";
        text += argv[i];
    }

    Model model = load_model("../model.json");

    // Debug print: Show a few vocab entries
    std::cout << "First 10 vocab entries from model.json:" << std::endl;
    int count = 0;
    for (const auto& [word, idx] : model.vocab) {
        std::cout << "'" << word << "' (" << idx << ")" << std::endl;
        if (++count > 10) break;
    }

    // Check specifically for "nigger"
    if (model.vocab.count("nigger")) {
        std::cout << "'nigger' IS IN vocab!\n";
    } else {
        std::cout << "'nigger' is NOT in vocab!\n";
    }

    std::set<std::string> unknowns;
    std::vector<double> input_vec = vectorize(text, model, unknowns);

    // Run prediction
    double z = model.bias;
    for (size_t i = 0; i < input_vec.size(); ++i) {
        z += model.weights[i] * input_vec[i];
    }
    double prob = sigmoid(z);

    if (prob > 0.5) {
        std::cout << "❌ Toxic (" << prob << ")\n";
    } else {
        std::cout << "✅ Safe (" << prob << ")\n";
    }

    // Prompt for unknowns
    if (!unknowns.empty()) {
        std::cout << "Unknown word(s) detected: ";
        for (const auto& w : unknowns) std::cout << "'" << w << "' ";
        std::cout << "\nIs this phrase toxic? (y/n): ";
        std::string answer;
        std::cin >> answer;
        int label = (answer == "y" || answer == "Y") ? 1 : 0;
        log_unknown(text, label);
        std::cout << "Feedback saved! This phrase will help improve future models.\n";
    }

    return 0;
}
