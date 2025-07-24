#pragma once
#include <string>
#include <vector>
#include <set>
#include "filter.h"

// Converts input text to a feature vector using the model's vocab,
// populates a set of unknown tokens.
std::vector<double> vectorize(const std::string& text, const Model& model, std::set<std::string>& unknowns);

// Helper: cleans tokens (lowercase, remove punctuation)
std::string clean_token(const std::string& token);
