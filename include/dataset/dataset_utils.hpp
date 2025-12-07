#pragma once

#include <string>
#include <vector>
#include "oxford_dataset.hpp"

// Load image pairs in Oxford dataset directory
std::vector<ImagePairSpec> discoverOxfordPairs(const std::string& datasetPath, float percentage = 1.0f);
