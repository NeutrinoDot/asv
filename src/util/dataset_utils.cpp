#include "util/dataset_utils.hpp"

#include <filesystem>
#include <algorithm>
#include <cmath>

std::vector<ImagePairSpec> discoverOxfordPairs(const std::string& datasetPath, float percentage) {
  std::vector<ImagePairSpec> specs;

  for (const auto& seqEntry : std::filesystem::directory_iterator(datasetPath)) {
    if (!seqEntry.is_directory()) continue;

    std::string seqName = seqEntry.path().filename().string();
    std::string seqPath = seqEntry.path().string();

    // Find all homography files H1to*p
    std::vector<int> pairIndices;
    for (const auto& file : std::filesystem::directory_iterator(seqPath)) {
      std::string filename = file.path().filename().string();
      if (filename.find("H1to") == 0 && filename.back() == 'p') {
        int idx = std::stoi(filename.substr(4, filename.length() - 5));
        pairIndices.push_back(idx);
      }
    }

    std::sort(pairIndices.begin(), pairIndices.end());

    for (int idx : pairIndices) {
      std::string pairId = seqName + "_1_" + std::to_string(idx);
      std::string img1 = seqPath + "/img1.ppm";
      std::string img2 = seqPath + "/img" + std::to_string(idx) + ".ppm";
      std::string H = seqPath + "/H1to" + std::to_string(idx) + "p";

      specs.push_back({ pairId, img1, img2, H });
    }
  }

  // Apply percentage filter
  if (percentage < 1.0f && percentage > 0.0f) {
    size_t targetSize = static_cast<size_t>(std::ceil(specs.size() * percentage));
    if (targetSize < specs.size()) {
      specs.resize(targetSize);
    }
  }

  return specs;
}
