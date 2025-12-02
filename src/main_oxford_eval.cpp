// src/main_oxford_eval.cpp
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

#include "dataset/oxford_dataset.hpp"
#include "features/descriptors.hpp"
#include "eval/matching.hpp"

// TODO: add metrics (PR, AP, mAP)

struct EvalConfig {
  DescriptorType descriptorType = DescriptorType::SIFT;
  MatchingConfig matchingConfig;
};

static int main_test(int argc, char** argv) {
  EvalConfig cfg;
  cfg.descriptorType = DescriptorType::ASV_REAL;
  cfg.matchingConfig.useRatioTest = true;
  cfg.matchingConfig.ratioThreshold = 0.8f;
  cfg.matchingConfig.epsilonPx = 3.0f;

  std::vector<ImagePairSpec> imagePairSpecs = {
      {"graf_1_2", "data/oxford/raw/graf/img1.ppm",
                   "data/oxford/raw/graf/img2.ppm",
                   "data/oxford/raw/graf/H1to2p"},
                   // Add more image pairs as needed
  };

  if (imagePairSpecs.empty()) {
    std::cerr << "No image pair specs configured. Please populate 'imagePairSpecs'." << std::endl;
    return 1;
  }

  try {
    DatasetLoader loader(imagePairSpecs);
    std::vector<ImagePair> imagePairs = loader.loadAll();

    auto descriptor = createDescriptor(cfg.descriptorType);

    for (const auto& pair : imagePairs) {
      std::cout << "Evaluating pair: " << pair.id << std::endl;

      DescriptorSet descA, descB;
      descriptor->detectAndCompute(pair.imgA, descA);
      descriptor->detectAndCompute(pair.imgB, descB);

      auto matches = matchDescriptors(descA, descB, cfg.matchingConfig);
      labelMatchesWithHomography(descA, descB, pair.H_AtoB,
                                 cfg.matchingConfig.epsilonPx, matches);

      // TODO: Compute PR curve + AP for this pair, then aggregate mAP.
    }
  }
  catch (const cv::Exception& e) {
    std::cerr << "OpenCV exception: " << e.what() << std::endl;
    return 1;
  }
  catch (const std::exception& e) {
    std::cerr << "Std exception: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
