// src/main_oxford_eval.cpp
#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

#include "dataset/oxford_dataset.hpp"
#include "dataset/dataset_utils.hpp"
#include "features/descriptors.hpp"
#include "eval/matching.hpp"
#include "metrics/metrics.h"

struct EvalConfig {
  DescriptorType descriptorType = DescriptorType::SIFT;
  MatchingConfig matchingConfig;
};

int main(int argc, char** argv) {
  EvalConfig cfg;
  cfg.descriptorType = DescriptorType::ASV_REAL;
  cfg.matchingConfig.useRatioTest = true;
  cfg.matchingConfig.ratioThreshold = 0.8f;
  cfg.matchingConfig.epsilonPx = 3.0f;

  std::cout << "Descriptor: ASV_REAL" << std::endl;
  std::cout << "Ratio test: " << cfg.matchingConfig.ratioThreshold << std::endl;
  std::cout << "Epsilon: " << cfg.matchingConfig.epsilonPx << "px\n" << std::endl;

  std::vector<ImagePairSpec> imagePairSpecs = discoverOxfordPairs("data/Oxford_dataset");

  if (imagePairSpecs.empty()) {
    std::cerr << "No image pair specs configured. Please populate 'imagePairSpecs'." << std::endl;
    return 1;
  }

  try {
    DatasetLoader loader(imagePairSpecs);
    std::vector<ImagePair> imagePairs = loader.loadAll();

    auto descriptor = createDescriptor(cfg.descriptorType);

    std::vector<PairMetrics> allPairMetrics;

    for (const auto& pair : imagePairs) {
      std::cout << "Evaluating pair: " << pair.id << std::endl;

      DescriptorSet descA, descB;
      descriptor->detectAndCompute(pair.imgA, descA);
      descriptor->detectAndCompute(pair.imgB, descB);

      auto matches = matchDescriptors(descA, descB, cfg.matchingConfig);
      labelMatchesWithHomography(descA, descB, pair.H_AtoB,
                                 cfg.matchingConfig.epsilonPx, matches);

      auto pairMetrics = computePairMetrics(pair.id, matches);
      allPairMetrics.push_back(pairMetrics);
      std::cout << "  AP: " << pairMetrics.averagePrecision << std::endl;
    }

    auto globalMetrics = computeGlobalMetrics(allPairMetrics);
    std::cout << "\nmAP: " << globalMetrics.mAP << std::endl;
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
