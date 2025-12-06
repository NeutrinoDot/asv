// src/main_oxford_eval.cpp
#include <iostream>
#include <vector>
#include <iomanip>

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

// Run evaluation for a single descriptor on all image pairs
GlobalMetrics runEvaluation(const std::vector<ImagePair>& imagePairs,
                            const EvalConfig& cfg,
                            const std::string& descriptorName) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Evaluating: " << descriptorName << std::endl;
  std::cout << "========================================" << std::endl;

  auto descriptor = createDescriptor(cfg.descriptorType);
  std::vector<PairMetrics> allPairMetrics;

  const size_t total = imagePairs.size();
  for (size_t i = 0; i < total; ++i) {
    const auto& pair = imagePairs[i];
    
    int percent = static_cast<int>((i + 1) * 100 / total);
    std::cout << "\n[" << std::setw(3) << percent << "%] Processing: " << pair.id << std::endl;

    std::cout << "  → Detecting keypoints in image A..." << std::flush;
    DescriptorSet descA, descB;
    descriptor->detectAndCompute(pair.imgA, descA);
    std::cout << " " << descA.keypoints.size() << " keypoints" << std::endl;

    std::cout << "  → Detecting keypoints in image B..." << std::flush;
    descriptor->detectAndCompute(pair.imgB, descB);
    std::cout << " " << descB.keypoints.size() << " keypoints" << std::endl;

    std::cout << "  → Matching descriptors..." << std::flush;
    auto matches = matchDescriptors(descA, descB, cfg.matchingConfig);
    std::cout << " " << matches.size() << " matches" << std::endl;

    std::cout << "  → Labeling with homography..." << std::flush;
    labelMatchesWithHomography(descA, descB, pair.H_AtoB,
                               cfg.matchingConfig.epsilonPx, matches);
    int correct = 0;
    for (const auto& m : matches) if (m.isCorrect) ++correct;
    std::cout << " " << correct << "/" << matches.size() << " correct" << std::endl;

    auto pairMetrics = computePairMetrics(pair.id, matches);
    allPairMetrics.push_back(pairMetrics);
    
    std::cout << "AP: " << std::fixed << std::setprecision(4) 
              << pairMetrics.averagePrecision << std::endl;
  }

  auto globalMetrics = computeGlobalMetrics(allPairMetrics);
  std::cout << "\nmAP: " << std::fixed << std::setprecision(4) 
            << globalMetrics.mAP << std::endl;

  return globalMetrics;
}

// Print comparison table
void printComparison(const std::vector<std::string>& names,
                    const std::vector<GlobalMetrics>& results) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "PERFORMANCE COMPARISON" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::left << std::setw(15) << "Descriptor" 
            << std::right << std::setw(10) << "mAP" << std::endl;
  std::cout << "----------------------------------------" << std::endl;

  for (size_t i = 0; i < names.size(); ++i) {
    std::cout << std::left << std::setw(15) << names[i]
              << std::right << std::setw(10) << std::fixed 
              << std::setprecision(4) << results[i].mAP << std::endl;
  }
  std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
  // Configuration
  EvalConfig cfg;
  cfg.matchingConfig.useRatioTest = true;
  cfg.matchingConfig.ratioThreshold = 0.8f;
  cfg.matchingConfig.epsilonPx = 3.0f;

  std::cout << "Oxford Dataset Evaluation" << std::endl;
  std::cout << "Ratio test threshold: " << cfg.matchingConfig.ratioThreshold << std::endl;
  std::cout << "Reprojection threshold: " << cfg.matchingConfig.epsilonPx << "px" << std::endl;

  try {
    // Load dataset
    std::vector<ImagePairSpec> specs = discoverOxfordPairs("data/Oxford_dataset");
    if (specs.empty()) {
      std::cerr << "No image pairs found." << std::endl;
      return 1;
    }

    DatasetLoader loader(specs);
    std::vector<ImagePair> imagePairs = loader.loadAll();
    std::cout << "Loaded " << imagePairs.size() << " image pairs\n" << std::endl;

    // Descriptors to evaluate
    std::vector<DescriptorType> descriptorTypes = {
      DescriptorType::SIFT,
      DescriptorType::ASV_REAL,
      DescriptorType::ASV_BINARY
    };

    std::vector<std::string> descriptorNames = {
      "SIFT",
      "ASV_REAL",
      "ASV_BINARY"
    };

    // Run evaluation for each descriptor
    std::vector<GlobalMetrics> results;
    for (size_t i = 0; i < descriptorTypes.size(); ++i) {
      cfg.descriptorType = descriptorTypes[i];
      auto metrics = runEvaluation(imagePairs, cfg, descriptorNames[i]);
      results.push_back(metrics);
    }

    // Print comparison
    printComparison(descriptorNames, results);

  } catch (const cv::Exception& e) {
    std::cerr << "OpenCV exception: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
