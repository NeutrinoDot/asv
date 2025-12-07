// src/main_oxford_eval.cpp
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>

#include <opencv2/core.hpp>

#include "dataset/oxford_dataset.hpp"
#include "dataset/dataset_utils.hpp"
#include "features/descriptors.hpp"
#include "eval/matching.hpp"
#include "metrics/metrics.hpp"

struct EvalConfig {
  DescriptorType descriptorType = DescriptorType::SIFT;
  MatchingConfig matchingConfig;
  ASVConfig asvConfig;
};

// Run evaluation for a single descriptor on all image pairs
static GlobalMetrics runEvaluation(const std::vector<ImagePair>& imagePairs,
                            const EvalConfig& cfg,
                            const std::string& descriptorName) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "Evaluating: " << descriptorName << std::endl;
  std::cout << "========================================" << std::endl;
  
  // Print ASV config for ASV descriptors
  if (cfg.descriptorType == DescriptorType::ASV_REAL || 
      cfg.descriptorType == DescriptorType::ASV_BINARY) {
    std::cout << "ASV Config:" << std::endl;
    std::cout << "  nScales: " << cfg.asvConfig.nScales << std::endl;
    std::cout << "  scale_min: " << cfg.asvConfig.scale_min << std::endl;
    std::cout << "  scale_max: " << cfg.asvConfig.scale_max << std::endl;
    std::cout << "  nThreshold1: " << cfg.asvConfig.nThreshold1 << std::endl;
    std::cout << "  nThreshold2: " << cfg.asvConfig.nThreshold2 << std::endl;
    std::cout << "  detectorType: " << cfg.asvConfig.detectorType 
              << " (0=SIFT, 1=ORB, 2=BRISK)" << std::endl;
  }

  auto descriptor = createDescriptor(cfg.descriptorType, cfg.asvConfig);
  std::vector<PairMetrics> allPairMetrics;

  const size_t total = imagePairs.size();
  double totalTime = 0.0;
  
  for (size_t i = 0; i < total; ++i) {
    const auto& pair = imagePairs[i];
    
    int percent = static_cast<int>((i + 1) * 100 / total);
    std::cout << "\n[" << std::setw(3) << percent << "%] Processing: " << pair.id << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();
    
    std::cout << "  → Detecting keypoints in image A..." << std::flush;
    DescriptorSet descA, descB;
    descriptor->detectAndCompute(pair.imgA, descA);
    auto t1 = std::chrono::high_resolution_clock::now();
    double timeProcessingImageA = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " " << descA.keypoints.size() << " keypoints (" 
              << std::fixed << std::setprecision(1) << timeProcessingImageA << "ms)" << std::endl;

    std::cout << "  → Detecting keypoints in image B..." << std::flush;
    descriptor->detectAndCompute(pair.imgB, descB);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeProcessingImageB = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << " " << descB.keypoints.size() << " keypoints (" 
              << std::fixed << std::setprecision(1) << timeProcessingImageB << "ms)" << std::endl;

    std::cout << "  → Matching descriptors..." << std::flush;
    auto matches = matchDescriptors(descA, descB, cfg.matchingConfig);
    auto t3 = std::chrono::high_resolution_clock::now();
    double timeMatch = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << " " << matches.size() << " matches (" 
              << std::fixed << std::setprecision(1) << timeMatch << "ms)" << std::endl;

    std::cout << "  → Labeling with homography..." << std::flush;
    labelMatchesWithHomography(descA, descB, pair.H_AtoB,
                               cfg.matchingConfig.epsilonPx, matches);
    auto t4 = std::chrono::high_resolution_clock::now();
    double timeLabel = std::chrono::duration<double, std::milli>(t4 - t3).count();
    int correct = 0;
    for (const auto& m : matches) if (m.isCorrect) ++correct;
    std::cout << " " << correct << "/" << matches.size() << " correct (" 
              << std::fixed << std::setprecision(1) << timeLabel << "ms)" << std::endl;

    auto pairMetrics = computePairMetrics(pair.id, matches);
    allPairMetrics.push_back(pairMetrics);
    
    double pairTime = std::chrono::duration<double, std::milli>(t4 - t0).count();
    totalTime += pairTime;
    
    std::cout << "AP: " << std::fixed << std::setprecision(4) 
              << pairMetrics.averagePrecision << " | Total: " 
              << std::fixed << std::setprecision(1) << pairTime << "ms" << std::endl;
  }

  auto globalMetrics = computeGlobalMetrics(allPairMetrics);
  double avgTime = totalTime / total;
  globalMetrics.avgTimePerPair = avgTime;
  globalMetrics.totalTime = totalTime;
  
  std::cout << "\nmAP: " << std::fixed << std::setprecision(4) 
            << globalMetrics.mAP << std::endl;
  std::cout << "Average time per pair: " << std::fixed << std::setprecision(1) 
            << avgTime << "ms" << std::endl;
  std::cout << "Total time: " << std::fixed << std::setprecision(1) 
            << totalTime << "ms" << std::endl;

  return globalMetrics;
}

// Print comparison table
static void printComparison(const std::vector<std::string>& names,
                    const std::vector<GlobalMetrics>& results) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "PERFORMANCE COMPARISON" << std::endl;
  std::cout << "========================================" << std::endl;
  std::cout << std::left << std::setw(15) << "Descriptor" 
            << std::right << std::setw(10) << "mAP"
            << std::setw(15) << "Avg Time (ms)"
            << std::setw(15) << "Total (ms)" << std::endl;
  std::cout << "--------------------------------------------------------" << std::endl;

  for (size_t i = 0; i < names.size(); ++i) {
    std::cout << std::left << std::setw(15) << names[i]
              << std::right << std::setw(10) << std::fixed << std::setprecision(4) << results[i].mAP
              << std::setw(15) << std::fixed << std::setprecision(1) << results[i].avgTimePerPair
              << std::setw(15) << std::fixed << std::setprecision(1) << results[i].totalTime << std::endl;
  }
  std::cout << "========================================\n" << std::endl;
}

int main(int argc, char** argv) {
  // Configuration
  EvalConfig cfg;
  
  // ASV configuration
  cfg.asvConfig.nScales = 5;
  cfg.asvConfig.scale_min = 0.7f;
  cfg.asvConfig.scale_max = 1.4f;
  cfg.asvConfig.nThreshold1 = 1;
  cfg.asvConfig.nThreshold2 = 1;
  cfg.asvConfig.detectorType = 0; // 0=SIFT, 1=ORB, 2=BRISK
  
  // Matching configuration
  cfg.matchingConfig.useRatioTest = true;
  cfg.matchingConfig.ratioThreshold = 0.8f;
  cfg.matchingConfig.epsilonPx = 3.0f;

  std::cout << "Oxford Dataset Evaluation" << std::endl;
  std::cout << "\nASV Configuration:" << std::endl;
  std::cout << "  nScales: " << cfg.asvConfig.nScales << std::endl;
  std::cout << "  scale_min: " << cfg.asvConfig.scale_min << std::endl;
  std::cout << "  scale_max: " << cfg.asvConfig.scale_max << std::endl;
  std::cout << "  nThreshold1: " << cfg.asvConfig.nThreshold1 << std::endl;
  std::cout << "  nThreshold2: " << cfg.asvConfig.nThreshold2 << std::endl;
  std::cout << "  detectorType: " << cfg.asvConfig.detectorType << " (0=SIFT, 1=ORB, 2=BRISK)" << std::endl;
  std::cout << "\nMatching Configuration:" << std::endl;
  std::cout << "  Ratio test threshold: " << cfg.matchingConfig.ratioThreshold << std::endl;
  std::cout << "  Reprojection threshold: " << cfg.matchingConfig.epsilonPx << "px" << std::endl;

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
