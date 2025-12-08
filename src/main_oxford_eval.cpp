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

  const int total = static_cast<int>(imagePairs.size());
  double totalTime = 0.0;

  for (int i = 0; i < total; ++i) {
    const auto& pair = imagePairs[i];

    int percent = (i + 1) * 100 / total;
    std::cout << "\n[" << std::setw(3) << percent << "%] Processing: " << pair.id << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    std::cout << " -> Detecting keypoints in image A..." << std::flush;
    DescriptorSet descA, descB;
    descriptor->detectAndCompute(pair.imgA, descA);
    auto t1 = std::chrono::high_resolution_clock::now();
    double timeProcessingImageA = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << " " << descA.keypoints.size() << " keypoints ("
      << std::fixed << std::setprecision(1) << timeProcessingImageA << "ms)" << std::endl;

    std::cout << " -> Detecting keypoints in image B..." << std::flush;
    descriptor->detectAndCompute(pair.imgB, descB);
    auto t2 = std::chrono::high_resolution_clock::now();
    double timeProcessingImageB = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::cout << " " << descB.keypoints.size() << " keypoints ("
      << std::fixed << std::setprecision(1) << timeProcessingImageB << "ms)" << std::endl;

    std::cout << " -> Matching descriptors..." << std::flush;
    auto matches = matchDescriptors(descA, descB, cfg.matchingConfig);
    auto t3 = std::chrono::high_resolution_clock::now();
    double timeMatch = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << " " << matches.size() << " matches ("
      << std::fixed << std::setprecision(1) << timeMatch << "ms)" << std::endl;

    std::cout << " -> Labeling with homography..." << std::flush;
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

// TABLE 1: SIFT + ASV-SIFT threshold sweep (real-valued, SIFT detector only)
static void printTable1(const std::vector<std::string>& names,
                        const std::vector<int>& threshValues,
                        const std::vector<GlobalMetrics>& metrics)
{
  std::cout << "\n==================== TABLE 1 - ASV-SIFT THRESHOLD SWEEP ====================\n";
  std::cout << std::left << std::setw(20) << "Descriptor"
    << std::setw(15) << "nThreshold1"
    << std::setw(12) << "mAP"
    << std::setw(15) << "Avg Time (ms)"
    << "\n--------------------------------------------------------------------------\n";

  for (size_t i = 0; i < names.size(); i++) {
    std::string thrStr = (threshValues[i] < 0 ? "-" : std::to_string(threshValues[i]));
    std::cout << std::left << std::setw(20) << names[i]
      << std::setw(15) << thrStr
      << std::setw(12) << std::fixed << std::setprecision(4) << metrics[i].mAP
      << std::setw(15) << std::fixed << std::setprecision(1) << metrics[i].avgTimePerPair
      << "\n";
  }

  std::cout << "============================================================================\n\n";
}

// TABLE 2: Real-valued descriptor comparison
static void printTable2(const std::vector<std::string>& names,
                        const std::vector<GlobalMetrics>& metrics)
{
  std::cout << "\n==================== TABLE 2 - REAL-VALUED DESCRIPTORS ====================\n";
  std::cout << std::left << std::setw(20) << "Descriptor"
    << std::setw(12) << "mAP"
    << std::setw(15) << "Avg Time (ms)"
    << "\n--------------------------------------------------------------------------\n";

  for (size_t i = 0; i < names.size(); i++) {
    std::cout << std::left << std::setw(20) << names[i]
      << std::setw(12) << std::fixed << std::setprecision(4) << metrics[i].mAP
      << std::setw(15) << std::fixed << std::setprecision(1) << metrics[i].avgTimePerPair
      << "\n";
  }

  std::cout << "============================================================================\n\n";
}

// TABLE 3: Binary descriptor comparison
static void printTable3(const std::vector<std::string>& names,
                        const std::vector<GlobalMetrics>& metrics)
{
  std::cout << "\n==================== TABLE 3 - BINARY DESCRIPTORS =========================\n";
  std::cout << std::left << std::setw(25) << "Descriptor"
    << std::setw(12) << "mAP"
    << std::setw(15) << "Avg Time (ms)"
    << "\n--------------------------------------------------------------------------\n";

  for (size_t i = 0; i < names.size(); i++) {
    std::cout << std::left << std::setw(25) << names[i]
      << std::setw(12) << std::fixed << std::setprecision(4) << metrics[i].mAP
      << std::setw(15) << std::fixed << std::setprecision(1) << metrics[i].avgTimePerPair
      << "\n";
  }

  std::cout << "============================================================================\n\n";
}

int main(int argc, char** argv) {
  EvalConfig cfg;

  // Global ASV settings (parameters suggested by authors)
  cfg.asvConfig.nScales = 10;
  cfg.asvConfig.scale_min = 1.0f / 6.0f;
  cfg.asvConfig.scale_max = 3.0f;

  // nThreshold2 default for Tables 2 & 3
  const int defaultT2 = 3;

  // Matching configuration
  cfg.matchingConfig.useRatioTest = true;
  cfg.matchingConfig.ratioThreshold = 0.8f;
  cfg.matchingConfig.epsilonPx = 3.0f;

  // Load Oxford dataset
  float datasetPercentage = 0.05f;
  auto specs = discoverOxfordPairs("data/Oxford_dataset", datasetPercentage);
  if (specs.empty()) {
    std::cerr << "No image pairs found.\n";
    return 1;
  }

  DatasetLoader loader(specs);
  auto imagePairs = loader.loadAll();
  std::cout << "Loaded " << imagePairs.size() << " image pairs.\n\n";

  std::vector<std::string> baselineNames = {
    "SIFT",
    "ORB",
    "BRISK"
  };

  cfg.descriptorType = DescriptorType::SIFT;
  cfg.asvConfig.detectorType = 0; // SIFT detector
  GlobalMetrics sift_metrics = runEvaluation(imagePairs, cfg, "SIFT");

  EvalConfig cfgORB = cfg;
  cfgORB.descriptorType = DescriptorType::ORB;
  cfgORB.asvConfig.detectorType = 1;
  GlobalMetrics orb_metrics = runEvaluation(imagePairs, cfgORB, "ORB");

  EvalConfig cfgBRISK = cfg;
  cfgBRISK.descriptorType = DescriptorType::BRISK;
  cfgBRISK.asvConfig.detectorType = 2;
  GlobalMetrics brisk_metrics = runEvaluation(imagePairs, cfgBRISK, "BRISK");

  // ==========================   TABLE 1   ================================
  // SIFT + ASV-SIFT threshold sweep (Real-valued descriptors)
  //
  // Entries:
  //  - SIFT baseline
  //  - ASV_SIFT (1S) → nThreshold1 = 1
  //  - ASV_SIFT (1M) → nThreshold1 ∈ {3, 7, 15}
  //
  // nThreshold2 = 1 for all ASV in Table 1 (ignored in stage 1)
  // ========================================================================
  std::vector<std::string> table1Names = {
      "SIFT",
      "ASV_SIFT_1S",
      "ASV_SIFT_1M_3",
      "ASV_SIFT_1M_7",
      "ASV_SIFT_1M_15"
  };

  // -1 = baseline SIFT (no threshold); others are ASV nThreshold1 values
  std::vector<int> table1Thresholds = { -1, 1, 3, 7, 15 };
  std::vector<GlobalMetrics> table1Results(table1Names.size());

  // --- SIFT baseline (row 0) ---
  table1Results[0] = sift_metrics;

  // --- ASV-SIFT variants (rows 1-4) ---
  for (size_t i = 1; i < table1Names.size(); i++) {
    EvalConfig cfgASV = cfg;
    cfgASV.descriptorType = DescriptorType::ASV_REAL;
    cfgASV.asvConfig.detectorType = 0; // SIFT detector
    cfgASV.asvConfig.nThreshold1 = table1Thresholds[i];
    cfgASV.asvConfig.nThreshold2 = 1; // ignored in stage 1

    table1Results[i] = runEvaluation(imagePairs, cfgASV, table1Names[i]);
  }

  // ------------------------------------------------------------------------
  // Pick BEST ASV nThreshold1 from Table 1
  // ------------------------------------------------------------------------
  int bestT1 = table1Thresholds[1];
  double bestMAP = table1Results[1].mAP;
  GlobalMetrics bestASVMetrics = table1Results[1];

  for (size_t i = 2; i < table1Names.size(); ++i) {
    if (table1Results[i].mAP > bestMAP) {
      bestMAP = table1Results[i].mAP;
      bestT1 = table1Thresholds[i];
      bestASVMetrics = table1Results[i];
    }
  }

  std::cout << "Best ASV_SIFT nThreshold1 from Table 1: "
    << bestT1 << " (mAP = " << std::fixed << std::setprecision(4)
    << bestMAP << ")\n\n";

  // ==========================   TABLE 2   ================================
  // Real-valued descriptor comparison (parallel)
  //
  // Entries:
  //  - SIFT
  //  - ASV_SIFT_REAL
  //  - ASV_ORB_REAL
  //  - ASV_BRISK_REAL
  //
  // All ASV use:
  //  - nThreshold1 = bestT1 (from Table 1)
  //  - nThreshold2 = defaultT2 (3)
  // ========================================================================
  std::vector<std::string> table2Names = {
      "SIFT",
      "ASV_SIFT_REAL",
      "ASV_ORB_REAL",
      "ASV_BRISK_REAL"
  };

  std::vector<GlobalMetrics> table2Results(table2Names.size());

  // --- SIFT baseline ---
  table2Results[0] = sift_metrics;
  table2Results[1] = bestASVMetrics;

  for (int i = 2; i <= 3; i++) {
    EvalConfig cfgASV = cfg;
    cfgASV.descriptorType = DescriptorType::ASV_REAL;
    cfgASV.asvConfig.nThreshold1 = bestT1;
    cfgASV.asvConfig.nThreshold2 = defaultT2;

    // Map row index to detector type:
    // 1 → SIFT (0), 2 → ORB (1), 3 → BRISK (2)
    cfgASV.asvConfig.detectorType = (i - 1);

    table2Results[i] = runEvaluation(imagePairs, cfgASV, table2Names[i]);
  }

  // ==========================   TABLE 3   ================================
  // Binary descriptor comparison (parallel)
  //
  // Entries:
  //  - ORB (native)
  //  - BRISK (native)
  //  - ASV_SIFT_BINARY
  //  - ASV_ORB_BINARY
  //  - ASV_BRISK_BINARY
  //
  // All ASV_BINARY use:
  //  - nThreshold1 = bestT1 (from Table 1)
  //  - nThreshold2 = defaultT2 (3)
  // ========================================================================
  std::vector<std::string> table3Names = {
      "ORB",
      "BRISK",
      "ASV_SIFT_BINARY",
      "ASV_ORB_BINARY",
      "ASV_BRISK_BINARY"
  };

  std::vector<GlobalMetrics> table3Results(table3Names.size());

  // Baseline ORB & BRISK
  table3Results[0] = orb_metrics;
  table3Results[1] = brisk_metrics;

  for (int i = 2; i <= 4; i++) {
    EvalConfig cfgASV = cfg;
    cfgASV.descriptorType = DescriptorType::ASV_BINARY;
    cfgASV.asvConfig.nThreshold1 = bestT1;
    cfgASV.asvConfig.nThreshold2 = defaultT2;

    // Map rows to detector types:
    // 2 → SIFT (0), 3 → ORB (1), 4 → BRISK (2)
    cfgASV.asvConfig.detectorType = (i - 2);

    table3Results[i] = runEvaluation(imagePairs, cfgASV, table3Names[i]);
  }

  printTable1(table1Names, table1Thresholds, table1Results);
  printTable2(table2Names, table2Results);
  printTable3(table3Names, table3Results);

  return 0;
}

