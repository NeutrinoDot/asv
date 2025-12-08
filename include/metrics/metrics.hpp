// include/metrics/metrics.hpp
// -------------------------------------------------------------------------------------------------
// Computes precision-recall (PR) curve, Average Precision (AP) per image pair, and mean AP (mAP).
//
// INPUT:
//   For a single image pair:
//     - vector<MatchWithLabel> matches
//       * distance  : used as ranking score (lower distance = better match).
//       * isCorrect : truth labels from homography.
//
// OUTPUT:
//   PairMetrics:
//     - PR curve (precision[], recall[]).
//     - AP (scalar).
//
//   GlobalMetrics:
//     - per-pair metrics.
//     - mAP (mean of APs).
//
// ASSUMPTIONS:
// - We treat smaller descriptor distances as higher confidence, and rank accordingly.
// - AP is computed via trapezoidal integration over the PR curve constructed by sweeping
//   over sorted matches (continuous version, not 11-point approximation).
//
#pragma once

#include <string>
#include <vector>
#include "eval/matching.hpp"
#include "features/descriptors.hpp"

struct PRCurve {
    std::vector<float> recall;
    std::vector<float> precision;
};

struct PairMetrics {
    std::string pairId;
    PRCurve pr;
    float averagePrecision = 0.0f; // AP for this pair
};

struct GlobalMetrics {
    std::vector<PairMetrics> perPair;
    float mAP = 0.0f; // mean AP across all pairs
    float avgPrecision = 0.0f; // average precision at final threshold
    float avgRecall = 0.0f; // average recall at final threshold
    double avgTimePerPair = 0.0; // average time per pair in ms
    double totalTime = 0.0; // total time in ms
};

// Compute PR curve + AP for a single image pair.
//
// INPUT: pairId, matches (with distance + isCorrect).
// OUTPUT: PairMetrics with filled PRCurve and AP.
PairMetrics computePairMetrics(const std::string& pairId,
                               const std::vector<MatchWithLabel>& matches);

// Compute PR curve + AP for a single image pair with ground truth total.
//
// INPUT: pairId, matches, descA, descB, H_AtoB, epsilonPx
// OUTPUT: PairMetrics with filled PRCurve and AP using true recall.
PairMetrics computePairMetrics(const std::string& pairId,
                               const std::vector<MatchWithLabel>& matches,
                               const DescriptorSet& descA,
                               const DescriptorSet& descB,
                               const cv::Mat& H_AtoB,
                               float epsilonPx);

// Compute mean AP across all pairs.
//
// INPUT: perPair (one PairMetrics per image pair).
// OUTPUT: GlobalMetrics with mAP.
GlobalMetrics computeGlobalMetrics(const std::vector<PairMetrics>& perPair);

