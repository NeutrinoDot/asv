// include/eval/matching.hpp
#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "features/descriptors.hpp"

/**
 * Configure for matching step. 
 */
struct MatchingConfig {
  bool useRatioTest = true;
  float ratioThreshold = 0.8f; // Lowe's ratio
  float epsilonPx = 3.0f;   // reprojection inlier threshold
};

/**
 * One match + correctness label.
 */
struct MatchWithLabel {
  int   idxA = -1; // index into descA.keypoints
  int   idxB = -1; // index into descB.keypoints
  float distance = 0.1f;
  bool  isCorrect = false;
};

/**
 * Match descriptors between two sets.
 */
std::vector<MatchWithLabel> matchDescriptors(const DescriptorSet& descA,
                                             const DescriptorSet& descB,
                                             const MatchingConfig& config);

/**
 * Label matches as correct or incorrect using a homography.
 */
void labelMatchesWithHomography(const DescriptorSet& descA,
                                const DescriptorSet& descB,
                                const cv::Mat& H_AtoB,
                                float epsilonPx,
                                std::vector<MatchWithLabel>& matches);
