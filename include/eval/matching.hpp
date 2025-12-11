// include/eval/matching.hpp
#pragma once

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "features/descriptors.hpp"

/**
 * @brief Configuration parameters for descriptor matching.
 */
struct MatchingConfig {
  bool useRatioTest = true;
  float ratioThreshold = 0.8f; // Lowe's ratio
  float epsilonPx = 3.0f;   // reprojection inlier threshold
};

/**
 * @brief One match + correctness label.
 */
struct MatchWithLabel {
  int   idxA = -1; // index into descA.keypoints
  int   idxB = -1; // index into descB.keypoints
  float distance = 0.1f;
  bool  isCorrect = false;
};

/**
 * @brief Match descriptors between two descriptor sets using BFMatcher.
 *
 * If useRatioTest=true, perform 2-NN matching with Lowe's ratio rule.
 * Otherwise, performs simple 1-NN matching.
 *
 * @param descA First descriptor set.
 * @param descB Second descriptor set.
 * @param config Matching configuration used by the matcher.
 * @return matches that have been labeled using homography.
 */
std::vector<MatchWithLabel> matchDescriptors(const DescriptorSet& descA,
                                             const DescriptorSet& descB,
                                             const MatchingConfig& config);

/**
 * @brief Use a ground-truth homography to label matches as correct or incorrect.
 *
 * A match is correct if projecting keypoint from image A to image B using
 * H_AtoB lands within epsilonPx of the matched keypoint in B.
 */
void labelMatchesWithHomography(const DescriptorSet& descA,
                                const DescriptorSet& descB,
                                const cv::Mat& H_AtoB,
                                float epsilonPx,
                                std::vector<MatchWithLabel>& matches);
