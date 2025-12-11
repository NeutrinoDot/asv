// src/eval/matching.cpp
#include "eval/matching.hpp"

#include <opencv2/features2d.hpp>
#include <cmath>
#include <limits>
#include <util/eval_utils.hpp>

std::vector<MatchWithLabel> matchDescriptors(const DescriptorSet& descA,
                                             const DescriptorSet& descB,
                                             const MatchingConfig& config) {
  std::vector<MatchWithLabel> result;

  if (descA.descriptors.empty() || descB.descriptors.empty()) {
    return result;
  }

  CV_Assert(descA.descriptors.type() == descB.descriptors.type());
  int normType = inferNormType(descA.descriptors);

  cv::BFMatcher matcher(normType, /*crossCheck*/ false);

  if (config.useRatioTest) {
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(descA.descriptors, descB.descriptors, knnMatches, 2);

    for (const auto& knn : knnMatches) {
      if (knn.size() < 2) continue;

      const cv::DMatch& m1 = knn[0];
      const cv::DMatch& m2 = knn[1];

      if (m1.distance < config.ratioThreshold * m2.distance) {
        MatchWithLabel mwl;
        mwl.idxA = m1.queryIdx;
        mwl.idxB = m1.trainIdx;
        mwl.distance = m1.distance;
        mwl.isCorrect = false;
        result.push_back(mwl);
      }
    }
  }
  else {
    std::vector<cv::DMatch> matches;
    matcher.match(descA.descriptors, descB.descriptors, matches);

    for (const auto& m : matches) {
      MatchWithLabel mwl;
      mwl.idxA = m.queryIdx;
      mwl.idxB = m.trainIdx;
      mwl.distance = m.distance;
      mwl.isCorrect = false;
      result.push_back(mwl);
    }
  }

  return result;
}

void labelMatchesWithHomography(const DescriptorSet& descA,
                                const DescriptorSet& descB,
                                const cv::Mat& H_AtoB,
                                float epsilonPx,
                                std::vector<MatchWithLabel>& matches) {
  if (matches.empty()) {
    return;
  }

  for (auto& m : matches) {
    // Ensure indices are valid
    if (m.idxA < 0 || m.idxA >= static_cast<int>(descA.keypoints.size()) ||
        m.idxB < 0 || m.idxB >= static_cast<int>(descB.keypoints.size())) {
      m.isCorrect = false;
      continue;
    }

    const cv::KeyPoint& kpA = descA.keypoints[m.idxA];
    const cv::KeyPoint& kpB = descB.keypoints[m.idxB];

    cv::Point2f projected = projectPoint(kpA.pt, H_AtoB);

    // If projection was invalid, mark as incorrect
    if (!std::isfinite(projected.x) || !std::isfinite(projected.y)) {
      m.isCorrect = false;
      continue;
    }

    // Compute reprojection error
    float dx = projected.x - kpB.pt.x;
    float dy = projected.y - kpB.pt.y;
    float error = std::sqrt(dx * dx + dy * dy);

    m.isCorrect = (error <= epsilonPx);
  }
}
