// metrics.cpp
#include "metrics/metrics.hpp"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <util/eval_utils.hpp>

// Helper: compute ground truth correspondences
static int computeGroundTruthCorrespondences(
  const DescriptorSet& descA,
  const DescriptorSet& descB,
  const cv::Mat& H_AtoB,
  float epsilonPx)
{
  int groundTruthTotal = 0;

  for (const auto& kpA : descA.keypoints) {
    cv::Point2f projected = projectPoint(kpA.pt, H_AtoB);

    if (!std::isfinite(projected.x) || !std::isfinite(projected.y)) {
      continue;
    }

    for (const auto& kpB : descB.keypoints) {
      float dx = projected.x - kpB.pt.x;
      float dy = projected.y - kpB.pt.y;
      float dist = std::sqrt(dx * dx + dy * dy);

      if (dist <= epsilonPx) {
        groundTruthTotal++;
        break;
      }
    }
  }

  return groundTruthTotal;
}

// Helper: compute AP from PR curve using trapezoidal rule.
// INPUT: prCurv.recall (monotonic non-decreasing), prCurv.precision (same length).
// OUTPUT: scalar AP in [0, 1].
static float computeAPFromPR(const PRCurve& prCurve) {
  const auto& R = prCurve.recall;
  const auto& P = prCurve.precision;

  if (R.empty() || P.empty() || R.size() != P.size()) {
    return 0.0f;
  }

  float ap = 0.0f;
  for (size_t i = 1; i < R.size(); ++i) {
    float deltaR = R[i] - R[i - 1];
    // Approximate area with P[i] (right-rectangle) or mid-point; we choose right-rectangle.
    ap += P[i] * deltaR;
  }
  return ap;
}

PairMetrics computePairMetrics(const std::string& pairId,
                               const std::vector<MatchWithLabel>& matchesInput,
                               const DescriptorSet& descA,
                               const DescriptorSet& descB,
                               const cv::Mat& H_AtoB,
                               float epsilonPx) {
  PairMetrics result;
  result.pairId = pairId;

  if (matchesInput.empty()) {
    return result;
  }

  // Compute ground truth total correspondences
  int groundTruthTotal = computeGroundTruthCorrespondences(descA, descB, H_AtoB, epsilonPx);

  if (groundTruthTotal == 0) {
    return result;
  }

  // Copy and sort matches by ascending distance
  std::vector<MatchWithLabel> matches = matchesInput;
  std::sort(matches.begin(), matches.end(),
            [](const MatchWithLabel& a, const MatchWithLabel& b) {
              return a.distance < b.distance;
            });

  // Sweep matches computing TP/FP with TRUE recall
  std::vector<float> precision;
  std::vector<float> recall;
  precision.reserve(matches.size());
  recall.reserve(matches.size());

  int tp = 0;
  int fp = 0;

  for (size_t i = 0; i < matches.size(); ++i) {
    if (matches[i].isCorrect) {
      ++tp;
    }
    else {
      ++fp;
    }

    float prec = static_cast<float>(tp) / static_cast<float>(tp + fp);
    float rec = static_cast<float>(tp) / static_cast<float>(groundTruthTotal);

    precision.push_back(prec);
    recall.push_back(rec);
  }

  result.pr.precision = std::move(precision);
  result.pr.recall = std::move(recall);
  result.averagePrecision = computeAPFromPR(result.pr);

  return result;
}

GlobalMetrics computeGlobalMetrics(const std::vector<PairMetrics>& perPair) {
  GlobalMetrics g;
  g.perPair = perPair;

  if (perPair.empty()) {
    g.mAP = 0.0f;
    g.avgPrecision = 0.0f;
    g.avgRecall = 0.0f;
    return g;
  }

  float sumAP = 0.0f;
  float sumPrecision = 0.0f;
  float sumRecall = 0.0f;
  int count = 0;

  for (const auto& m : perPair) {
    sumAP += m.averagePrecision;

    // Get final precision and recall (last point in PR curve)
    if (!m.pr.precision.empty() && !m.pr.recall.empty()) {
      sumPrecision += m.pr.precision.back();
      sumRecall += m.pr.recall.back();
    }
    ++count;
  }

  g.mAP = (count > 0) ? (sumAP / static_cast<float>(count)) : 0.0f;
  g.avgPrecision = (count > 0) ? (sumPrecision / static_cast<float>(count)) : 0.0f;
  g.avgRecall = (count > 0) ? (sumRecall / static_cast<float>(count)) : 0.0f;

  return g;
}
