// metrics.cpp
#include "metrics.h"

#include <algorithm>
#include <numeric>

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
                               const std::vector<MatchWithLabel>& matchesInput) {
    PairMetrics result;
    result.pairId = pairId;

    if (matchesInput.empty()) {
        return result; // AP = 0, PR empty
    }

    // Copy and sort matches by ascending distance (best matches first).
    std::vector<MatchWithLabel> matches = matchesInput;
    std::sort(matches.begin(), matches.end(),
              [](const MatchWithLabel& a, const MatchWithLabel& b) {
                  return a.distance < b.distance;
              });

    // Count total number of positives (correct matches).
    int totalPositives = 0;
    for (const auto& m : matches) {
        if (m.isCorrect) {
            ++totalPositives;
        }
    }

    if (totalPositives == 0) {
        // No positives: define AP = 0, PR empty for this pair.
        return result;
    }

    // Sweep matches in rank order, computing cumulative TP/FP.
    std::vector<float> precision;
    std::vector<float> recall;
    precision.reserve(matches.size());
    recall.reserve(matches.size());

    int tp = 0;
    int fp = 0;

    for (size_t i = 0; i < matches.size(); ++i) {
        if (matches[i].isCorrect) {
            ++tp;
        } else {
            ++fp;
        }

        float prec = static_cast<float>(tp) / static_cast<float>(tp + fp);
        float rec  = static_cast<float>(tp) / static_cast<float>(totalPositives);

        precision.push_back(prec);
        recall.push_back(rec);
    }

    // Fill PR curve.
    result.pr.precision = std::move(precision);
    result.pr.recall = std::move(recall);

    // Compute AP from PR curve.
    result.averagePrecision = computeAPFromPR(result.pr);

    return result;
}

GlobalMetrics computeGlobalMetrics(const std::vector<PairMetrics>& perPair) {
    GlobalMetrics g;
    g.perPair = perPair;

    if (perPair.empty()) {
        g.mAP = 0.0f;
        return g;
    }

    float sumAP = 0.0f;
    int count = 0;
    for (const auto& m : perPair) {
        sumAP += m.averagePrecision;
        ++count;
    }

    g.mAP = (count > 0) ? (sumAP / static_cast<float>(count)) : 0.0f;
    return g;
}