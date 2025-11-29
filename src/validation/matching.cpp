// matching.cpp
#include "matching.h"

#include <opencv2/features2d.hpp>
#include <cmath>

// Helper: project a point (x, y, 1) in image A to image B using H_AtoB.
//
// INPUT:  ptA   : cv::Point2f in image A (non-homogeneous).
//         H_AtoB: 3x3 homography, CV_64F.
// OUTPUT: cv::Point2f in image B (non-homogeneous), normalized by w.
static cv::Point2f projectPoint(const cv::Point2f& ptA, const cv::Mat& H_AtoB) {
    CV_Assert(H_AtoB.rows == 3 && H_AtoB.cols == 3);
    CV_Assert(H_AtoB.type() == CV_64F);

    double x = ptA.x;
    double y = ptA.y;

    const double* h = H_AtoB.ptr<double>(0);
    double X = h[0] * x + h[1] * y + h[2];
    double Y = h[3] * x + h[4] * y + h[5];
    double W = h[6] * x + h[7] * y + h[8];

    if (std::abs(W) < 1e-9) {
        // Invalid mapping; return something, caller can interpret large error later.
        return cv::Point2f(std::numeric_limits<float>::infinity(),
                           std::numeric_limits<float>::infinity());
    }

    return cv::Point2f(static_cast<float>(X / W),
                       static_cast<float>(Y / W));
}

std::vector<MatchWithLabel> matchDescriptors(const DescriptorSet& descA,
                                             const DescriptorSet& descB,
                                             const MatchingConfig& config) {
    std::vector<MatchWithLabel> result;

    if (descA.descriptors.empty() || descB.descriptors.empty()) {
        // No descriptors: return empty set, this is not an error.
        return result;
    }

    // Brute-force matcher with L2 norm for float descriptors (SIFT-style).
    cv::BFMatcher matcher(cv::NORM_L2, /*crossCheck=*/false);

    if (config.useRatioTest) {
        // 2-NN matches for ratio test.
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher.knnMatch(descA.descriptors, descB.descriptors, knnMatches, 2);

        for (const auto& knn : knnMatches) {
            if (knn.size() < 2) {
                continue;
            }
            const cv::DMatch& m1 = knn[0];
            const cv::DMatch& m2 = knn[1];

            if (m1.distance < config.ratioThreshold * m2.distance) {
                MatchWithLabel mwl;
                mwl.idxA = m1.queryIdx;
                mwl.idxB = m1.trainIdx;
                mwl.distance = m1.distance;
                mwl.isCorrect = false; // labeled later
                result.push_back(mwl);
            }
        }
    } else {
        // Simple nearest-neighbor matches without ratio.
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

    // Iterate over all matches and compute reprojection error.
    for (auto& m : matches) {
        // Defensive checks for index validity.
        if (m.idxA < 0 || m.idxA >= static_cast<int>(descA.keypoints.size()) ||
            m.idxB < 0 || m.idxB >= static_cast<int>(descB.keypoints.size())) {
            m.isCorrect = false;
            continue;
        }

        const cv::KeyPoint& kpA = descA.keypoints[m.idxA];
        const cv::KeyPoint& kpB = descB.keypoints[m.idxB];

        cv::Point2f projected = projectPoint(kpA.pt, H_AtoB);

        // If projection is invalid, treat as incorrect.
        if (!std::isfinite(projected.x) || !std::isfinite(projected.y)) {
            m.isCorrect = false;
            continue;
        }

        float dx = projected.x - kpB.pt.x;
        float dy = projected.y - kpB.pt.y;
        float error = std::sqrt(dx * dx + dy * dy);

        // Correct if reprojection error is within threshold epsilonPx.
        // SOURCE: Going with common default value of ε = 3 pixels; we default to that in MatchingConfig.
        m.isCorrect = (error <= epsilonPx);
    }
}
