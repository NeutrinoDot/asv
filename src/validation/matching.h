// matching.h
// -------------------------------------------------------------------------------------------------
// Provides utilities to:
//   - Match descriptors between two images (BFMatcher + L2).
//   - Label each match as "correct" or not using ground-truth homography and reprojection error.
//
// INPUT:
//   DescriptorSet descA, descB
//   Homography H_AtoB (3x3, CV_64F) mapping points from image A → image B
//
// OUTPUT:
//   std::vector<MatchWithLabel>:
//       idxA, idxB: indices into descA.keypoints / descB.keypoints
//       distance  : descriptor distance used by the matcher
//       isCorrect : correctness label based on reprojection error
//
// ASSUMPTIONS:
// - Only nearest-neighbor matches (no symmetric or cross-check by default).
// - Reprojection error threshold epsilonPx (e.g., 3 pixels) comes from standard choice:
//     INTUITION: 3 px is commonly used in literature as a "tight" correctness threshold for
//                planar homography-based matching (HPatches / Oxford-style).
//
#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include "descriptors.h"

struct MatchWithLabel {
    int idxA;        // index into descA.keypoints
    int idxB;        // index into descB.keypoints
    float distance;  // descriptor distance (L2 norm)
    bool isCorrect;  // filled after homography-based check
};

// Configuration for matching step.
struct MatchingConfig {
    // Whether to use Lowe's ratio test.
    bool useRatioTest = true;

    // Lowe's ratio threshold (typical values: 0.7 - 0.8).
    // SOURCE: D. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints"
    //         commonly uses 0.8.
    float ratioThreshold = 0.8f;

    // Pixel threshold epsilon for correctness (default = 3px).
    float epsilonPx = 3.0f;
};

// Compute raw matches using BFMatcher and (optionally) Lowe's ratio test.
//
// INPUT:
//   descA, descB    : descriptor sets (N x D, N' x D), float.
//   config          : matching parameters.
//
// OUTPUT:
//   matches         : vector of MatchWithLabel with isCorrect = false initially;
//                     correctness labeling is done in a separate step.
//
std::vector<MatchWithLabel> matchDescriptors(const DescriptorSet& descA,
                                             const DescriptorSet& descB,
                                             const MatchingConfig& config);

// Label matches as correct/incorrect based on homography and reprojection error.
//
// INPUT:
//   descA, descB    : descriptor sets (only keypoints are used here).
//   H_AtoB          : homography (3x3, CV_64F) mapping A→B.
//   epsilonPx       : reprojection error threshold in pixels.
//
// OUTPUT:
//   matches         : modifies isCorrect field in-place for each match.
//
void labelMatchesWithHomography(const DescriptorSet& descA,
                                const DescriptorSet& descB,
                                const cv::Mat& H_AtoB,
                                float epsilonPx,
                                std::vector<MatchWithLabel>& matches);

